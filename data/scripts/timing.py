"""The paper's contraction timings with this package: Fig. C1 and Tables C1 and C2 (`amplitude`), and the FP32/FP64
comparison of Fig. C2 and Table C3 (`precision`). Fig. 2's GPEPS points are a separate run of the `amplitude`
protocol on the same card (within 1.3 % of Table C1's contraction times); its dense determinant and Pfaffian are
the fermix library's and are not timed here.

    uv run python -u scripts/timing.py amplitude --sizes 8,16,32,64 --out runs/timing        # Tables C1-C2, Fig. C1
    uv run python -u scripts/timing.py precision --sizes 8,16,32,64 --nv 3,4 --out runs/prec  # Table C3, Fig. C2

`amplitude`: the Nv = 3 network at its seed-0 initialisation (`build_model(nv=3, net_seed=0)`, the paper's "same
initial weights" at every size: equal to the weights the paper timed to float32 rounding), uniformly random
half-filled configurations (fixed seed), float32 contraction with the DMMA close, TF32 CNN, pivot threshold 0.05,
20 delay slots, no symmetry projection. The CNN, the contraction and the whole amplitude are timed separately; each
batch of the scan is timed `--rounds` times, each round the best of `--repeats` calls, after compilation and
warm-up; the paper quotes the fastest batch's median per amplitude.

`precision`: the contraction alone on synthetic site entries (Gaussian, standard deviation 0.5, rounded once to
float32) and random half-filled occupations, identical between the arms: float32 (DMMA recipe) and float64, at
Nv = 3 and 4, two-site leaves, threshold 0.05, 20 delay slots.

Every (size, precision) cell runs in its own process: a process builds one kernel library (`recipe.apply`). Each
cell writes `<out>-<mode>-L<L>-<dtype>[-nv<Nv>].json`. Run on an otherwise idle card; the paper's numbers
(`timing/amplitude.csv`, `timing/precision.csv`) are an A100-SXM4-80GB's on NERSC Perlmutter.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner="
                      "false --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def random_configurations(L, batch, seed=0):
    """`(batch, 2 L^2)` int8 +-1 in quantax's order (spin-up sites, then spin-down), half filling, uniformly at
    random — the paper's timing inputs."""
    import numpy as np
    N = L * L
    rng = np.random.default_rng(seed)
    s = -np.ones((batch, 2 * N), np.int8)
    for b in range(batch):
        s[b, rng.choice(N, N // 2, replace=False)] = 1
        s[b, N + rng.choice(N, N // 2, replace=False)] = 1
    return s


def synthetic_inputs(L, nv, batch, seed=0):
    """The precision table's inputs: site entries N(0, 0.5^2) rounded to float32, per-site occupation bits."""
    import numpy as np
    rng = np.random.default_rng(seed)
    m = 2 + 4 * nv
    free = (0.5 * rng.normal(size=(batch, L * L, m * (m - 1) // 2))).astype(np.float32)
    occ = np.zeros((batch, 2 * L * L), np.int32)
    for row in occ:
        row[2 * rng.choice(L * L, L * L // 2, replace=False)] = 1
        row[2 * rng.choice(L * L, L * L // 2, replace=False) + 1] = 1
    return free, occ


def best(fn, arg, rounds, repeats):
    """Per round, the fastest of `repeats` calls (seconds)."""
    import jax
    out = []
    for _ in range(rounds):
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            jax.block_until_ready(fn(arg))
            ts.append(time.perf_counter() - t)
        out.append(min(ts))
    return out


def summary(rounds, batch):
    import numpy as np
    return dict(round_min_seconds=rounds, median_us=float(np.median(rounds) / batch * 1e6),
                min_us=float(np.min(rounds) / batch * 1e6), max_us=float(np.max(rounds) / batch * 1e6))


def default_batches(mode, L):
    if mode == "amplitude":
        return [256, 1024, 2048, 4096, 8192, 16384] if L <= 12 else [256, 1024, 2048] if L <= 24 else \
            [128, 512, 1024] if L <= 32 else [32, 128, 256, 512]
    return [2048, 8192] if L <= 8 else [512, 2048] if L <= 16 else [256, 1024] if L <= 32 else [128, 512]


def cell(args):
    """One (mode, L, dtype[, nv]) in this process."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    import nngtn as ng
    from nngtn import model as mdl
    L, dtype = args.L, args.dtype
    dmma = dtype == "float32" and not args.no_dmma
    nv = args.nv_one
    row = dict(mode=args.mode, L=L, N=L * L, dtype=dtype, nv=nv, dmma=dmma, threshold=args.threshold,
               delay_slots=args.delay_slots, seed=args.seed, rounds=args.rounds, repeats=args.repeats,
               device=jax.devices()[0].device_kind, jax_version=jax.__version__, batches=[])
    ng.make_sites(L, L, (L * L // 2, L * L // 2))
    model = ng.build_model(L, L, nv=nv, net_seed=0, contraction_dtype=dtype, threshold=args.threshold,
                           delay_slots=args.delay_slots, dmma=dmma)
    row["settings"] = {k: v for k, v in os.environ.items() if k.startswith(("PIVOTED_", "NNGTN_"))}
    if args.mode == "amplitude":
        cnn = jax.jit(jax.vmap(model.free_vectors))
        config = jax.vmap(lambda s: mdl.config_of_spins(s, model.n_sites, model.d))
        contraction = jax.jit(jax.vmap(lambda c, f: model.log_psi(c, f)))
        whole = jax.jit(jax.vmap(model))
    else:
        f = mdl._contraction(L, L, nv, 2, args.threshold, args.delay_slots, 2)
        contraction = jax.jit(jax.vmap(lambda x, o: f(x, o)[:2]))
    dt = jnp.dtype(dtype)
    for B in (list(map(int, args.batches.split(","))) if args.batches else default_batches(args.mode, L)):
        try:
            t0 = time.perf_counter()
            if args.mode == "amplitude":
                s = jnp.asarray(random_configurations(L, B, args.seed))
                free = jax.block_until_ready(cnn(s)).astype(jnp.float64)
                conf = jax.block_until_ready(config(s))
                logpsi = jax.block_until_ready(contraction(conf, free))
                jax.block_until_ready(whole(s))
                finite = float(np.mean(np.isfinite(np.asarray(logpsi.real))))
                parts = [("whole", whole, s), ("cnn", cnn, s), ("contraction", lambda p: contraction(*p), (conf, free))]
            else:
                free, occ = synthetic_inputs(L, nv, B, args.seed)
                pair = (jnp.asarray(free).astype(dt), jnp.asarray(occ))
                sign, logabs = jax.block_until_ready(contraction(*pair))
                finite = float(np.mean(np.isfinite(np.asarray(logabs)) & (np.asarray(sign) != 0)))
                parts = [("contraction", lambda p: contraction(*p), pair)]
            obs = dict(batch=B, compile_and_warm_seconds=time.perf_counter() - t0, finite_fraction=finite,
                       components={name: summary(best(fn, a, args.rounds, args.repeats), B) for name, fn, a in parts})
        except Exception as e:      # a batch the card cannot hold: recorded, the scan goes on
            obs = dict(batch=B, error=f"{type(e).__name__}: {str(e)[:300]}")
        row["batches"].append(obs)
        print(json.dumps(obs), flush=True)
    tag = f"-nv{nv}" if args.mode == "precision" else ""
    path = Path(f"{args.out}-{args.mode}-L{L}-{dtype}{tag}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1) + "\n")
    print("wrote", path, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("amplitude", "precision"))
    ap.add_argument("--sizes", default="8,16,32,64")
    ap.add_argument("--dtypes", default=None, help="default: float32 (amplitude); float32,float64 (precision)")
    ap.add_argument("--nv", default="3", help="networks for `precision`, e.g. 3,4 (amplitude: 3)")
    ap.add_argument("--batches", default="", help="comma list (default: a scan per size)")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--delay-slots", dest="delay_slots", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dmma", dest="no_dmma", action="store_true", help="float32 without the DMMA close")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--L", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--dtype", help=argparse.SUPPRESS)
    ap.add_argument("--nv-one", dest="nv_one", type=int, default=3, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.L is not None:
        return cell(args)
    dtypes = (args.dtypes or ("float32" if args.mode == "amplitude" else "float32,float64")).split(",")
    nvs = [3] if args.mode == "amplitude" else [int(x) for x in args.nv.split(",")]
    for L in map(int, args.sizes.split(",")):
        for nv in nvs:
            for dtype in dtypes:
                cmd = [sys.executable, "-u", __file__, args.mode, "--L", str(L), "--dtype", dtype, "--nv-one", str(nv),
                       "--threshold", str(args.threshold), "--delay-slots", str(args.delay_slots),
                       "--rounds", str(args.rounds), "--repeats", str(args.repeats), "--seed", str(args.seed),
                       "--out", args.out] + (["--batches", args.batches] if args.batches else []) + \
                      (["--no-dmma"] if args.no_dmma else [])
                print(f"== {args.mode} L={L} nv={nv} {dtype}", flush=True)
                r = subprocess.run(cmd)
                if r.returncode:
                    print(f"cell failed: L={L} nv={nv} {dtype} (exit {r.returncode})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
