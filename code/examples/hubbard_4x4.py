"""A usage demo: the 4x4 periodic Hubbard model at U = 8 and half filling (8 up + 8 down), trained from random
weights with the production network (Nv = 3, 2.4 M parameters) for 100 steps.

    uv run python -u examples/hubbard_4x4.py [--steps 100] [--samples 4096] [--out runs/4x4]

A demonstration of the package, not a converged ground state.

Prints the sampled energy and writes `<out>/trajectory.json` and `<out>/model.eqx` (the current weights) every 10
steps and at the last one. The exact ground-state energy is -8.468875 (-0.529305 a site).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# the original runs' XLA flags: the autotuners and command buffers off
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false "
                                   "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXACT = -8.468875


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--nparticles", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--sweeps", type=int, default=256)
    ap.add_argument("--thermal", type=int, default=4096)
    ap.add_argument("--max-parallel", type=int, nargs="+", default=[2048, 256], help="forward [backward] batch a device")
    ap.add_argument("--cap", type=float, nargs=3, default=(0.5, 1000, 1500.0), metavar=("C0", "T0", "DECAY"),
                    help="step-norm ceiling c0 / (1 + max(step - t0, 0) / decay)")
    ap.add_argument("--nv", type=int, default=3)
    ap.add_argument("--nblocks", type=int, default=None, help="default: the preset for --nv")
    ap.add_argument("--channels", type=int, default=None, help="default: the preset for --nv")
    ap.add_argument("--pad-channels", type=int, default=None, help="default: the preset for --nv; 0: no padding")
    ap.add_argument("--contraction", default="float32", choices=("float32", "float64"))
    ap.add_argument("--net-seed", type=int, default=0)
    ap.add_argument("--chain-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/4x4")
    args = ap.parse_args()

    import jax
    import nngtn as ng
    ng.make_sites(args.L, args.L, args.nparticles)
    model = ng.build_model(args.L, args.L, nv=args.nv, nblocks=args.nblocks, channels=args.channels,
                           pad_channels=args.pad_channels, net_seed=args.net_seed,
                           contraction_dtype=args.contraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.L * args.L
    size = dict(ng.NETWORKS.get(args.nv, {}))
    size.update({k: v for k, v in (("nblocks", args.nblocks), ("channels", args.channels)) if v is not None})
    print(f"{jax.devices()[0].device_kind}: {args.L}x{args.L} Hubbard U={args.U:g} N=({args.nparticles[0]},"
          f"{args.nparticles[1]}), Nv={args.nv}, {size.get('nblocks')} blocks x {size.get('channels')} channels, "
          f"{args.contraction} contraction, {args.samples} samples", flush=True)
    rows, t0 = [], time.perf_counter()

    def on_step(row, model, sampler):
        rows.append(row)
        if row["step"] % 10 == 0 or row["step"] == args.steps - 1:
            print(f"  step {row['step']:5d}  E/N = {row['energy'] / n:+.6f} +- {row['energy_err'] / n:.6f}  "
                  f"VarE/N = {row['var_e'] / n:.4e}  |step| = {row['step_norm']:.3e}  {row['seconds']:.2f} s",
                  flush=True)
            ng.save_weights(model, str(out / "model.eqx"))
            (out / "trajectory.json").write_text(json.dumps({"args": vars(args), "rows": rows}))

    rows_, model = ng.run(model, n_particles=args.nparticles, u=args.U, steps=args.steps, n_samples=args.samples,
                          thermal=args.thermal, sweeps=args.sweeps, chain_seed=args.chain_seed,
                          max_parallel=tuple(args.max_parallel), cap_c0=args.cap[0], cap_t0=args.cap[1],
                          cap_decay=args.cap[2], on_step=on_step)
    last = rows[-max(1, len(rows) // 10):]     # the last tenth of the run
    e = sum(r["energy"] for r in last) / len(last)
    print(f"mean E over the last {len(last)} steps: {e:.6f} ({e / n:.6f} a site); "
          f"relative error {abs((e - EXACT) / EXACT):.2e} against the exact {EXACT}"
          if (args.L, args.U, tuple(args.nparticles)) == (4, 8.0, (8, 8)) else f"mean E over the last steps: {e:.6f}")
    print(f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
