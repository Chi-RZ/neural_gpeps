"""Measure a state with fixed weights: energy, variance, d-wave pair correlations, stripe spin and charge order.

    nngtn-measure --config measurement.toml --weights state.eqx --chains-in chains.npz --out runs/m/x
    nngtn-measure --weights w.eqx --Lx 16 --Ly 16 --nparticles 112 112 --tprime -0.2 \\
        --symm flip0@flip1@spininverse --chains-in chains.npy --nsamples 8192 --nsweeps 64 --out runs/m/x

With `--reread-series <series.npz>` it measures nothing: the series' saved non-finite configurations are re-read
with `--reread-slots` (64) delay slots against the `--delay-slots` the series was measured with, and the re-read
series, `<out>-reread.json` (the changes, sweep by sweep, and both assemblies) and its results are written. The
re-read is a diagnostic of how far psi-tilde is from the wider-slot state, not a variational estimate: its samples
come from the measured state while the re-read terms use more slots. The measurement's own values are psi-tilde's.

A `--config` TOML file supplies defaults for every option below (same names, `_` for `-`); the command line
overrides it. `--precision highest` (the default) runs every matmul at full float32 precision (no TF32).
Writes `<out>-series.npz`, `<out>-chains.npy`, `<out>.json` and `<out>-maps.npz` (`nngtn.measure`). One process
per node under `srun` for several GPUs or nodes; process 0 writes.
"""
import argparse
import os
import sys
import tomllib
from pathlib import Path

XLA_FLAGS = ("--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false "
             "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_FLAGS", XLA_FLAGS)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        with open(known.config, "rb") as f:
            defaults = tomllib.load(f)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], parents=[pre])
    ap.add_argument("--weights", required="weights" not in defaults)
    ap.add_argument("--out", required="out" not in defaults, help="output prefix")
    ap.add_argument("--Lx", type=int, default=16)
    ap.add_argument("--Ly", type=int, default=16)
    ap.add_argument("--nparticles", type=int, nargs=2, default=(112, 112))
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--tprime", type=float, default=0.0)
    ap.add_argument("--nv", type=int, default=3)
    ap.add_argument("--symm", default=None)
    ap.add_argument("--chains-in", dest="chains_in", default=None, help=".npy or packed .npz chains to start from")
    ap.add_argument("--nsamples", type=int, default=8192)
    ap.add_argument("--nsweeps", type=int, default=64)
    ap.add_argument("--thermal", type=int, default=10240, help="burn-in proposals a chain")
    ap.add_argument("--sweeps", type=int, default=1024, help="proposals a chain between sweeps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-parallel", dest="max_parallel", type=int, default=128)
    ap.add_argument("--nref", type=int, default=8)
    ap.add_argument("--channels", default="energy,dwave,spin,charge")
    ap.add_argument("--save-every", dest="save_every", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"), help="the contraction's scalar")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--delay-slots", dest="delay_slots", type=int, default=20)
    ap.add_argument("--precision", default="highest", choices=("highest", "default"),
                    help="matmul precision of the whole measurement (highest: no TF32)")
    ap.add_argument("--card", choices=("hopper", "a100"), default=None)
    ap.add_argument("--dmma", action="store_true",
                    help="float32 contraction with the global close on the fp64 tensor cores (off by default)")
    ap.add_argument("--reread-series", dest="reread_series", default=None,
                    help="instead of measuring: re-read this series' saved non-finite configurations with "
                         "--reread-slots delay slots (the series was measured with --delay-slots)")
    ap.add_argument("--reread-slots", dest="reread_slots", type=int, default=64)
    unknown = set(defaults) - {a.dest for a in ap._actions} - {"name", "note"}
    if unknown:
        raise SystemExit(f"{known.config}: unknown keys {sorted(unknown)}")
    ap.set_defaults(**{k: v for k, v in defaults.items() if k not in ("name", "note")})
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    from .. import measure
    from . import distributed
    distributed.initialize(quiet_followers=False)
    measure.set_matmul_precision(None if args.precision == "default" else args.precision)
    import nngtn as ng
    ng.make_sites(args.Lx, args.Ly, args.nparticles)
    model = ng.load_weights(ng.build_model(args.Lx, args.Ly, nv=args.nv, contraction_dtype=args.dtype,
                                           threshold=args.threshold, delay_slots=args.delay_slots, card=args.card,
                                           dmma=args.dmma),
                            args.weights)
    state = ng.make_state(model, symm=args.symm, max_parallel=args.max_parallel)
    H = ng.hamiltonian(args.Lx, args.Ly, t=1.0, u=args.U, tprime=args.tprime)
    if args.reread_series:
        return _reread(args, ng, measure, model, state, H)
    extra = dict(weights=args.weights, symm=str(args.symm), nv=args.nv, contraction=args.dtype,
                 threshold=args.threshold, delay_slots=args.delay_slots, precision=args.precision, U=args.U)
    measure.measure(state, H, args.out, nparticles=tuple(args.nparticles), nsamples=args.nsamples,
                    nsweeps=args.nsweeps, thermal=args.thermal, sweeps=args.sweeps, seed=args.seed,
                    chains_in=args.chains_in, tprime=args.tprime, nref=args.nref,
                    channels=args.channels.split(","), save_every=args.save_every, extra=extra)
    return 0


def _reread(args, ng, measure, model, state, H):
    """`--reread-series`: the re-read series (a diagnostic, see the module docstring), the report and its assembled
    results under `--out`."""
    import json

    import jax
    import numpy as np

    from .. import obs
    with np.load(args.reread_series, allow_pickle=False) as z:
        series = {k: z[k] for k in z.files}
    more = ng.make_state(model.with_contraction(delay_slots=args.reread_slots), symm=args.symm,
                         max_parallel=args.max_parallel)
    out, report = measure.reread(state, more, H, series)
    if jax.process_index() != 0:
        return 0
    report.update(series=args.reread_series, slots=args.delay_slots, reread_slots=args.reread_slots)
    tp = float(series.get("tprime", args.tprime))
    before = obs.assemble(series, args.Lx, args.Ly, tprime=tp)
    after = obs.assemble(out, args.Lx, args.Ly, tprime=tp)
    keys = ("E_per_site", "E_err", "var_per_site", "var_err", "Delta2_SC", "Delta2_SC_err", "M2_str", "M2_str_err")
    report["assembled"] = {k: {"measured": float(before[k]), "reread": float(after[k])}
                           for k in keys if k in before and k in after}
    Path(f"{args.out}-series.npz").parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{args.out}-series.npz", **out)
    np.savez(f"{args.out}-maps.npz", **{k: v for k, v in after.items() if isinstance(v, np.ndarray)})
    Path(f"{args.out}.json").write_text(json.dumps({k: v for k, v in after.items() if not isinstance(v, np.ndarray)},
                                                    indent=1, default=str))
    Path(f"{args.out}-reread.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["assembled"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
