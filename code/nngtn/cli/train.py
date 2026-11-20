"""Run one stage of a training config.

    nngtn-train config.toml --stage <id> --out runs/x          (or: python -m nngtn.cli.train ...)

A config (`nngtn.config`) lists the stages of a training, in order: each with its lattice, its start (random
weights or the weights an earlier stage ended with), its chains, its step-norm schedule, its samples and sweeps and its
contraction. A stage reads its start from `<out>/<start stage>/weights.eqx` (or `--start`) and, when
`chains = "previous"`, its chains from `<out>/<start stage>/chains.npy` (or `--chains-in`), and writes
`<out>/<stage>/trajectory.json` (every step), `weights.eqx` (every `save_every` steps and at the end) and
`chains.npy` (every step).

Overrides for a smoke test (`--steps`, `--samples`, `--sweeps`, `--thermal`, `--chains-thermal`, `--save-every`,
`--max-parallel`)
are recorded in the trajectory. Several GPUs and nodes: launch one process per node under `srun`
(`nngtn.distributed`); `samples` is the total over all cards and `max_parallel` is per card.

Unless `XLA_FLAGS` is set, the XLA flags below are set before jax starts: the autotuners and command buffers off,
so XLA makes the same choices every run; bit-identical runs additionally need `--xla_gpu_deterministic_ops=true`
(see the README).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

XLA_FLAGS = ("--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false "
             "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_FLAGS", XLA_FLAGS)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from .. import config as cfgmod  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True, help="the stage id (`id` in the config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=None, help="start weights (default: the start stage's output)")
    ap.add_argument("--chains-in", dest="chains_in", default=None,
                    help="start chains (default: the start stage's output when the stage says chains = 'previous')")
    for k in ("steps", "samples", "sweeps", "thermal", "chains_thermal", "save_every"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=int, default=None)
    ap.add_argument("--max-parallel", dest="max_parallel", type=int, nargs="+", default=None)
    ap.add_argument("--card", choices=("hopper", "a100"), default=None, help="kernel recipe (default: detected)")
    args = ap.parse_args(argv)

    from .. import distributed
    distributed.initialize()
    import jax
    import nngtn as ng

    config = cfgmod.load(args.config)
    st = cfgmod.stage(config, args.stage)
    overrides = {k: getattr(args, k) for k in ("steps", "samples", "sweeps", "thermal", "chains_thermal",
                                               "save_every", "max_parallel") if getattr(args, k) is not None}
    st = {**st, **overrides}
    out = Path(args.out)
    here = out / st["id"]
    start = args.start
    if start is None and st["start"] != "random":
        start = str(out / st["start"] / "weights.eqx")
    chains_in = args.chains_in
    if chains_in is None and st["chains"] == "previous":
        chains_in = str(out / st["start"] / "chains.npy")
    writer = distributed.is_writer()
    if writer:
        here.mkdir(parents=True, exist_ok=True)

    lat, con = st["lattice"], st["contraction"]
    ng.make_sites(lat["Lx"], lat["Ly"], lat["n_particles"])
    model = ng.build_model(lat["Lx"], lat["Ly"], nv=config["network"]["nv"], net_seed=config["network"]["net_seed"],
                           contraction_dtype=con["dtype"], threshold=con["threshold"],
                           delay_slots=con["delay_slots"], card=args.card, dmma=bool(con.get("dmma", False)))
    if start is not None:
        model = ng.load_weights(model, start)
    cap = st["cap"]
    print(f"{jax.devices()[0].device_kind} x {jax.device_count()} over {jax.process_count()} processes; "
          f"stage {st['id']} of {config['name']}: {lat['Lx']}x{lat['Ly']} N={tuple(lat['n_particles'])} "
          f"t'={lat['tprime']:g} U={lat['U']:g}, symm={st['symm']}, start={start or 'random'}, "
          f"chains={chains_in or 'fresh'}, contraction {con}, cap {cap}, offset {st['cap_step_offset']}",
          flush=True)
    record = dict(config=config["name"], stage=st, overrides=overrides, start=start, chains_in=chains_in,
                  devices=[str(d) for d in jax.devices()], xla_flags=os.environ.get("XLA_FLAGS"),
                  package=ng.__version__, rows=[])
    t0 = time.perf_counter()
    n = lat["Lx"] * lat["Ly"]

    def on_step(row, model, sampler):
        record["rows"].append(row)
        last = row["step"] == st["steps"] - 1
        if row["step"] % st["save_every"] == 0 or last:
            print(f"  step {row['step']:5d}  E/N = {row['energy'] / n:+.6f} +- {row['energy_err'] / n:.6f}  "
                  f"VarE/N = {row['var_e'] / n:.4e}  |step| = {row['step_norm']:.3e}  {row['seconds']:.2f} s",
                  flush=True)
            if writer:
                ng.save_weights(model, str(here / "weights.eqx"))
        if writer:
            record["elapsed_s"] = time.perf_counter() - t0
            (here / "trajectory.json").write_text(json.dumps(record, indent=1))

    if not st["exact"]:
        print(f"NOTE: stage {st['id']} cannot be run exactly as it ran: {st.get('note', '')}", flush=True)
    rows, model = ng.run(model, n_particles=lat["n_particles"], t=lat["t"], u=lat["U"], tprime=lat["tprime"],
                         mu=lat["mu"], symm=st["symm"], steps=st["steps"], n_samples=st["samples"],
                         thermal=st["thermal"], sweeps=st["sweeps"], chain_seed=st["chain_seed"],
                         chains_in=chains_in, chains_thermal=st["chains_thermal"],
                         chains_out=str(here / "chains.npy"), max_parallel=st["max_parallel"],
                         cap_c0=cap["c0"], cap_t0=cap["t0"], cap_decay=cap["decay"],
                         cap_step_offset=st["cap_step_offset"], clip=st["clip"], solver=st["solver"],
                         shift_abs=st["shift_abs"],
                         fused_matvec=st["fused_matvec"], eloc_precision=st["eloc_precision"] or None,
                         eloc_dual=bool(st["eloc_dual"]),
                         on_step=on_step)
    if writer:
        ng.save_weights(model, str(here / "weights.eqx"))
    print(f"wrote {here}; wall {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
