"""A measurement's results from its per-sweep series, no GPU: block-jackknife over sweeps of the energy and variance,
the d-wave pair correlation Delta^2_SC and the stripe spin order M^2_str (`nngtn.obs.assemble`).

    uv run python scripts/assemble.py proj16x4-t0-nv4               # print, and compare with results.json
    uv run python scripts/assemble.py proj16x4-t0-nv4 --write       # (re)write results.json
    uv run python scripts/assemble.py proj16x4-t0-nv4 --maps maps.npz   # also C_p, C_s, C_c and errors
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(rel):
    """A data path, relative to data/."""
    return ROOT / str(rel)


def index():
    return json.loads((ROOT / "states.json").read_text())


def state(name):
    for row in index()["states"]:
        if row["name"] == name:
            return row
    raise SystemExit(f"no state {name!r}; states: {[r['name'] for r in index()['states']]}")


def path(row, key):
    """The file `key` of a state (`weights`, `chains`, `training`, `measurement_config`, ...)."""
    files = row.get("files") or {}
    if key not in files:
        raise SystemExit(f"state {row['name']} ships no {key!r} ({row.get('status', '')})")
    return resolve(files[key]["path"])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(row):
    """`[(key, path, ok)]` of every file of a state against its recorded sha256."""
    out = []
    for key, f in (row.get("files") or {}).items():
        p = resolve(f["path"])
        out.append((key, p, p.exists() and sha256(p) == f["sha256"]))
    return out


SCALARS = ("nsweeps", "E_per_site", "E_err", "var_per_site", "var_err", "Delta2_SC", "Delta2_SC_err", "Delta_SC",
           "r_max", "n_far", "M2_str", "M2_str_err")


def agree(a, b, rtol=1e-10):
    """Two results dicts equal up to floating-point rounding: every float to `rtol` relative (the CPU's summation
    order differs between architectures — x86-64 and aarch64 differ in the last digits), everything else exactly."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or abs(x - y) <= rtol * max(abs(x), abs(y))):
                return False
        elif x != y:
            return False
    return True


def assemble(name):
    import numpy as np
    from nngtn import obs
    row = state(name)
    series = dict(np.load(path(row, "measurement_series")))
    Lx, Ly = int(series["Lx"]), int(series["Ly"])
    res = obs.assemble(series, Lx, Ly, tprime=float(series["tprime"]))
    scalars = {k: (float(res[k]) if k not in ("nsweeps", "n_far") else int(res[k])) for k in SCALARS}
    scalars["k_alpha"] = np.asarray(res["k_alpha"]).tolist()
    scalars["q_c"] = np.asarray(res["q_c"]).tolist()
    scalars["excluded_sweeps"] = np.asarray(series["excluded"]).tolist()
    for k in ("E_excluded", "dwave_nonfinite", "spin_nonfinite"):
        if k in series:
            scalars[k + "_total"] = int(np.sum(series[k]))
    maps = {k: v for k, v in res.items() if isinstance(v, np.ndarray) and v.ndim and k not in ("k_alpha", "q_c")}
    return row, scalars, maps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("state")
    ap.add_argument("--write", action="store_true", help="write results.json")
    ap.add_argument("--maps", default=None, help="write the correlation maps to this .npz")
    args = ap.parse_args()
    row, scalars, maps = assemble(args.state)
    print(json.dumps(scalars, indent=1))
    target = path(row, "measurement_results")
    if args.write:
        target.write_text(json.dumps(scalars, indent=1) + "\n")
        print(f"wrote {target}")
    elif target.exists():
        old = json.loads(target.read_text())
        new = json.loads(json.dumps(scalars))
        exact = old == new
        same = exact or agree(old, new)
        print("results.json " + ("reproduced exactly" if exact else
                                  "reproduced to rounding (CPU summation order; within 1e-10 relative)" if same
                                  else "DIFFERS"))
        if not same:
            return 1
    if args.maps:
        import numpy as np
        np.savez(args.maps, **maps)
        print(f"wrote {args.maps}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")    # no GPU needed; set only when run as a script
    sys.exit(main())
