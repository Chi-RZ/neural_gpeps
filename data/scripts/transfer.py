"""The 32x32 transfer reads (the paper's Table II and Table E1) from their per-reading table, no GPU.

    uv run python scripts/transfer.py            # print, and compare with measurements/transfer32x32/results.json
    uv run python scripts/transfer.py --write    # (re)write measurements/transfer32x32/results.json

Each arm is one read of an unprojected 16x16 state on the 32x32 torus at exactly its weights: 1024 samples a
reading, 64 readings (32 for the projected arms). A reading's first and second moments are pooled with equal mass
(the readings have equal sample counts; a reweighted reading's weights are normalised within it), and the energy
and the variance per site are those of the pooled moments. Their error is the largest delete-block jackknife
standard error over blocks of 1, 2, 4 and 8 readings, among the widths that leave at least eight blocks.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "measurements" / "transfer32x32" / "readings.csv"
RESULTS = ROOT / "measurements" / "transfer32x32" / "results.json"
SITES = 1024


def read_table(path=TABLE):
    """`{arm: {"state", "projection", "sampling", "estimator", "E", "var"}}` in the table's order."""
    arms = {}
    with open(path) as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            a = arms.setdefault(row["arm"], {k: row[k] for k in ("state", "projection", "sampling", "estimator")}
                                | {"reading": [], "E": [], "var": []})
            a["reading"].append(int(row["reading"]))
            a["E"].append(float(row["E"]))
            a["var"].append(float(row["var"]))
    for a in arms.values():
        assert a["reading"] == list(range(len(a["reading"]))), "readings out of order"
    return arms


def reduce(E, var, sites=SITES, widths=(1, 2, 4, 8)):
    """Energy and variance per site of equal-mass pooled moments, each with its largest block-jackknife error."""
    E, var = np.asarray(E, np.float64), np.asarray(var, np.float64)
    origin = float(E.mean())
    c = E - origin
    sums = np.column_stack([np.ones_like(c), c, var + c ** 2])

    def estimate(total):
        mean = total[..., 1] / total[..., 0]
        return np.stack([(mean + origin) / sites, (total[..., 2] / total[..., 0] - mean ** 2) / sites], axis=-1)

    value = estimate(sums.sum(0))
    errors = {}
    for w in widths:
        blocks = len(sums) // w
        if blocks < 8:
            continue
        grouped = sums[:blocks * w].reshape(blocks, w, 3).sum(1)
        replicates = estimate(grouped.sum(0) - grouped)
        errors[w] = np.sqrt(blocks - 1) * replicates.std(axis=0)
    err = np.max(np.stack(list(errors.values())), axis=0)
    return {"E_per_site": float(value[0]), "E_err": float(err[0]), "var_per_site": float(value[1]),
            "var_err": float(err[1]), "readings": int(len(E)), "block_widths": [int(w) for w in errors]}


def results(arms):
    out = {}
    for name, a in arms.items():
        out[name] = {k: a[k] for k in ("state", "projection", "sampling", "estimator")} | reduce(a["E"], a["var"])
    return out


def agree(a, b, rtol=1e-10):
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, dict):
            if not agree(x, y, rtol):
                return False
        elif isinstance(x, float) and isinstance(y, float):
            if not (x == y or abs(x - y) <= rtol * max(abs(x), abs(y))):
                return False
        elif x != y:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write measurements/transfer32x32/results.json")
    args = ap.parse_args()
    res = results(read_table())
    for name, r in res.items():
        print(f"{name:10s} {r['state']:19s} {r['projection'][:5]:5s} {r['sampling']:7s} {r['estimator']:18s} "
              f"E/N {r['E_per_site']:.6f} ({r['E_err']:.1e})  var/N {r['var_per_site']:.5f} ({r['var_err']:.1e})")
    if args.write:
        RESULTS.write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {RESULTS}")
    elif RESULTS.exists():
        same = agree(json.loads(json.dumps(res)), json.loads(RESULTS.read_text()))
        print("results.json " + ("reproduced (within 1e-10 relative)" if same else "DIFFERS"))
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
