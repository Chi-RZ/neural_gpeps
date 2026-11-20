"""The data repository's tests. CPU: every shipped file matches its sha256 in `states.json`; each measurement's
`results.json` is reproduced from its series (to rounding: 1e-10 relative) by `scripts/assemble.py`; `states.json` quotes `results.json`;
the chains are in the state's particle sector; every measurement config is nngtn-measure input as it stands; the 32x32
transfer reads' results are reproduced from their readings by `scripts/transfer.py`."""
import json
import os
import sys

# the production runs' XLA flags, under which the golden values were computed; set before jax starts
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner="
                      "false --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assemble as _states  # noqa: E402  (the index helpers live in scripts/assemble.py)

MEASURED = [r["name"] for r in _states.index()["states"] if r.get("files")]


@pytest.mark.parametrize("name", MEASURED)
def test_files_match_their_checksums(name):
    bad = [str(p) for k, p, ok in _states.verify(_states.state(name)) if not ok]
    assert not bad, f"missing or changed: {bad}"


@pytest.mark.parametrize("name", [n for n in MEASURED if "measurement_results" in _states.state(n)["files"]])
def test_assembly_reproduces_results(name):
    from assemble import assemble
    row, scalars, _ = assemble(name)
    published = json.loads(_states.path(row, "measurement_results").read_text())
    from assemble import agree
    assert agree(json.loads(json.dumps(scalars)), published)
    # states.json quotes E/N to 7 decimals and its errors to 2 significant figures
    assert abs(row["energy"]["E_per_site"] - published["E_per_site"]) <= 5e-8
    assert abs(row["energy"]["E_err"] - published["E_err"]) <= 0.05 * published["E_err"]
    for k, v in row["order_parameters"].items():
        assert v == published[k], k


@pytest.mark.parametrize("name", MEASURED)
def test_chains_in_sector(name):
    import nngtn as ng
    row = _states.state(name)
    s = ng.read_chains(str(_states.path(row, "chains")))
    lat = row["lattice"]
    n = lat["Lx"] * lat["Ly"]
    assert s.shape == (row["chains"]["n_chains"], 2 * n)
    occ = s > 0
    assert (occ[:, :n].sum(1) == lat["n_particles"][0]).all() and (occ[:, n:].sum(1) == lat["n_particles"][1]).all()


@pytest.mark.parametrize("path", sorted(ROOT.glob("measurements/*/config.toml")), ids=lambda p: p.parent.name)
def test_measurement_config_runs_as_is(path, monkeypatch):
    """Every measurement config is `nngtn-measure --config` input as it stands (run from data/): it parses with the
    command's own options, and its weights and start chains exist."""
    from nngtn.cli.measure import parse
    monkeypatch.chdir(ROOT)
    args = parse(["--config", str(path.relative_to(ROOT)), "--out", "unused"])
    assert (ROOT / args.weights).exists() and (ROOT / args.chains_in).exists()
    assert args.nsamples == 8192 and args.nsweeps == 64


def test_transfer_files_match_their_checksums():
    section = _states.index()["transfer32x32"]
    bad = [f["path"] for f in section["files"].values() if _states.sha256(ROOT / f["path"]) != f["sha256"]]
    assert not bad, f"changed: {bad}"


def test_transfer_results_are_reproduced_and_quoted():
    """scripts/transfer.py reproduces measurements/transfer32x32/results.json from readings.csv; states.json quotes it; the
    projected arms have 32 readings, the others 64."""
    import transfer
    res = json.loads(json.dumps(transfer.results(transfer.read_table())))
    published = json.loads((ROOT / "measurements" / "transfer32x32" / "results.json").read_text())
    assert transfer.agree(res, published)
    quoted = _states.index()["transfer32x32"]["arms"]
    for arm, r in published.items():
        assert r["readings"] == (32 if arm.endswith("_proj") else 64)
        assert all(quoted[arm][k] == r[k] for k in quoted[arm])


@pytest.mark.parametrize("path", sorted(ROOT.glob("measurements/transfer32x32/*.toml")), ids=lambda p: p.stem)
def test_transfer_config_runs_as_is(path, monkeypatch):
    """Every 32x32 read config is `nngtn-measure --config` input as it stands, from random chains, energy only."""
    from nngtn.cli.measure import parse
    monkeypatch.chdir(ROOT)
    args = parse(["--config", str(path.relative_to(ROOT)), "--out", "unused"])
    assert (ROOT / args.weights).exists() and args.chains_in is None
    assert (args.Lx, args.Ly, args.nsamples, args.thermal, args.sweeps, args.delay_slots) == (32, 32, 1024, 40960, 4096, 64)
    assert args.nsweeps == (32 if path.stem.endswith("_proj") else 64) and args.channels == "energy"
