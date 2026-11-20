"""Files: Markov-chain configurations in and out, and training configs.

Chains: two formats of the same `(n_chains, 2N)` array of `+-1` occupations, spin-up modes first:

* `.npy` — int8 (what a run writes after every step and a continuation reads);
* `.npz` — packed for distribution, `bits = np.packbits(spins > 0, axis=-1)` and `shape = (n_chains, 2N)`, eight
  times smaller.

`load_chains` refuses a file with fewer chains than the run needs (of more it takes the first), or whose width or
any chain's `(N_up, N_dn)` differs from the run's: the sampler's moves conserve `N_up` and `N_dn`, so a chain born in
another sector would stay there for the whole run.

Training configs: one TOML file a training, listing its stages (jobs) in order (`load_config`).

    name = "hubbard-4x4-u8"
    [network]          nv, net_seed
    [lattice]          Lx, Ly, n_particles, t, U, tprime, mu      (the default; a stage may override any of them)
    [contraction]      dtype, threshold, delay_slots[, dmma]       (the default; a stage may override)
    [[stage]]          id, start ("random" or a stage id), symm, steps, samples, sweeps, thermal, chains
                       ("fresh" or "previous"), chains_thermal, cap = {c0, t0, decay}, cap_step_offset, max_parallel,
                       fused_matvec, eloc_precision, eloc_dual, clip, chain_seed, save_every, solver ("lsmr" or
                       "minsr_chunked", with shift_abs), exact (false: this package cannot run the stage
                       as it ran — see its note), and a free-text `note`

TOML is read by the standard library (`tomllib`), so a config needs no extra dependency.
"""
from __future__ import annotations

import copy
import os
import tomllib

import numpy as np

__all__ = ["pack", "unpack", "read_chains", "write_chains", "load_chains", "save_chains", "load_config",
           "config_stage", "STAGE_DEFAULTS"]


# ---- chains -------------------------------------------------------------------------------------------------------
def pack(spins):
    """`{"bits", "shape"}` of a `+-1` array."""
    spins = np.asarray(spins)
    return {"bits": np.packbits(spins > 0, axis=-1), "shape": np.asarray(spins.shape, np.int64)}


def unpack(bits, shape):
    """The int8 `+-1` array `pack` packed."""
    shape = tuple(int(k) for k in np.asarray(shape))
    occ = np.unpackbits(np.asarray(bits, np.uint8), axis=-1, count=shape[-1])
    return np.where(occ.reshape(shape) > 0, 1, -1).astype(np.int8)


def read_chains(path):
    """A chains file in either format, as int8 `+-1`."""
    path = str(path)
    if path.endswith(".npz"):
        with np.load(path) as z:
            return unpack(z["bits"], z["shape"])
    return np.load(path).astype(np.int8)


def write_chains(path, spins):
    """`spins` to `path` (`.npz`: packed, otherwise `.npy` int8), through a temporary file."""
    path = str(path)
    spins = np.asarray(spins).astype(np.int8)
    if path.endswith(".npz"):
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **pack(spins))
    else:
        tmp = path + ".tmp.npy"
        np.save(tmp, spins)
    os.replace(tmp, path)


def load_chains(path, n_samples=None):
    """Saved chains checked against quantax's current lattice: the width and every chain's `(N_up, N_dn)`; with
    `n_samples`, the first `n_samples` of them (the file must hold at least that many). Returns int8 `+-1`."""
    import quantax as qtx
    s = read_chains(path)
    sites = qtx.get_sites()
    n_want = s.shape[0] if n_samples is None else int(n_samples)
    if s.ndim != 2 or s.shape[1] != sites.Nmodes or s.shape[0] < n_want:
        raise ValueError(f"{path} holds {s.shape[0]} chains of {s.shape[-1]} modes; "
                         f"the run needs {n_want} of {sites.Nmodes}")
    s = s[:n_want]
    nup, ndn = sites.Nparticles
    n = sites.Nsites
    occ = s > 0
    bad = (occ[:, :n].sum(axis=1) != nup) | (occ[:, n:].sum(axis=1) != ndn)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} chains of {path} lie outside the sector ({nup}, {ndn})")
    return s


def save_chains(path, spins):
    """The chains' configurations, gathered from every process and written by the first."""
    import jax
    from quantax.utils import to_replicated_numpy
    s = np.asarray(to_replicated_numpy(spins)).astype(np.int8)
    if jax.process_index() == 0:
        write_chains(path, s)


# ---- training configs ---------------------------------------------------------------------------------------------
STAGE_DEFAULTS = dict(symm=None, chains="fresh", chains_thermal=-1, cap_step_offset=0, clip=5.0, fused_matvec=True,
                      eloc_precision="", eloc_dual=False, chain_seed=0, save_every=10, thermal=4096, solver="lsmr",
                      shift_abs=None,
                      exact=True)
_REQUIRED = ("id", "start", "steps", "samples", "sweeps", "cap", "max_parallel")
#: the keys `nngtn-train` reads from each table (a stage's lattice and contraction after its overrides)
_NEEDED = {"network": ("nv", "net_seed"), "lattice": ("Lx", "Ly", "n_particles", "t", "U", "tprime", "mu"),
           "contraction": ("dtype", "threshold", "delay_slots"), "cap": ("c0", "t0", "decay")}
#: every key a config may use (an unknown one is a typo or an option this package does not have, never ignored)
_KEYS = {
    "top": {"name", "network", "lattice", "contraction", "stage"},
    "network": {"nv", "net_seed"},
    "lattice": {"Lx", "Ly", "n_particles", "t", "U", "tprime", "mu"},
    "contraction": {"dtype", "threshold", "delay_slots", "dmma"},
    "stage": set(STAGE_DEFAULTS) | set(_REQUIRED) | {"lattice", "contraction", "note"},
    "cap": {"c0", "t0", "decay"},
}


def _check_keys(path, where, table, kind):
    unknown = sorted(set(table) - _KEYS[kind])
    if unknown:
        raise ValueError(f"{path}: {where} has unknown keys {unknown}; allowed: {sorted(_KEYS[kind])}")


def _check_needed(path, where, table, kind):
    missing = [k for k in _NEEDED[kind] if k not in table]
    if missing:
        raise ValueError(f"{path}: {where} lacks {missing}")


def load_config(path):
    """The config at `path`, every stage completed with the defaults and the state's lattice and contraction."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("name", "network", "lattice", "contraction", "stage"):
        if key not in cfg:
            raise ValueError(f"{path}: no `{key}`")
    _check_keys(path, "the config", cfg, "top")
    for kind in ("network", "lattice", "contraction"):
        _check_keys(path, f"[{kind}]", cfg[kind], kind)
    _check_needed(path, "[network]", cfg["network"], "network")
    ids = []
    stages = []
    for raw in cfg["stage"]:
        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"{path}: stage {raw.get('id', '?')} lacks {missing}")
        where = f"stage {raw.get('id', '?')}"
        _check_keys(path, where, raw, "stage")
        _check_keys(path, f"{where} cap", raw["cap"], "cap")
        for kind in ("lattice", "contraction"):
            _check_keys(path, f"{where} {kind}", raw.get(kind, {}), kind)
        st = {**STAGE_DEFAULTS, **copy.deepcopy(raw)}
        st["lattice"] = {**cfg["lattice"], **raw.get("lattice", {})}
        st["contraction"] = {**cfg["contraction"], **raw.get("contraction", {})}
        for kind, table in (("lattice", st["lattice"]), ("contraction", st["contraction"]), ("cap", st["cap"])):
            _check_needed(path, f"{where} {kind}", table, kind)
        if st["symm"] in ("", "none"):
            st["symm"] = None
        if st["start"] != "random" and st["start"] not in ids:
            raise ValueError(f"{path}: stage {st['id']} starts from {st['start']!r}, not an earlier stage")
        if st["chains"] not in ("fresh", "previous"):
            raise ValueError(f"{path}: stage {st['id']}: chains is 'fresh' or 'previous'")
        if st["chains"] == "previous" and st["start"] == "random":
            raise ValueError(f"{path}: stage {st['id']} reads previous chains but starts from random weights")
        ids.append(st["id"])
        stages.append(st)
    cfg["stage"] = stages
    return cfg


def config_stage(cfg, stage_id):
    for st in cfg["stage"]:
        if st["id"] == stage_id:
            return copy.deepcopy(st)
    raise KeyError(f"no stage {stage_id!r} in {cfg['name']}; stages: {[s['id'] for s in cfg['stage']]}")
