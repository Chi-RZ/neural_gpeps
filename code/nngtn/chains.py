"""Markov-chain configurations in and out.

Two formats of the same `(n_chains, 2N)` array of `+-1` occupations, spin-up modes first:

* `.npy` — int8 (what a run writes after every step and a continuation reads);
* `.npz` — packed for distribution, `bits = np.packbits(spins > 0, axis=-1)` and `shape = (n_chains, 2N)`, eight
  times smaller.

`load_chains` refuses a file with fewer chains than the run needs (of more it takes the first), or whose width or
any chain's `(N_up, N_dn)` differs from the run's: the
sampler's moves conserve `N_up` and `N_dn`, so a chain born in another sector would stay there for the whole run.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["pack", "unpack", "read_chains", "write_chains", "load_chains", "save_chains"]


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
