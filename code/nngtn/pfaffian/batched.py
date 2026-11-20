"""The parity of the pivot sequence on the device (`_parity_modes`): `fast`'s sign when the parity kernel is not
taken.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _parity_modes(sequence, alive):
    """The parity of a pivot sequence: the valid entries of `sequence` (-1 = absent) are the modes pivoted, in
    order, and `alive` (n_modes,) marks the modes that can appear — a complete elimination pivots every alive mode
    once. The parity is that of the permutation taking the alive modes in increasing order to the order pivoted:
    the k-th alive mode (by a prefix sum over the mode ids) is sent to the k-th pivot; a dead mode is a fixed point.
    Cycles by pointer doubling over the n_modes entries, (n_modes - cycles) mod 2. With the elimination incomplete
    the value is meaningless — the caller zeroes the amplitude then."""
    S, n = sequence.shape[0], alive.shape[0]
    if S == 0:
        return jnp.zeros((), jnp.int32)
    valid = sequence >= 0
    pos = jnp.cumsum(valid)-1                                   # the valid entries' order
    seq = jnp.full((n,), -1, jnp.int32).at[jnp.where(valid, pos, n)].set(sequence.astype(jnp.int32), mode="drop")
    rank = jnp.cumsum(alive)-1                                  # the alive modes' order by id
    ids = jnp.arange(n, dtype=jnp.int32)
    mode_of_rank = jnp.full((n,), -1, jnp.int32).at[jnp.where(alive, rank, n)].set(ids, mode="drop")
    # perm[the k-th alive mode] = the k-th pivot; dead modes and absent pivots leave the identity
    target = jnp.where(seq >= 0, mode_of_rank, n)               # the k-th alive mode for every k with a pivot
    perm = ids.at[target].set(jnp.where(seq >= 0, seq, 0), mode="drop")
    low, nxt = ids, perm
    for _ in range(max(1, int(np.ceil(np.log2(max(n, 2)))))):
        low = jnp.minimum(low, low[nxt])
        nxt = nxt[nxt]
    cycles = jnp.sum(low == ids)
    return (n-cycles) % 2


