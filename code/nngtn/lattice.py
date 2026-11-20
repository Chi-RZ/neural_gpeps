"""The periodic Lx x Ly lattice: quantax's global `Sites`, and the bond lists the Hamiltonian is built from.

Site `(i, j)`, `0 <= i < Lx`, `0 <= j < Ly`, has index `i * Ly + j`: quantax's `Grid([Lx, Ly])` order, the order the
network's feature map `(channels, Lx, Ly)` flattens to, and the order of the free vectors. A configuration is
`(2N,)` of `+-1` occupations, the N spin-up modes first.

The bond lists enumerate one pair per site and direction, in site order, multiplicity kept: the order the
Hamiltonian's terms are summed in.
"""
from __future__ import annotations

__all__ = ["make_sites", "square_bonds", "diagonal_bonds"]


def make_sites(Lx, Ly, n_particles, dtype=None):
    """quantax's global lattice: a periodic spinful `Lx x Ly` grid at fixed `(N_up, N_dn)`, and quantax's default
    dtype (float32 unless given: the network, the amplitude and the Jacobian are stored in 32 bits). quantax keeps
    both in module-level globals, so this runs before any model or sampler is built."""
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.sites import Grid
    qtx.set_default_dtype(jnp.float32 if dtype is None else dtype)
    return Grid([int(Lx), int(Ly)], boundary=1, particle_type="spinful_fermion",
                Nparticles=tuple(int(n) for n in n_particles), double_occ=True)


def square_bonds(Lx, Ly):
    """Nearest-neighbour `(site, site)` pairs of the periodic torus: per site `(i, j)` in index order, `+x` then
    `+y`. A 2-site direction keeps both wrappings (multiplicity kept); a 1-site direction has no bond."""
    def site(i, j):
        return i * Ly + j

    out = []
    for i in range(Lx):
        for j in range(Ly):
            if i + 1 < Lx:
                out.append((site(i, j), site(i + 1, j)))
            elif Lx > 1:
                out.append((site(i, j), site(0, j)))
            if j + 1 < Ly:
                out.append((site(i, j), site(i, j + 1)))
            elif Ly > 1:
                out.append((site(i, j), site(i, 0)))
    return out


def diagonal_bonds(Lx, Ly):
    """Next-nearest (diagonal) `(site, site)` pairs of the periodic torus: per site, `+x+y` then `+x-y` — the t'
    bonds of the t-t' model."""
    if Lx < 2 or Ly < 2:
        return []

    def site(i, j):
        return i * Ly + j

    out = []
    for i in range(Lx):
        for j in range(Ly):
            for dj in (1, -1):
                out.append((site(i, j), site((i + 1) % Lx, (j + dj) % Ly)))
    return out
