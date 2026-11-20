"""The problem: the periodic Lx x Ly lattice (quantax's global `Sites`), its bonds, the t-t'-U Hubbard
Hamiltonian, and the symmetry a state is projected onto.

Site `(i, j)`, `0 <= i < Lx`, `0 <= j < Ly`, has index `i * Ly + j`: quantax's `Grid([Lx, Ly])` order, the order the
network's feature map `(channels, Lx, Ly)` flattens to, and the order of the free vectors. A configuration is
`(2N,)` of `+-1` occupations, the N spin-up modes first.

The bond lists enumerate one pair per site and direction, in site order, multiplicity kept: the order the
Hamiltonian's terms are summed in.

    H = -t sum_<ij>,sigma (c+_i c_j + h.c.) - t' sum_<<ij>>,sigma (c+_i c_j + h.c.) + U sum_i n_i,up n_i,dn
        - mu sum_i n_i

The symmetry is named by quantax's symmetry classes joined by `@`: `"flip0@flip1@spininverse"` is quantax's
`Flip(axis=0) @ Flip(axis=1) @ SpinInverse()`, every generator in its trivial sector, a group of 8. quantax evaluates
the model on all 8 images of every configuration and sums them with their fermionic signs, so each forward pass
costs 8 times as much.
"""
from __future__ import annotations

__all__ = ["make_sites", "square_bonds", "diagonal_bonds", "hubbard_operator", "hamiltonian", "SYMMETRY_TOKENS",
           "build_symm"]


# ---- lattice ------------------------------------------------------------------------------------------------------
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


# ---- Hamiltonian --------------------------------------------------------------------------------------------------
def hubbard_operator(bonds, n_sites, t=1.0, u=0.0, mu=0.0, bonds2=(), t2=0.0):
    """The Hubbard Hamiltonian as a quantax `Operator`, built from explicit bond lists (a repeated pair is counted
    twice): hopping `-t` on `bonds`, `-t2` on `bonds2` (the t' of the t-t' model on the diagonals, quantax's
    `Hubbard(t=[t, t2])` sign convention), `U n_up n_dn` on every site, `-mu n`."""
    from quantax.operator import annihilate_d, annihilate_u, create_d, create_u, number_d, number_u

    def hop(i, j):
        return (create_u(i) @ annihilate_u(j) + create_u(j) @ annihilate_u(i)
                + create_d(i) @ annihilate_d(j) + create_d(j) @ annihilate_d(i))

    H = 0
    for i, j in bonds:
        H = H + (-t) * hop(int(i), int(j))
    if t2:
        for i, j in bonds2:
            H = H + (-t2) * hop(int(i), int(j))
    if u:
        H = H + u * sum(number_u(i) @ number_d(i) for i in range(n_sites))
    if mu:
        H = H + (-mu) * sum(number_u(i) + number_d(i) for i in range(n_sites))
    return H


def hamiltonian(Lx, Ly, *, t=1.0, u=8.0, tprime=0.0, mu=0.0):
    """`hubbard_operator` on the periodic Lx x Ly torus (`square_bonds`, and `diagonal_bonds` for t')."""
    return hubbard_operator(square_bonds(Lx, Ly), Lx * Ly, t=t, u=u, mu=mu,
                            bonds2=diagonal_bonds(Lx, Ly) if tprime else (), t2=tprime)


# ---- symmetry -----------------------------------------------------------------------------------------------------
SYMMETRY_TOKENS = ("flip0", "flip1", "spininverse")


def build_symm(spec=None):
    """The quantax symmetry `spec` names (`@`-joined tokens from `SYMMETRY_TOKENS`: `flip0` and `flip1` the
    lattice's mirrors `Flip(axis=0)` and `Flip(axis=1)`, `spininverse` quantax's `SpinInverse()`, each in its trivial
    sector), or `None` for `None`/`"none"` (no projection).
    Needs the lattice defined (`make_sites`): quantax reads the coordinates from it."""
    if spec is None or spec == "none":
        return None
    from quantax.symmetry import Flip, SpinInverse
    table = {"flip0": lambda: Flip(axis=0), "flip1": lambda: Flip(axis=1),
             "spininverse": SpinInverse}
    names = spec.split("@")
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown symmetry {unknown}; expected '@'-joined names from {SYMMETRY_TOKENS}")
    symm = table[names[0]]()
    for n in names[1:]:
        symm = symm @ table[n]()
    return symm
