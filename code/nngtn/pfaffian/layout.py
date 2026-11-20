"""The PEPS layout as a `Network`, and a free vector as a site's `Z`.

Only the layout's STATIC data is read — leg sizes, leg names, index names —
and the free vector's real chart (the strict upper triangle of the site's skew
matrix in `triu_indices` order).
"""
from __future__ import annotations

from .network_matrix import Network

#: which side of a bond a leg is — the original layout's `LEG_DUAL`, restated here
#: only as the orientation of the bond operator (the tests hold it to the
#: brute-force Fock-space amplitude)
KET_SIDE = {"up": True, "right": True, "down": False, "left": False}


def network_of_layout(layout):
    """`layout` entries `(modes, names, inds, _, label)` in site order."""
    legs = {}
    modes, physical = [], []
    for s, (sizes, names, inds, _, _) in enumerate(layout):
        offset, phys = 0, ()
        for size, name, ind in zip(sizes, names, inds):
            if name == "phys":
                phys = tuple(range(offset, offset+size))
            else:
                legs.setdefault(ind, []).append((s, name, offset, size))
            offset += size
        modes.append(offset)
        physical.append(phys)
    bonds = []
    for ind, ends in legs.items():
        if len(ends) != 2:
            raise ValueError(f"bond {ind} has {len(ends)} ends")
        (sa, na, oa, wa), (sb, nb, ob, wb) = ends
        if KET_SIDE[na] == KET_SIDE[nb] or wa != wb:
            raise ValueError(f"bond {ind} joins {na} to {nb}")
        if not KET_SIDE[na]:
            (sa, na, oa, wa), (sb, nb, ob, wb) = (sb, nb, ob, wb), (sa, na, oa, wa)
        bonds += [(sa, oa+t, sb, ob+t, 1) for t in range(wa)]
    return Network(tuple(modes), tuple(physical), tuple(bonds))


# ---- copied from the original's networks module ------------------------------------------------------------------------
LEGS = ("up", "left", "phys", "down", "right")


def virtual_dims(i, j, Lx, Ly, nvx, nvy, periodic=False):
    """Modes on each of a site's four virtual legs; 0 where a leg is absent.

    Open boundaries drop the legs that would leave the lattice; periodic ones
    keep every leg and let the wrap-around index pair them up.
    """
    if periodic:
        return nvy, nvy, nvx, nvx
    up = nvy if j < Ly - 1 else 0
    down = nvy if j > 0 else 0
    left = nvx if i > 0 else 0
    right = nvx if i < Lx - 1 else 0
    return up, down, left, right


def site_modes(i, j, Lx, Ly, d, nvx, nvy, periodic=False):
    """The site's leg modes in `LEGS` order, zeros dropped, and the leg names."""
    up, down, left, right = virtual_dims(i, j, Lx, Ly, nvx, nvy, periodic)
    sizes = {"up": up, "left": left, "phys": d, "down": down, "right": right}
    names = [leg for leg in LEGS if sizes[leg]]
    return tuple(sizes[leg] for leg in names), names


def bond_names(i, j, Lx, Ly, periodic=False):
    """Index name per leg, so two sites that share a bond share a name."""
    return {
        "phys": f"p{i},{j}",
        "up": f"vy{i},{j}",
        "down": f"vy{i},{(j - 1) % Ly}",
        "left": f"vx{(i - 1) % Lx},{j}",
        "right": f"vx{i},{j}",
    }


# ---- the layout ---------------------------------------------------------------------------------------------------
def peps_layout(Lx, Ly, d, nvx, nvy, periodic=False):
    """`(modes, leg names, index names, Majorana size, site label)` per site,
    in site order — one layout, whichever chart fills it.

    `site = i * Ly + j`, the label `networks`' builders put in a pairing
    array's `labels` and the order `physical_inds` returns.
    """
    out = []
    for i in range(Lx):
        for j in range(Ly):
            modes, names = site_modes(i, j, Lx, Ly, d, nvx, nvy, periodic)
            bn = bond_names(i, j, Lx, Ly, periodic)
            out.append((modes, names, tuple(bn[n] for n in names),
                        2 * sum(modes), i * Ly + j))
    return out
