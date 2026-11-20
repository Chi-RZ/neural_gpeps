"""The elimination tree, over global MODES: a front owns a set of modes.

`lattice_dissection`: nested dissection of the Lx x Ly lattice by bonds (Appendix A), down to regions of
`leaf_sites` sites. Every bond then joins a front to itself or to an ancestor, which `multifrontal` needs.
"""
from __future__ import annotations

import numpy as np

from .multifrontal import Front


def lattice_dissection(net, Lx, Ly, leaf_sites=2):
    """Nested dissection by BONDS on an `Lx x Ly` lattice (site `i * Ly + j`).

    A region of sites is split across its longer extent — a closed ring into
    two halves (two cut lines), an open strip at its middle (one) — and the
    separator is, for every bond the cut crosses, its endpoint mode on the
    first half: the cut bonds' virtual modes only, never a site's other modes,
    which stay below in their own half. At `leaf_sites` sites or fewer a region
    is a leaf owning all its sites' remaining modes.
    """
    if Lx*Ly != len(net.modes):
        raise ValueError("the lattice shape does not match the network")
    o = net.offsets
    site_of_mode = np.repeat(np.arange(len(net.modes)), net.modes)
    partner = {}
    for sa, ma, sb, mb, _ in net.bonds:
        p, q = net.global_mode(sa, ma), net.global_mode(sb, mb)
        partner[p], partner[q] = q, p
    taken = set()

    def coords(site):
        return divmod(site, Ly)

    def split(sites, extent):
        """`extent[d] = (start, length, closed)`: the region's span in dimension d."""
        if len(sites) <= leaf_sites:
            own = [m for site in sorted(sites) for m in range(o[site], o[site+1]) if m not in taken]
            taken.update(own)
            return Front(own=own)
        d = 0 if extent[0][1] >= extent[1][1] else 1
        start, length, closed = extent[d]
        half = length//2
        L = Lx if d == 0 else Ly

        def side(site):
            x = coords(site)[d]
            return ((x-start) % L) < half
        first = {s_ for s_ in sites if side(s_)}
        second = set(sites)-first
        separator = sorted(m for site in first for m in range(o[site], o[site+1])
                           if m not in taken and m in partner and site_of_mode[partner[m]] in second
                           and partner[m] not in taken)
        taken.update(separator)
        ext_a, ext_b = list(extent), list(extent)
        ext_a[d] = (start, half, False)
        ext_b[d] = ((start+half) % L, length-half, False)
        children = [split(first, ext_a), split(second, ext_b)]
        return Front(own=separator, children=children)

    root = split(set(range(len(net.modes))), [(0, Lx, True), (0, Ly, True)])
    missing = int(o[-1])-len(taken)
    if missing:
        raise ValueError(f"{missing} modes were assigned to no front")
    return [root]
