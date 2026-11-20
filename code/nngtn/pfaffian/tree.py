"""Build the contraction tree of a projected Gaussian tensor network, once per lattice (Appendix A).

1. The network (`Network`, `peps_layout`, `network_of_layout`): each site's modes in the order (up, left, phys,
   down, right) and the bonds joining them - the sparsity pattern of the network matrix M.
2. The elimination tree (`lattice_dissection`, `Front`, `postorder`): nested dissection from two-site leaves to
   the root, each front owning the modes eliminated there.
3. The plan (`build`): per front, the modes eliminated there and those passed up, the delay slots and the index
   tables `contract.log_amplitude` reads to evaluate amplitudes on the tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---- the network: modes and bonds ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Network:
    """Topology only: per-site mode counts, which modes are physical, bonds."""
    modes: tuple            # modes per site
    physical: tuple         # per site: tuple of local physical mode indices, in order
    bonds: tuple            # (site_a, mode_a, site_b, mode_b, sigma), local mode indices

    @property
    def offsets(self):
        return np.concatenate([[0], np.cumsum(self.modes)]).astype(int)

    def global_mode(self, site, local):
        return int(self.offsets[site]+local)

    def bond_pairs(self):
        """`(p, q, sigma)` with global `p < q`."""
        out = []
        for sa, ma, sb, mb, sigma in self.bonds:
            p, q = self.global_mode(sa, ma), self.global_mode(sb, mb)
            # M_pq = sigma in the bond's own orientation (a then b)
            out.append((p, q, sigma) if p < q else (q, p, -sigma))
        return out


# ---- the PEPS layout -----------------------------------------------------------------------------------------------------
# The PEPS layout as a `Network`, and a free vector as a site's `Z`.
#
# Only the layout's STATIC data is read — leg sizes, leg names, index names —
# and the free vector's real chart (the strict upper triangle of the site's skew
# matrix in `triu_indices` order).

#: which side of a bond a leg is: the orientation of the bond operator (the tests hold it to the brute-force
#: Fock-space amplitude)
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


# ---- a site's legs and modes ---------------------------------------------------------------------------------------------
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


# ---- the lattice's layout ---------------------------------------------------------------------------------------------
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


# ---- the elimination tree ------------------------------------------------------------------------------------------------
# The elimination tree, over global MODES: a front owns a set of modes.
#
# `lattice_dissection`: nested dissection of the Lx x Ly lattice by bonds (Appendix A), down to regions of
# `leaf_sites` sites. Every bond then joins a front to itself or to an ancestor, which the elimination needs.

@dataclass
class Front:
    """One node of the elimination tree: the variables it owns, its children."""
    own: list
    children: list = field(default_factory=list)


def postorder(roots):
    """Fronts children-first."""
    out = []

    def visit(front):
        for child in front.children:
            visit(child)
        out.append(front)
    for root in roots:
        visit(root)
    return out


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


# ---- the plan ------------------------------------------------------------------------------------------------------------
# The static plan a fixed-shape (jax) multifrontal factorisation runs on.
#
# Everything here is a function of the network and the elimination tree alone —
# no configuration, no value — so one trace serves every configuration:
#
# * fronts in postorder; front `t`'s slots are its own modes (the tree is over
#   global modes, physical ones included — a configuration masks the unoccupied),
#   then `delay_slots` slots per child for the variables that child could not
#   pivot, then its update set (modes above `t` its subtree's structure reaches);
# * where every original entry is assembled — a site's entry `Z_s[i, j]` and a
#   bond alike — into the deeper owner of its two modes (a site's modes may sit
#   in several fronts when the tree cuts bonds);
# * where a child's contribution lands in its parent: its delayed slots onto the
#   parent's slots reserved for that child, its update modes onto the parent's
#   slots of the same modes.

@dataclass(frozen=True)
class FrontPlan:
    own: np.ndarray             # global mode ids the front owns, slot order
    update: np.ndarray          # global mode ids of the update set, slot order
    children: tuple             # indices (into the postorder) of the children
    delay_offset: tuple         # first delayed slot reserved for each child
    delay_count: tuple          # slots reserved for each child: delay_slots, or fewer when the child has fewer
    n_delay: int                # sum of delay_count
    child_update_slots: tuple   # per child: slot in THIS front of each child update mode
    site_entries: np.ndarray    # (site, free index, slot_i, slot_j) of every Z_s[i, j], i < j, assembled here
    bond_entries: np.ndarray    # (slot_p, slot_q, sigma) assembled here, oriented
    size: int

    @property
    def n_fully(self):
        return len(self.own)+self.n_delay


@dataclass(frozen=True)
class Plan:
    fronts: tuple               # FrontPlan, postorder
    root: int
    n_modes: int
    delay_slots: int
    offsets: np.ndarray         # first global mode of each site
    modes: tuple                # modes per site
    physical: np.ndarray        # global ids of every physical mode
    sign_weight: np.ndarray     # per physical mode: bond endpoints after it, mod 2
    sign_constant: int          # prod sigma times (-1)^(inversions of the bond pairs)


def build(net, roots, delay_slots):
    """The plan of a tree over global modes (`tree.lattice_dissection`)."""
    if len(roots) != 1:
        raise ValueError("one connected network, one root")
    order = postorder(roots)
    index = {id(f): t for t, f in enumerate(order)}
    offsets = net.offsets
    n_modes = int(offsets[-1])
    mode_owner = np.full(n_modes, -1)
    for t, f in enumerate(order):
        mode_owner[np.asarray(f.own, dtype=int)] = t
    if (mode_owner < 0).any():
        raise ValueError("every mode needs an owning front")
    pairs = net.bond_pairs()            # (p, q, sigma), p < q, sigma oriented for M[p, q]
    adjacency = {m: set() for m in range(n_modes)}
    for p, q, _ in pairs:
        adjacency[p].add(q)
        adjacency[q].add(p)
    for site in range(len(net.modes)):  # a site's block is dense
        block = set(range(offsets[site], offsets[site+1]))
        for m in block:
            adjacency[m] |= block-{m}
    # every Z_s[i, j], i < j, by its free index, at the deeper owner of its two modes
    site_entries = [[] for _ in order]
    for site, m in enumerate(net.modes):
        for k, (i, j) in enumerate(zip(*np.triu_indices(m, 1))):
            p, q = offsets[site]+i, offsets[site]+j
            site_entries[min(mode_owner[p], mode_owner[q])].append((site, k, p, q))
    # symbolic: a front's update set is what its own modes and its children's
    # update sets reach above it
    updates = []
    fronts = []
    for t, f in enumerate(order):
        own = np.asarray(f.own, dtype=int)
        children = tuple(index[id(c)] for c in f.children)
        reach = set()
        for m in own:
            reach |= adjacency[int(m)]
        for c in children:
            reach |= set(updates[c].tolist())
        reach = {m for m in reach if mode_owner[m] > t}
        update = np.array(sorted(reach), dtype=int)
        updates.append(update)
        # a child can leave at most its own fully summed count unpivoted
        delay_count = tuple(min(delay_slots, fronts[c].n_fully) for c in children)
        n_delay = sum(delay_count)
        slot = {int(m): i for i, m in enumerate(own)}
        delay_offset = tuple(len(own)+int(np.sum(delay_count[:i])) for i in range(len(children)))
        for i, m in enumerate(update):
            slot[int(m)] = len(own)+n_delay+i
        child_update_slots = tuple(np.array([slot[int(m)] for m in updates[c]], dtype=int) for c in children)
        entries = np.array([(site, k, slot[p], slot[q]) for site, k, p, q in site_entries[t]], dtype=int).reshape(-1, 4)
        bonds = [(slot[p], slot[q], sigma) for p, q, sigma in pairs if min(mode_owner[p], mode_owner[q]) == t]
        fronts.append(FrontPlan(own=own, update=update, children=children, delay_offset=delay_offset,
                                delay_count=delay_count, n_delay=n_delay, child_update_slots=child_update_slots,
                                site_entries=entries, bond_entries=np.array(bonds, dtype=float).reshape(-1, 3),
                                size=len(own)+n_delay+len(update)))
    if len(updates[-1]):
        raise ValueError("the root front reaches above itself")
    physical = np.array([offsets[s]+l for s in range(len(net.modes)) for l in net.physical[s]], dtype=int)
    # epsilon(n) (the configuration sign the tests compute from M's definition) is linear in the occupations:
    # the sequence [occupied descending, (q_b, p_b) per bond] has, against descending
    # order, the bond pairs' own inversions plus one for every bond endpoint above
    # an occupied mode
    ends = np.array([net.global_mode(*b[i:i+2]) for b in net.bonds for i in (2, 0)], dtype=int)
    inversions = sum(1 for i in range(len(ends)) for j in range(i+1, len(ends)) if ends[i] < ends[j])
    sigma = int(np.prod([b[4] for b in net.bonds]))
    weight = np.array([int(np.sum(ends > x)) % 2 for x in physical], dtype=int)
    return Plan(fronts=tuple(fronts), root=len(fronts)-1, n_modes=int(offsets[-1]), delay_slots=delay_slots,
                offsets=offsets, modes=tuple(net.modes), physical=physical, sign_weight=weight,
                sign_constant=sigma*(-1)**inversions)
