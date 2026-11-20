"""The static plan a fixed-shape (jax) multifrontal factorisation runs on.

Everything here is a function of the network and the elimination tree alone —
no configuration, no value — so one trace serves every configuration:

* fronts in postorder; front `t`'s slots are its own modes (the tree is over
  global modes, physical ones included — a configuration masks the unoccupied),
  then `delay_slots` slots per child for the variables that child could not
  pivot, then its update set (modes above `t` its subtree's structure reaches);
* where every original entry is assembled — a site's entry `Z_s[i, j]` and a
  bond alike — into the deeper owner of its two modes (a site's modes may sit
  in several fronts when the tree cuts bonds);
* where a child's contribution lands in its parent: its delayed slots onto the
  parent's slots reserved for that child, its update modes onto the parent's
  slots of the same modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multifrontal import postorder


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
    # epsilon(n) (network_matrix.configuration_sign) is linear in the occupations:
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
