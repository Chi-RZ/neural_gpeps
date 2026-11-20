"""The network matrix's Pfaffian along the elimination tree (Appendix A), each tree level's fronts eliminated by
one CUDA call.

The plan, the compaction of leftover candidates into the parent's delay slots
and the sign are `batched`'s; the elimination moves into `front_cuda` — a
level's fronts padded to its widest (`n_L`, dead padding slots) and stacked,
one thread block each, the front staying in shared memory for all its steps.
The gradient is the kernel's own backward pass.

The glue is per LEVEL, not per front: a level's fronts assemble themselves in
the kernel from PIECES — each site's packed block of the free vector and each
child's contribution block, through a slot map — plus the bonds' constants,
from tables built once from the plan; the kernel reads only the alive rows, so
nothing is masked in memory. The kernel
itself compacts what is left into the contribution block the parent reads,
`(nf, Dmax + Umax, Dmax + Umax)`: up to `Dmax` delayed rows (the valid ones
first), then the update rows, zero-padded — no front matrix comes back.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .batched import _parity_modes
from .front_cuda import fused_fn, max_width, parity_fits, parity_fn



def levels_of(plan, leaf_cutoff=0):
    """Height groups, optionally splitting leaves at a structural width; zero keeps the original grouping."""
    if leaf_cutoff < 0:
        raise ValueError("leaf_cutoff must be nonnegative")
    height = []
    for fp in plan.fronts:
        height.append(1+max((height[c] for c in fp.children), default=-1))
    levels = {}
    for t, h in enumerate(height):
        levels.setdefault(h, []).append(t)
    groups = [levels[h] for h in sorted(levels)]
    if leaf_cutoff:
        leaves = groups[0]
        small = [t for t in leaves if plan.fronts[t].size <= leaf_cutoff]
        large = [t for t in leaves if plan.fronts[t].size > leaf_cutoff]
        groups = [g for g in (small, large) if g]+groups[1:]
    return groups


@dataclass
class _Level:
    fronts: list
    n: int
    n_fully: tuple
    steps: int
    root: bool
    alive_base: np.ndarray      # (nf, n) 1 on update slots and own virtual modes
    mode_base: np.ndarray       # (nf, n) own and update modes, -1 elsewhere
    phys: tuple                 # (front, slot, physical index) of every own physical slot
    sites: tuple                # (front, slot_i, slot_j, site, k) of every site entry
    bonds: tuple                # (front, slot_p, slot_q, sigma)
    children: dict              # source level -> (src front, src row, src col, dst front, dst row, dst col)
    delayed_in: dict            # source level -> (src front, src row, dst front, dst slot)
    slots_here: np.ndarray      # (nf,) delay slots the parent reserved for each front
    d_max: int
    u_slots: np.ndarray         # (nf, u_max) update slots, padded with 0
    u_valid: np.ndarray         # (nf, u_max)
    u_mode: np.ndarray          # (nf, u_max)
    n_update: tuple             # update rows per front
    src_levels: tuple           # the child levels whose contribution blocks feed this level, in S order
    inv_pos: np.ndarray         # (len S,) the one flat F position each source feeds, nf*n*n = none
    inv_sgn: np.ndarray         # (len S,) its sign
    fused: tuple                # front_fwd_fused's tables
    fused_child: tuple          # (nf_c, cd_c) of each source level, in src_levels order


def _levels(plan):
    D = plan.delay_slots
    groups = levels_of(plan, leaf_cutoff=int(os.environ.get("PIVOTED_LEAF_CUTOFF", "0")))
    where = {}
    for li, lev in enumerate(groups):
        for i, t in enumerate(lev):
            where[t] = (li, i)
    out = []
    for li, lev in enumerate(groups):
        fps = [plan.fronts[t] for t in lev]
        n = max(fp.size for fp in fps)
        nf = len(lev)
        alive_base = np.zeros((nf, n), np.int8)
        mode_base = -np.ones((nf, n), np.int32)
        phys_index = {int(m): i for i, m in enumerate(plan.physical)}
        phys, sites, bonds = [], [], []
        children, delayed_in = {}, {}
        for i, fp in enumerate(fps):
            for slot, m in enumerate(fp.own):
                mode_base[i, slot] = m
                if int(m) in phys_index:
                    phys.append((i, slot, phys_index[int(m)]))
                else:
                    alive_base[i, slot] = 1
            u0 = fp.n_fully
            alive_base[i, u0:u0+len(fp.update)] = 1
            mode_base[i, u0:u0+len(fp.update)] = fp.update
            sites += [(i, int(a), int(b), int(s), int(k)) for s, k, a, b in fp.site_entries]
            bonds += [(i, int(p), int(q), float(sg)) for p, q, sg in fp.bond_entries]
            for c, off, cnt, upd in zip(fp.children, fp.delay_offset, fp.delay_count, fp.child_update_slots):
                lc, ic = where[c]
                child_rows = list(range(cnt))           # the child's delayed rows land on off + r
                child_dst = [off+r for r in range(cnt)]
                d_max_c = max(min(D, plan.fronts[t].n_fully) for t in groups[lc])
                child_rows += [d_max_c+u for u in range(len(upd))]
                child_dst += [int(s) for s in upd]
                tab = children.setdefault(lc, [])
                for r, dr in zip(child_rows, child_dst):
                    for cc, dc in zip(child_rows, child_dst):
                        tab.append((ic, r, cc, i, dr, dc))
                delayed_in.setdefault(lc, []).extend((ic, r, i, off+r) for r in range(cnt))
        slots_here = np.array([min(D, fp.n_fully) for fp in fps])
        u_max = max((len(fp.update) for fp in fps), default=0)
        u_slots = np.zeros((nf, u_max), np.int32)
        u_valid = np.zeros((nf, u_max), bool)
        u_mode = -np.ones((nf, u_max), np.int32)
        for i, fp in enumerate(fps):
            u_slots[i, :len(fp.update)] = fp.n_fully+np.arange(len(fp.update))
            u_valid[i, :len(fp.update)] = True
            u_mode[i, :len(fp.update)] = fp.update
        # the sources: S = [0, free (n_sites * L), C of each source level, flattened]
        L = max(m*(m-1)//2 for m in plan.modes)
        src_levels = tuple(sorted(children))
        offset, c_off = 1+len(plan.modes)*L, {}
        for lc in src_levels:
            c_off[lc] = offset
            cd_c = out[lc].d_max+out[lc].u_slots.shape[1]
            offset += len(groups[lc])*cd_c*cd_c
        entries = {}
        for i_, a, b, site, k in sites:
            key, sgn = ((i_, a, b), 1.0) if a < b else ((i_, b, a), -1.0)
            entries.setdefault(key, []).append((1+site*L+k, sgn))
        for lc in src_levels:
            cd_c = out[lc].d_max+out[lc].u_slots.shape[1]
            for ic, r, cc, i_, dr, dc in children[lc]:     # a child's block is read in its strict upper triangle only
                if dr < dc:
                    src = (c_off[lc]+(ic*cd_c+r)*cd_c+cc, 1.0) if r < cc else (c_off[lc]+(ic*cd_c+cc)*cd_c+r, -1.0)
                    entries.setdefault((i_, dr, dc), []).append(src)
        # every source feeds exactly one entry, so the assembly's adjoint is a gather
        inv_pos = np.full(offset, nf*n*n, np.int32)
        inv_sgn = np.zeros(offset)
        for (i_, a, b), srcs in entries.items():
            for ix, sg in srcs:
                if inv_pos[ix] != nf*n*n:
                    raise ValueError("a source feeds two front entries")
                inv_pos[ix] = (i_*n+a)*n+b
                inv_sgn[ix] = sg
        # the fused kernels' tables, for every level: its source levels' contribution blocks (and their rows'
        # validity and modes) arrive concatenated in `src_levels` order, as they lie in S. A front is assembled from
        # PIECES — each of its sites' packed Z blocks and each child's contribution block, placed through a slot
        # map — then the bonds as constants (the multifrontal extend-add), so the kernel reads every source once
        # and contiguously
        cds = {lc: out[lc].d_max+out[lc].u_slots.shape[1] for lc in src_levels}
        v_off, acc = {}, 0
        for lc in src_levels:
            v_off[lc] = acc
            acc += len(groups[lc])*cds[lc]
        c0 = c_off[src_levels[0]] if src_levels else 0
        pieces = _pieces(plan, fps, sites, {lc: c_off[lc]-c0 for lc in src_levels}, cds,
                         {lc: out[lc].d_max for lc in src_levels}, where, L)
        asrc = np.full((nf, n), -2, np.int32)
        msrc = np.array(mode_base, np.int32)
        for i_, fp in enumerate(fps):
            for slot, m in enumerate(fp.own):
                asrc[i_, slot] = phys_index.get(int(m), -1)
            asrc[i_, fp.n_fully:fp.n_fully+len(fp.update)] = -1
        for lc in src_levels:
            for (ic, r, i_, dst) in delayed_in.get(lc, ()):
                asrc[i_, dst] = -3-(v_off[lc]+ic*cds[lc]+r)
                msrc[i_, dst] = -3-(v_off[lc]+ic*cds[lc]+r)
        counts = np.stack([[fp.n_fully for fp in fps], [min(D, fp.n_fully) for fp in fps],
                           [len(fp.update) for fp in fps]], axis=1).astype(np.int32)
        bd_off, bd_pq, bd_val = [0], [], []
        for i_ in range(nf):
            for j_, p, q, sg in bonds:
                if j_ == i_:
                    bd_pq.append((min(p, q), max(p, q)))
                    bd_val.append(sg if p < q else -sg)
            bd_off.append(len(bd_pq))
        # the packed triangle's (i, j) for every width m <= min(n, 320), stored as (16 i) << 16 | 16 j (the kernels
        # decode i = (v >> 16) >> 4, j = (v & 0xffff) >> 4); front_cuda.cu FusedIn::ijtab: per width m the lane-major
        # table, entry e = lane + 32 t at lane * T + t, T a multiple of 4, 0 past the end
        ijoff = np.zeros(n+2, np.int32)
        parts = []
        if n <= 320:
            for m_ in range(n+1):
                E = m_*(m_-1)//2
                T = ((E+31)//32+3)//4*4
                ij = np.zeros(32*T, np.uint32)
                e_ = 0
                for i_ in range(m_):
                    for j_ in range(i_+1, m_):
                        ij[(e_ % 32)*T + e_//32] = ((16*i_) << 16) | (16*j_)
                        e_ += 1
                parts.append(ij)
                ijoff[m_+1] = ijoff[m_]+32*T
        ijtab = np.concatenate(parts) if parts else np.zeros(1, np.uint32)
        fused = pieces+(np.array(bd_off, np.int32), np.array(bd_pq or [(0, 0)], np.int32).reshape(-1, 2),
                        np.array(bd_val or [0.0], np.float64), asrc, msrc, counts, ijtab, ijoff)
        fused_child = tuple((len(groups[lc]), cds[lc]) for lc in src_levels)
        out.append(_Level(fronts=list(lev), n=n, n_fully=tuple(fp.n_fully for fp in fps),
                          steps=max(fp.n_fully//2 for fp in fps), root=plan.root in lev,
                          alive_base=alive_base, mode_base=mode_base, phys=tuple(phys), sites=tuple(sites),
                          bonds=tuple(bonds), children={k: tuple(v) for k, v in children.items()},
                          delayed_in={k: tuple(v) for k, v in delayed_in.items()}, slots_here=slots_here,
                          d_max=0 if plan.root in lev else int(slots_here.max()), u_slots=u_slots,
                          u_valid=u_valid, u_mode=u_mode, n_update=tuple(len(fp.update) for fp in fps),
                          src_levels=src_levels, inv_pos=inv_pos, inv_sgn=inv_sgn, fused=fused,
                          fused_child=fused_child))
    return out


def _pieces(plan, fps, sites, child_base, cds, dms, where, L):
    """The fused kernels' assembly pieces of one level: `(pc_off (nf + 1,), pc_desc (P, 4), pc_map (M,))`. Piece p
    of front i (p in pc_off[i]:pc_off[i + 1]) has `pc_desc[p] = (base, w, map offset, nown)`: a source block of w
    rows — site s's packed Z block at `base = s * L` of the flat free vector (nown = the front's own slots, >= 0),
    or a child's contribution block at `base` of the flat child blocks (nown = -1) — and a map of w slots (-1 =
    not in this front). Entry (r, c), r < c, of the block adds to F[map[r], map[c]] when both are mapped and, for
    a site, at least one is an own slot (the plan places Z_s[i, j] at the deeper owner: a pair of update modes
    belongs to an ancestor, a delayed mode's pairs were placed where it was own). The site rule is checked here
    against the plan's entries."""
    pc_off, desc, maps = [0], [], []
    for i_, fp in enumerate(fps):
        nown = len(fp.own)
        by_site = {}
        for j_, a, b, s, k in sites:
            if j_ == i_:
                by_site.setdefault(s, []).append((k, a, b))
        for s in sorted(by_site):
            m = plan.modes[s]
            ti, tj = np.triu_indices(m, 1)
            slot = -np.ones(m, np.int32)
            for k, a, b in by_site[s]:
                for mode, sl in ((ti[k], a), (tj[k], b)):
                    if slot[mode] not in (-1, sl):
                        raise ValueError("a site mode in two slots of one front")
                    slot[mode] = sl
            si, sj = slot[ti], slot[tj]
            rule = (si >= 0) & (sj >= 0) & ((si < nown) | (sj < nown))
            if set(np.flatnonzero(rule).tolist()) != {k for k, _, _ in by_site[s]}:
                raise ValueError("the site piece rule does not reproduce the plan's entries")
            desc.append((s*L, m, len(maps), nown))
            maps.extend(slot.tolist())
        for c, off, cnt, upd in zip(fp.children, fp.delay_offset, fp.delay_count, fp.child_update_slots):
            lc, ic = where[c]
            cd = cds[lc]
            rows = -np.ones(cd, np.int32)       # the child's rows: cnt delayed (padded to its level's d_max), then its update
            rows[:cnt] = off+np.arange(cnt)
            rows[dms[lc]:dms[lc]+len(upd)] = upd
            desc.append((child_base[lc]+ic*cd*cd, cd, len(maps), -1))
            maps.extend(rows.tolist())
        pc_off.append(len(desc))
    return (np.array(pc_off, np.int32), np.array(desc or [(0, 0, 0, -1)], np.int32).reshape(-1, 4),
            np.array(maps or [-1], np.int32))


def log_amplitude(plan, threshold):
    """`f(free, occupation) -> (sign, logabs, stats)`: one configuration's Pfaffian, log|psi| and its sign."""
    n_bonds = sum(len(f.bond_entries) for f in plan.fronts)
    log_scale = float(-0.5*n_bonds*np.log(2.0))    # a Python float: weakly typed, so a float32 program stays float32
    weight = jnp.asarray(plan.sign_weight)
    levels = _levels(plan)
    panel = int(os.environ.get("PIVOTED_PANEL", "8"))    # the global kernels' panel; the shared kernels run panel 1
    gbpanel = int(os.environ.get("PIVOTED_GBWD_PANEL", str(panel)))   # the global BACKWARD's panel; the global panel unless set
    # the block kernel's panel: 4 or 8 closes it on the tensor cores, 1 applies each pivot's update at once
    bpanel = int(os.environ.get("PIVOTED_BPANEL", "1"))
    # PIVOTED_ROOT_BPANEL: the root level's block-kernel panel alone (its sweep is a large share of the
    # root's cycles); 0 = the other block levels' PIVOTED_BPANEL
    root_bpanel = int(os.environ.get("PIVOTED_ROOT_BPANEL", "0"))
    # the widest front the one-warp kernel takes (its own bound is 96); narrower levels than this are occupancy-bound
    # on the warp kernel, so a lower value hands them to the block kernel
    warp_max = int(os.environ.get("PIVOTED_WARP_MAX", "96"))
    # the sign's permutation parity: one kernel a configuration taking every level's pivot array
    # (front_cuda.parity_fn) where its four arrays a mode fit this device's shared memory (at most 16384 modes);
    # else batched._parity_modes, which ranks the alive modes by a prefix sum and counts the cycles of the induced
    # permutation in jax ops (the same value)
    n_free = sum(k * (k - 1) // 2 for k in plan.modes)    # each site's strict upper triangle
    physical = np.asarray(plan.physical, np.int32)
    search_window = int(os.environ.get("PIVOTED_SEARCH_WINDOW", "1"))   # one column a round; 0 = the full window
    if search_window < 0:
        raise ValueError("PIVOTED_SEARCH_WINDOW must be nonnegative")
    # a level wider than this device's shared memory holds (for the scalar of the inputs: float32 fronts take half
    # the bytes), or at least PIVOTED_GLOBAL_FROM wide, takes the global-memory kernels
    gfrom = int(os.environ.get("PIVOTED_GLOBAL_FROM", "0"))
    # a level at least PIVOTED_GLOBAL_BWD_FROM wide on the shared-memory kernels takes its BACKWARD on the global-memory
    # kernel (a shorter gradient on the wide levels where the forward prefers the block kernel); 0 = off
    gbfrom = int(os.environ.get("PIVOTED_GLOBAL_BWD_FROM", "0"))
    # PIVOTED_BWD_UU: the shared backward keeps the update block out of its shared triangle and takes the update
    # rows' products over the update columns from one kernel a level (bwd_uu_kernel); needs PIVOTED_BWD_WORK bit 8
    # in the build
    bwduu = int(os.environ.get("PIVOTED_BWD_UU", "0"))      # 1: every shared level; 2: the block kernel's levels only
    # PIVOTED_SHARED_TO: a level wider than the shared-memory bound but at most this wide takes the block
    # kernel with its U x U block out of shared memory (and its backward on the global kernel); 0 = the bound decides
    shared_to = int(os.environ.get("PIVOTED_SHARED_TO", "0"))

    def f(free, occupation):
        # the kernels index both through tables: a wrong shape would read past them, so it stops here (trace time)
        if free.size != n_free or occupation.shape != (len(physical),):
            raise ValueError(f"free vectors of {free.size} entries and occupation of shape {occupation.shape}; the plan "
                             f"needs {n_free} entries and ({len(physical)},)")
        occupation = occupation.astype(jnp.int8)
        rt = free.dtype       # the kernels' scalar: float32 inputs run the float library, else float64
        if rt not in (jnp.float32, jnp.float64):
            raise TypeError(f"free vectors of dtype {rt}; float32 or float64")
        wmax = max_width("float" if rt == jnp.float32 else "double")
        logabs, sign, biggest = jnp.zeros((), free.dtype), jnp.ones((), free.dtype), jnp.zeros((), free.dtype)
        sequences, overflow = [], jnp.zeros((), bool)
        delayed = [None]*len(plan.fronts)
        complete = jnp.ones((), bool)
        scale = jnp.maximum(jnp.max(jnp.abs(free)), 1.0)
        contrib = []        # per level: (C, valid, mode)
        acc = free.reshape(-1)   # the free vector's gradient accumulator threaded through the levels
        for lv in levels:
            nf, n = len(lv.fronts), lv.n
            glob = (n > wmax and not (shared_to > 0 and n <= shared_to)) or (gfrom > 0 and n >= gfrom)
            gbwd = (not glob) and ((gbfrom > 0 and n >= gbfrom) or n > wmax)   # above the bound the shared backward cannot run
            if lv.src_levels:     # the source levels' blocks, rows' validity and modes, concatenated flat
                C_ = jnp.concatenate([contrib[lc][0].reshape(-1) for lc in lv.src_levels])
                V_ = jnp.concatenate([contrib[lc][1].reshape(-1) for lc in lv.src_levels]).astype(jnp.int8)
                M_ = jnp.concatenate([contrib[lc][2].reshape(-1) for lc in lv.src_levels]).astype(jnp.int32)
            else:
                C_, V_, M_ = jnp.zeros(1, rt), jnp.zeros(1, jnp.int8), jnp.zeros(1, jnp.int32)
            level = fused_fn(id(lv), lv.fused, n, lv.steps, float(threshold), bool(lv.root), lv.d_max,
                             0 if lv.root else lv.u_slots.shape[1], int(free.size), lv.fused_child,
                             int(occupation.shape[0]), lv.inv_pos, lv.inv_sgn, panel if glob else (root_bpanel if (lv.root and root_bpanel > 0) else bpanel), glob,
                             search_window=search_window, warp_max=warp_max, dtype=rt, gbwd=gbwd, gpanel=gbpanel,
                             bwduu=bwduu)
            # the free gradient's accumulator rides along (front_cuda.fused_fn)
            C, cvalid, cmode, count, ao, la, sg, piv, big, acc = level(free.reshape(-1), C_, V_, M_, occupation, acc)
            logabs, sign = logabs+jnp.sum(la), sign*jnp.prod(sg)
            biggest = jnp.maximum(biggest, jnp.max(big))
            sequences.append(piv.reshape(-1))
            for i, t in enumerate(lv.fronts):
                delayed[t] = count[i]
            if lv.root:
                complete = jnp.sum(ao) == 0
                contrib.append(None)
                continue
            overflow = overflow | jnp.any(count > jnp.asarray(lv.slots_here))
            contrib.append((C, cvalid.astype(bool), cmode))
        # the accumulator's final value enters log|psi| with weight zero (XLA keeps x * 0 for floats), so its
        # cotangent is instantiated BATCHED under vmap — the FFI's aliased output needs the operand's shape — and
        # that one zero fill a sample is the chain's start
        logabs = logabs + 0.0*(acc[0]*free.reshape(-1)[0])   # times a batched input: grad's seed alone is unbatched
        real = "float" if rt == jnp.float32 else "double"
        if plan.n_modes <= 16384 and parity_fits(real, plan.n_modes):
            alive_modes = jnp.ones((plan.n_modes,), jnp.int8).at[jnp.asarray(physical)].set(occupation.astype(jnp.int8))
            parity = parity_fn(real)(alive_modes, *sequences)
        else:
            alive_modes = jnp.ones((plan.n_modes,), bool).at[jnp.asarray(physical)].set(occupation.astype(bool))
            parity = _parity_modes(jnp.concatenate(sequences), alive_modes)
        eps = int(plan.sign_constant)*(1-2*(jnp.sum(occupation.astype(jnp.int32)*weight) % 2))
        total = jnp.where(complete, sign*(1-2*parity)*eps, jnp.zeros((), rt))
        stats = dict(delayed=jnp.stack(delayed), overflow=overflow, growth=biggest/scale)
        return total, logabs+log_scale, stats
    return f
