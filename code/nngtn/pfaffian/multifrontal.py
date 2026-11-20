"""The Pfaffian of a sparse skew matrix by a multifrontal skew Parlett-Reid
factorisation with threshold and delayed pivoting.

THE ELIMINATION.  A pivot is a pair `(k, r)` of variables; eliminating it
replaces the remaining block by its Schur complement

    A'_SS = A_SS - (A_Sr A_kS - A_Sk A_rS) / a,      a = A_kr,

and `Pf A = sgn(order) * prod a` over the pivots in elimination order. The
only division is by the pivot, and only row `k` is divided (`A_kS / a`), so
the multipliers are bounded by `1/u` when `|a| >= u * max |column k|` — the
threshold test. A fully summed variable whose column has no acceptable pivot
inside its own front is DELAYED: handed to the parent front with its
contribution, where more of its column is fully summed. The root front holds
every remaining variable, so there the test always passes with the largest
entry (`u = 1` is complete pivoting at every front).

THE FRONTS.  An elimination tree whose nodes own disjoint variable sets,
children before parents, every nonzero `A_ij` joining a node to itself or to
an ancestor (nested dissection gives one). A front's matrix is its own and
delayed variables, then the variables its column structure reaches above it;
original entries are assembled at the deeper owner, children's contribution
blocks extend-added.

Generic in the scalar: float64 arrays, or numpy object arrays of mpmath
numbers for a high-precision evaluation of the SAME elimination.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Front:
    """One node of the elimination tree: the variables it owns, its children."""
    own: list
    children: list = field(default_factory=list)


@dataclass
class Factor:
    """`Pf A = sign * exp(logabs)`, and how the elimination went."""
    sign: int
    logabs: object
    pivots: list            # (k, r, a) in elimination order
    delayed: list           # number of variables each front passed upward
    growth: float           # max |entry| ever formed / max |A|


def _perm_sign(seq):
    """Sign of the permutation `seq` of `range(len(seq))`."""
    seen, sign = [False]*len(seq), 1
    for i in range(len(seq)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j], j, length = True, seq[j], length+1
        if length % 2 == 0:
            sign = -sign
    return sign


def _log(x):
    import math
    if isinstance(x, float):
        return math.log(x)
    import mpmath
    return mpmath.log(x)


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


def factorise(A, roots, threshold=0.1):
    """Multifrontal skew factorisation of dense-stored skew `A` along `roots`.

    `A` is read only at the nonzero pattern; `threshold` is `u` in (0, 1].
    """
    n = A.shape[0]
    order = postorder(roots)
    owner = np.full(n, -1)
    for t, front in enumerate(order):
        owner[front.own] = t
    if (owner < 0).any():
        raise ValueError("every variable needs an owning front")
    rows, cols = np.nonzero(A != 0)
    # an entry is assembled at the deeper owner: the earlier one in postorder
    entries = [[] for _ in order]
    for i, j in zip(rows, cols):
        if i < j:
            entries[min(owner[i], owner[j])].append((i, j))
    scale = max((abs(A[i, j]) for i, j in zip(rows, cols)), default=0)
    contribution = {}           # id(front) -> (index list, matrix)
    pivots, delayed, biggest = [], [], scale
    zero = A.dtype.type(0) if A.dtype != object else A.flat[0]*0
    for t, front in enumerate(order):
        fully = list(front.own)
        update = set()
        for child in front.children:
            idx, _, n_delay = contribution[id(child)]
            fully += idx[:n_delay]
            update |= set(idx[n_delay:])
        for i, j in entries[t]:
            update |= {i, j}
        update -= set(fully)
        # an above-front variable this front's original entries reach
        idx = fully+sorted(update)
        where = {v: p for p, v in enumerate(idx)}
        F = np.full((len(idx), len(idx)), zero, dtype=A.dtype)
        for i, j in entries[t]:
            F[where[i], where[j]] += A[i, j]
            F[where[j], where[i]] -= A[i, j]
        for child in front.children:
            cidx, C, _ = contribution.pop(id(child))
            p = [where[v] for v in cidx]
            F[np.ix_(p, p)] += C
        alive = list(range(len(idx)))
        candidates = list(range(len(fully)))
        while len(candidates) >= 2:
            best = None
            for k in candidates:
                column = np.abs(F[alive, k])
                colmax = column.max()
                inside = [r for r in candidates if r != k]
                values = np.abs(F[inside, k])
                m = int(np.argmax(values))
                if values[m] == 0:
                    continue
                ratio = values[m]/colmax
                key = (ratio >= threshold, values[m]) if front_is_root(front, roots) else (ratio, values[m])
                if best is None or key > best[0]:
                    best = (key, k, inside[m], ratio)
            if best is None or best[3] < threshold:
                break
            _, k, r, _ = best
            a = F[k, r]
            rest = [p for p in alive if p not in (k, r)]
            if rest:
                tau = F[k, rest]/a
                column = F[rest, r]
                F[np.ix_(rest, rest)] -= np.outer(column, tau)-np.outer(tau, column)
                grown = np.abs(F[np.ix_(rest, rest)]).max()
                biggest = max(biggest, grown)
            pivots.append((idx[k], idx[r], a))
            alive = rest
            candidates = [c for c in candidates if c not in (k, r)]
        delayed.append(len(candidates))
        remaining = candidates+[p for p in alive if p >= len(fully)]
        contribution[id(front)] = ([idx[p] for p in remaining], F[np.ix_(remaining, remaining)], len(candidates))
    leftover = [c for root in roots for c in [contribution[id(root)]]]
    if any(len(c[0]) for c in leftover):
        # an odd count, or a singular remainder: the Pfaffian vanishes
        return Factor(0, None, pivots, delayed, float(biggest/scale) if scale else 0.0)
    sequence = [v for k, r, _ in pivots for v in (k, r)]
    sign = _perm_sign(sequence)
    logabs = 0
    for _, _, a in pivots:
        sign = sign*(1 if a > 0 else -1)
        logabs = logabs+_log(abs(a))
    return Factor(sign, logabs, pivots, delayed, float(biggest/scale))


def front_is_root(front, roots):
    return any(front is root for root in roots)
