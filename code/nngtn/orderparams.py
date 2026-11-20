"""The order parameters of a sampled state on the periodic Lx x Ly Hubbard torus: d-wave pair correlations, stripe
spin order and charge correlations.

THE OBSERVABLES.  Site (x, y) has index x * Ly + y, and a configuration is (batch, 2N) of +-1 occupations, the N
spin-up modes first — quantax's `Grid` and spinful-fermion layout. A nearest-neighbour bond is keyed (x, y, axis)
by its lower end, axis 0 along +x and 1 along +y.

* d-wave pairing.  A^dag(bond, o) = c^dag_{p up} c^dag_{q dn}, (p, q) the bond's (lower, upper) ends for o = 0 and
  swapped for o = 1; the bond singlet is B^dag = (A^dag(0) + A^dag(1)) / sqrt(2) and
  Delta^dag_r = 1/4 sum_eta h_eta B^dag(r, r + eta), h = +1 on x bonds and -1 on y bonds. Measured are
  F[bl, oL, br, oR] = <A^dag(bl, oL) A(br, oR)> for the left bonds at R and every right bond, and
  G[i, j, s] = <c^dag_{i s} c_{j s}> for the left bonds' sites and every j. Assembled:
  C_p(r) = <Delta^dag_R Delta_{R+r}> - N_{R,R+r}, N the same sum with each F replaced by its normal Wick product of
  G; Delta^2_SC is the mean of C_p over the displacements of minimum-image length at least half the largest.
* Stripe spin order.  C_s(r) = C_zz(r) + (C_+-(r) + C_-+(r)) / 2, C_+-(r) = <S^+_a S^-_{a+r}> averaged over
  reference sites a; S(k) = sum_r e^{ik.r} C_s(r) and M^2_str is the mean of S(k_alpha) / N over the stripe peaks
  k_alpha = (pi -+ q, pi), (pi, pi -+ q), q = pi/8 at t' = 0, that lie on the momentum grid: all four on a square
  torus, (1/4N) sum_alpha S(k_alpha); on 16x4 only the two along x (k_y = pi -+ pi/8 is not on the grid of Ly = 4),
  (1/2N) sum_alpha S(k_alpha).
* Charge.  C_c(r) = <(n_0 - nbar)(n_r - nbar)>, its peaks at q_c = +-(2q, 0), +-(0, 2q).

THE PACKING.  One `Oloc` per term would evaluate the state on every term's connected configurations separately,
most of each forward chunk wasted on terms that connect few samples. Instead all terms sharing a left end are summed
into one quantax operator, its connected configurations are built and evaluated once, and each local value
<s|O|s'> psi(s') / psi(s) is binned by the term it comes from — told from how s' differs from s, since every term
moves at most one electron of each spin. `packed_sum` is that loop; a term's bin is chosen by `bins(s, s')`.

NON-FINITE TERMS.  A connected configuration whose amplitude is not finite — the sparse pivoted Pfaffian's overflowed
front (`model.NeuralGTN.log_psi`) — contributes zero and is counted; the sample is kept. The counts are returned
beside every measurement.
"""
from __future__ import annotations

import copy
from functools import partial, reduce
from operator import iadd

import jax
import jax.numpy as jnp
import numpy as np

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FORM = {(1, 0): 1.0, (-1, 0): 1.0, (0, 1): -1.0, (0, -1): -1.0}
SPINS = ("u", "d")
STRIPE_Q = {0.0: (1, 8), -0.2: (1, 4)}   # t' -> stripe wavevector q / pi as (numerator, denominator): pi/8 at
# t' = 0, pi/4 at t' = -0.2


# ---------------------------------------------------------------- geometry
class Torus:
    """Sites and nearest-neighbour bonds of the periodic Lx x Ly lattice."""

    def __init__(self, Lx, Ly):
        self.Lx, self.Ly, self.N = Lx, Ly, Lx * Ly
        self.sites = [(x, y) for x in range(Lx) for y in range(Ly)]
        self.bonds = [(x, y, ax) for (x, y) in self.sites for ax in (0, 1)]

    def index(self, p):
        return int(p[0]) * self.Ly + int(p[1])

    def wrap(self, p):
        return (p[0] % self.Lx, p[1] % self.Ly)

    def shift(self, p, t):
        return self.wrap((p[0] + t[0], p[1] + t[1]))

    def bond_at(self, p, d):
        """The bond from site p along the unit vector d."""
        if d[0] + d[1] < 0:
            p = (p[0] + d[0], p[1] + d[1])
        return (*self.wrap(p), 0 if d[0] else 1)

    def ends(self, bond):
        x, y, ax = bond
        return (x, y), self.wrap((x + 1, y) if ax == 0 else (x, y + 1))

    def pair_sites(self, bond, o):
        """(site of c^dag_up, site of c^dag_dn) of A^dag(bond, o)."""
        p, q = self.ends(bond)
        return (p, q) if o == 0 else (q, p)

    def min_image(self, dr):
        dr = np.asarray(dr)
        dx = np.minimum(dr[..., 0] % self.Lx, -dr[..., 0] % self.Lx)
        dy = np.minimum(dr[..., 1] % self.Ly, -dr[..., 1] % self.Ly)
        return np.hypot(dx, dy)


def _sum(terms):
    """A sum of quantax operators, added in place (a chain of `+` costs quadratically)."""
    return reduce(iadd, terms[1:], copy.deepcopy(terms[0]))


def pair_creation(geo, bond, o):
    from quantax.operator import create_d, create_u
    p, q = geo.pair_sites(bond, o)
    return create_u(*p) @ create_d(*q)


def hopping(i, j, spin):
    """c^dag_{i s} c_{j s} (i != j)."""
    from quantax.operator import annihilate_d, annihilate_u, create_d, create_u
    if spin == "u":
        return create_u(*i) @ annihilate_u(*j)
    return create_d(*i) @ annihilate_d(*j)


def spin_flip(a, j, channel):
    """S^+_a S^-_j (channel 0) or S^-_a S^+_j (channel 1), S^+ = c^dag_up c_dn."""
    from quantax.operator import annihilate_d, annihilate_u, create_d, create_u

    def sp(p):
        return create_u(*p) @ annihilate_d(*p)

    def sm(p):
        return create_d(*p) @ annihilate_u(*p)

    return sp(a) @ sm(j) if channel == 0 else sm(a) @ sp(j)


# ---------------------------------------------------------------- the packed sum
def packed_sum(state, samples, op, bins, nbins):
    """sum over the samples of sum_{s'} <s|op|s'> psi(s') / psi(s), each term added to bin `bins(s, s')` of
    `nbins`.

    `samples.psi` must hold the amplitudes of `samples.spins`. A term whose value is not finite contributes zero.
    Returns (sums (nbins,), the count of non-finite terms, the mask (nsamples,) of the samples that owned one) —
    the mask names the configurations to save, so a run can be repaired at a wider pivot plan without resampling.
    """
    from quantax.operator.operator import _get_conn, _get_conn_size
    s, psi = samples.spins, samples.psi
    total, bad, owner = jnp.zeros(nbins), jnp.zeros((), jnp.int32), jnp.zeros(s.shape[0], bool)
    for _, s_conn, h_conn in op.apply_off_diag(s):
        size = int(_get_conn_size(h_conn, state.forward_chunk, False))
        segment, s_conn, h_conn = _get_conn(s_conn, h_conn, size, s, False)
        t, b, m = _binned(s, psi, segment, s_conn, state(s_conn), h_conn, bins, nbins)
        total, bad, owner = total + t, bad + b, owner | m
    return total, bad, owner


def _is_term(segment, h_conn):
    """The entries of quantax `_get_conn`'s packed connections that are terms. Its padding carries segment -1 and
    H = 0 on each card, but the segment is then offset by the card's first sample, so on every card after the first
    a padding entry names the previous card's last sample: `segment >= 0` alone takes it for a term, and its ratio
    of an unrelated configuration to that sample can leave float32. A term's H is never 0 — `_get_conn` keeps only entries with H not close to 0."""
    return (segment >= 0) & (h_conn != 0)


@partial(jax.jit, static_argnums=(6, 7))
def _binned(s, psi, segment, s_conn, psi_conn, h_conn, bins, nbins):
    valid = _is_term(segment, h_conn)
    seg = jnp.where(valid, segment, 0)
    val = jnp.real(jnp.asarray(psi_conn / psi[seg]) * h_conn)
    bad = valid & ~jnp.isfinite(val)
    val = jnp.where(valid & ~bad, val, 0.0)
    owner = jax.ops.segment_sum(bad.astype(jnp.int32), seg, num_segments=s.shape[0]) > 0
    return jax.ops.segment_sum(val, bins(s[seg], s_conn), num_segments=nbins), jnp.sum(bad).astype(jnp.int32), owner


def classified_sum(state, samples, op, bins, nbins, stats, psitilde=True):
    """`packed_sum` with every non-finite term classified, for re-reading saved configurations: a term whose
    connected amplitude psi(s') is not finite (the pivoted contraction's overflowed buffer) is an AMPLITUDE overflow
    and adds zero, which is the measured state psi-tilde = psi x 1[contraction succeeds]; a term whose two amplitudes
    are finite but whose ratio psi(s') / psi(s) leaves float32 is a RATIO overflow, which `packed_sum` zeroes and
    which psi-tilde counts at its value, taken here in float64 from the log amplitudes when `psitilde` (else zeroed,
    reproducing `packed_sum`); anything else non-finite is OTHER and adds zero. Adds the counts and the largest
    |log ratio| of the ratio overflows into `stats` (keys "amplitude", "ratio", "other", "max_log_ratio").
    Returns `packed_sum`'s (sums, non-finite count, owner mask), the count being every non-finite term."""
    from quantax.operator.operator import _get_conn, _get_conn_size
    s, psi = samples.spins, samples.psi
    total, bad, owner = jnp.zeros(nbins, jnp.float64), jnp.zeros((), jnp.int32), jnp.zeros(s.shape[0], bool)
    for _, s_conn, h_conn in op.apply_off_diag(s):
        size = int(_get_conn_size(h_conn, state.forward_chunk, False))
        if size == 0:                             # a small batch can have no connected configuration of an operator
            continue
        segment, s_conn, h_conn = _get_conn(s_conn, h_conn, size, s, False)
        t, b, m, c = _binned_classified(s, psi, segment, s_conn, state(s_conn), h_conn, bins, nbins, psitilde)
        total, bad, owner = total + t, bad + b, owner | m
        for k, v in zip(("amplitude", "ratio", "other"), c[:3]):
            stats[k] = stats.get(k, 0) + int(v)
        stats["max_log_ratio"] = max(stats.get("max_log_ratio", 0.0), float(c[3]))
    return total, bad, owner


def _log_parts(x):
    """(sign, log|x|) in float64 of a `LogArray` or a plain array."""
    if hasattr(x, "logabs"):
        return jnp.asarray(x.sign).astype(jnp.complex128), jnp.asarray(x.logabs).astype(jnp.float64)
    x = jnp.asarray(x).astype(jnp.complex128)
    a = jnp.abs(x)
    return jnp.where(a > 0, x / jnp.where(a > 0, a, 1.0), 0.0), jnp.log(a)


@partial(jax.jit, static_argnums=(6, 7, 8))
def _binned_classified(s, psi, segment, s_conn, psi_conn, h_conn, bins, nbins, psitilde):
    valid = _is_term(segment, h_conn)
    seg = jnp.where(valid, segment, 0)
    val32 = jnp.real(jnp.asarray(psi_conn / psi[seg]) * h_conn)          # exactly `_binned`'s value
    sc, lc = _log_parts(psi_conn)
    ss, ls = _log_parts(psi[seg])
    amp_ok = jnp.isfinite(lc) | jnp.isneginf(lc)                           # -inf is a true zero amplitude
    amp_ok = amp_ok & jnp.isfinite(jnp.abs(sc))
    own_ok = (jnp.isfinite(ls) | jnp.isneginf(ls)) & jnp.isfinite(jnp.abs(ss))
    bad = valid & ~jnp.isfinite(val32)
    amplitude = bad & ~amp_ok
    lr = jnp.where(amp_ok & own_ok & jnp.isfinite(lc) & jnp.isfinite(ls), lc - ls, 0.0)
    ratio = bad & amp_ok & own_ok & jnp.isfinite(lc) & jnp.isfinite(ls)
    other = bad & ~amplitude & ~ratio
    val64 = jnp.real(sc * jnp.conj(ss) * jnp.exp(lr) * h_conn.astype(jnp.complex128))
    val = jnp.where(valid & ~bad, val32.astype(jnp.float64), 0.0)
    if psitilde:
        val = jnp.where(ratio, val64, val)
    owner = jax.ops.segment_sum(bad.astype(jnp.int32), seg, num_segments=s.shape[0]) > 0
    counts = jnp.stack([jnp.sum(amplitude), jnp.sum(ratio), jnp.sum(other),
                        jnp.max(jnp.where(ratio, jnp.abs(lr), 0.0))]).astype(jnp.float64)
    return (jax.ops.segment_sum(val, bins(s[seg], s_conn), num_segments=nbins), jnp.sum(bad).astype(jnp.int32),
            owner, counts)


def _arrival(s0, s1, N, spin, default):
    """The site where an electron of `spin` sits in s1 but not in s0, or `default` if none does."""
    lo = 0 if spin == "u" else N
    gained = s1[:, lo:lo + N] > s0[:, lo:lo + N]
    return jnp.where(jnp.any(gained, axis=1), jnp.argmax(gained, axis=1), default)


def _pair_bins(lut, N, a, b):
    """The right term of a pair operator whose left pair is (a up, b dn): told by where the up and the down
    electron arrive, a (b) when that electron does not move."""
    def bins(s0, s1):
        return lut[_arrival(s0, s1, N, "u", a) * N + _arrival(s0, s1, N, "d", b)]
    return bins


def _hop_bins(N, spin):
    """The site j of c^dag_i c_j: where the electron arrives."""
    def bins(s0, s1):
        return _arrival(s0, s1, N, spin, 0)
    return bins


def _spin_bins(geo, a):
    """The bin c * N + r, r = j - a, for S^+_a S^-_j (c = 0) and S^-_a S^+_j (c = 1)."""
    N, ia, (ax, ay) = geo.N, geo.index(a), a

    def bins(s0, s1):
        up0, up1 = s0[:, :N], s1[:, :N]
        gain = jnp.argmax(up1 > up0, axis=1)        # the site the up electron moves to
        lose = jnp.argmax(up1 < up0, axis=1)        # and from
        c = jnp.where(lose == ia, 0, 1)             # S^+_a S^-_j moves it from a to j
        j = jnp.where(c == 0, gain, lose)
        return c * N + ((j // geo.Ly - ax) % geo.Lx) * geo.Ly + (j % geo.Ly - ay) % geo.Ly
    return bins


# ---------------------------------------------------------------- d-wave terms
class DwaveTerms:
    """F and G of the left bonds at R — the +x and +y bonds (`left="primary"`, the rest of C_p filled by
    translation) or all four (`left="all"`, no translation needed) — against every right bond and site."""

    def __init__(self, geo, R=(0, 0), left="primary"):
        if min(geo.Lx, geo.Ly) < 3:
            raise ValueError(f"a {geo.Lx}x{geo.Ly} torus has two bonds between one pair of sites; the d-wave terms "
                             "need both sides at least 3")
        self.geo, self.R = geo, tuple(R)
        dirs = DIRS[::2] if left == "primary" else DIRS
        self.left_bonds = sorted({geo.bond_at(self.R, d) for d in dirs})
        self.left_sites = sorted({p for b in self.left_bonds for p in geo.ends(b)})
        N = geo.N
        right = [(br, o) for br in geo.bonds for o in (0, 1)]
        self.F_keys = [(bl, oL, br, oR) for bl in self.left_bonds for oL in (0, 1) for br, oR in right]
        self.G_keys = [(i, j, sp) for i in self.left_sites for j in geo.sites for sp in SPINS]
        lut = np.full(N * N, -1, dtype=np.int32)    # (up arrival, dn arrival) -> right term
        for t, (br, oR) in enumerate(right):
            c, d = (geo.index(p) for p in geo.pair_sites(br, oR))
            lut[c * N + d] = t
        self._lut = jnp.asarray(lut)
        self._right = right
        self._pairs = []
        for bl in self.left_bonds:
            for oL in (0, 1):
                a, b = (geo.index(p) for p in geo.pair_sites(bl, oL))
                op = _sum([pair_creation(geo, bl, oL) @ pair_creation(geo, br, oR).H
                           for br, oR in right if (br, oR) != (bl, oL)])
                self._pairs.append((a, b, right.index((bl, oL)), op, _pair_bins(self._lut, N, a, b)))
        # the bin functions are built once: `_binned` is compiled per function object
        self._hops = [(geo.index(i), sp, _sum([hopping(i, j, sp) for j in geo.sites if j != i]),
                       _hop_bins(N, sp)) for i in self.left_sites for sp in SPINS]

    def measure(self, state, samples, sum_fn=None):
        """`sum_fn` defaults to `packed_sum` (`classified_sum` with its keywords bound re-reads saved configurations).
        (F, G, non-finite count, owner mask): sample means in the order of F_keys and G_keys; the mask marks
        the samples that owned a non-finite term."""
        s = samples.spins
        N, n = self.geo.N, s.shape[0]
        occ = s > 0
        F, bad, owner = [], 0, jnp.zeros(n, bool)
        for a, b, diag, op, bins in self._pairs:
            T, k, m = (sum_fn or packed_sum)(state, samples, op, bins, len(self._right))
            nn = jnp.sum(occ[:, a] & occ[:, N + b])        # A^dag A of the same pair: n_{a up} n_{b dn}
            F.append(T.at[diag].set(nn) / n)
            bad, owner = bad + k, owner | m
        G = {}
        for i, sp, op, bins in self._hops:
            T, k, m = (sum_fn or packed_sum)(state, samples, op, bins, N)
            G[i, sp] = T.at[i].set(jnp.sum(occ[:, i if sp == "u" else N + i])) / n
            bad, owner = bad + k, owner | m
        G = [jnp.stack([G[self.geo.index(p), "u"], G[self.geo.index(p), "d"]], axis=1).reshape(-1)
             for p in self.left_sites]
        return jnp.concatenate(F), jnp.concatenate(G), int(bad), owner


# ---------------------------------------------------------------- transverse spin terms
class TransverseTerms:
    """T[c, r] = <S^+_a S^-_{a+r}> (c = 0) and <S^-_a S^+_{a+r}> (c = 1) for each reference site a."""

    def __init__(self, geo, refs):
        self.geo, self.refs = geo, [tuple(a) for a in refs]
        self._ops = [_sum([spin_flip(a, j, c) for j in geo.sites if j != a for c in (0, 1)]) for a in self.refs]
        self._bins = [_spin_bins(geo, a) for a in self.refs]

    def measure(self, state, samples, sum_fn=None):
        """(T (nref, 2, N), non-finite count, owner mask); `sum_fn` as `DwaveTerms.measure`'s."""
        s = samples.spins
        geo, N, n = self.geo, self.geo.N, s.shape[0]
        out, bad, owner = [], 0, jnp.zeros(n, bool)
        for a, op, bins in zip(self.refs, self._ops, self._bins):
            ia = geo.index(a)
            T, k, m = (sum_fn or packed_sum)(state, samples, op, bins, 2 * N)
            nu, nd = s[:, ia] > 0, s[:, N + ia] > 0
            T = T.reshape(2, N).at[:, 0].set(jnp.stack([jnp.sum(nu & ~nd), jnp.sum(nd & ~nu)]).astype(T.dtype))
            out.append(T / n)
            bad, owner = bad + k, owner | m
        return jnp.stack(out), int(bad), owner


def reference_sites(Lx, Ly, nref=8):
    """nref sites along the diagonal (x, x mod Ly)."""
    return [(x, x % Ly) for x in range(0, Lx, max(Lx // nref, 1))][:nref]


# ---------------------------------------------------------------- diagonal estimators
def sz_grid(spins, Lx, Ly):
    N = Lx * Ly
    s = spins.astype(jnp.float64)
    return ((s[:, :N] - s[:, N:]) / 4).reshape(-1, Lx, Ly)


def dn_grid(spins, Lx, Ly, nbar):
    N = Lx * Ly
    s = spins.astype(jnp.float64)
    return ((s[:, :N] + s[:, N:]) / 2 + 1).reshape(-1, Lx, Ly) - nbar


def autocorr(g):
    """The batch mean of (1/N) sum_i g_i g_{i+r}, shape (N,), r in site order."""
    N = g.shape[1] * g.shape[2]
    A = jnp.real(jnp.fft.ifft2(jnp.abs(jnp.fft.fft2(g, axes=(1, 2))) ** 2, axes=(1, 2)))
    return jnp.mean(A, axis=0).reshape(-1) / N


# ---------------------------------------------------------------- assembly
def keys_to_array(F_keys, G_keys):
    """F keys ((x, y, axis), oL, (x, y, axis), oR) and G keys ((x, y), (x, y), spin) as int arrays (n, 8), (n, 5)."""
    F = np.array([(*bl, oL, *br, oR) for bl, oL, br, oR in F_keys], dtype=np.int32)
    G = np.array([(*i, *j, SPINS.index(sp)) for i, j, sp in G_keys], dtype=np.int32)
    return F, G


def keys_from_array(F, G):
    """The inverse of `keys_to_array`."""
    F = [((a, b, c), o1, (d, e, f), o2) for a, b, c, o1, d, e, f, o2 in np.asarray(F).tolist()]
    G = [((a, b), (c, d), SPINS[sp]) for a, b, c, d, sp in np.asarray(G).tolist()]
    return F, G


class PairIndex:
    """Which F and G entries make C_p(r) for every displacement r, with the form factors.

    A left bond (site) that was not measured is the measured one of the same axis (R) translated:
    F[bl + t, oL, br + t, oR] = F[bl, oL, br, oR], exact for a translation-invariant state."""

    def __init__(self, geo, F_keys, G_keys, R=(0, 0)):
        R = tuple(R)
        fpos = {k: i for i, k in enumerate(F_keys)}
        gpos = {k: i for i, k in enumerate(G_keys)}
        measured = sorted({k[0] for k in F_keys})
        self.translated = False

        def f(bl, oL, br, oR):
            if (bl, oL, br, oR) in fpos:
                return fpos[bl, oL, br, oR]
            self.translated = True
            bp = next(b for b in measured if b[2] == bl[2])
            t = (bp[0] - bl[0], bp[1] - bl[1])
            return fpos[bp, oL, (*geo.shift(br[:2], t), br[2]), oR]

        def g(i, j, sp):
            if (i, j, sp) in gpos:
                return gpos[i, j, sp]
            self.translated = True
            return gpos[R, geo.shift(j, (R[0] - i[0], R[1] - i[1])), sp]

        N = geo.N
        self.fidx = np.empty((N, 16, 4), dtype=np.int64)
        self.gidx = np.empty((N, 16, 4, 2), dtype=np.int64)
        self.form = np.empty(16)
        for n, dr in enumerate(geo.sites):
            r = geo.shift(R, dr)
            m = 0
            for dl in DIRS:
                bl = geo.bond_at(R, dl)
                for d2 in DIRS:
                    br = geo.bond_at(r, d2)
                    self.form[m] = FORM[dl] * FORM[d2]
                    for k, (oL, oR) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                        self.fidx[n, m, k] = f(bl, oL, br, oR)
                        (pu, pd), (qu, qd) = geo.pair_sites(bl, oL), geo.pair_sites(br, oR)
                        # <c^dag_pu c^dag_pd c_qd c_qu> -> <c^dag_pu c_qu><c^dag_pd c_qd>
                        self.gidx[n, m, k] = (g(pu, qu, "u"), g(pd, qd, "d"))
                    m += 1

    def pair_correlation(self, F, G=None):
        """C_p(r), shape (..., N); <Delta^dag_R Delta_{R+r}> alone when G is None."""
        F = np.asarray(F, dtype=np.float64)
        out = 0.5 * F[..., self.fidx].sum(axis=-1)
        if G is not None:
            G = np.asarray(G, dtype=np.float64)
            out = out - 0.5 * (G[..., self.gidx[..., 0]] * G[..., self.gidx[..., 1]]).sum(axis=-1)
        return (out * self.form).sum(axis=-1) / 16


BLOCKS, MIN_BLOCKS = (1, 2, 4, 8), 8


def jackknife(fn, *series):
    """(value, error): fn of the sweep means, its error the largest block-jackknife error over blocks of 1, 2, 4
    and 8 sweeps that leave at least 8 blocks (NaN when none does)."""
    x = [np.asarray(a, dtype=np.float64) for a in series]
    value = fn(*[a.mean(axis=0) for a in x])
    errs = []
    for b in BLOCKS:
        nb = x[0].shape[0] // b
        if nb < MIN_BLOCKS:
            continue
        xb = [a[:nb * b].reshape(nb, b, *a.shape[1:]).mean(axis=1) for a in x]
        tot = [a.sum(axis=0) for a in xb]
        reps = np.array([fn(*[(t - a[i]) / (nb - 1) for t, a in zip(tot, xb)]) for i in range(nb)])
        errs.append(np.sqrt(nb - 1) * reps.std(axis=0))
    return value, (np.max(np.stack(errs), axis=0) if errs else np.full(np.shape(value), np.nan))


def structure_factor(C, Lx, Ly):
    """S(k) = sum_r e^{ik.r} C(r) on the momentum grid, (..., Lx, Ly)."""
    C = np.asarray(C, dtype=np.float64)
    return (np.fft.ifft2(C.reshape(*C.shape[:-1], Lx, Ly), axes=(-2, -1)) * (Lx * Ly)).real


def stripe_momenta(Lx, Ly, tprime=0.0):
    """Integer momenta of (pi -+ q, pi), (pi, pi -+ q) and of the charge's +-(2q, 0), +-(0, 2q) — each direction
    only where q lies on its momentum grid: on a strip whose short side does not fit q the stripe can run along x alone, so only its peaks;
    on a square torus both, as before. Raises ValueError for an unrecorded t', for an odd side (pi is not on its
    momentum grid) or when neither direction fits."""
    if Lx % 2 or Ly % 2:
        raise ValueError(f"the stripe peaks sit at momentum pi, which an odd side does not have ({Lx}x{Ly})")
    if tprime not in STRIPE_Q:
        raise ValueError(f"no stripe wavevector recorded for t' = {tprime}")
    num, den = STRIPE_Q[tprime]
    spin, charge = [], []
    if Lx * num % (2 * den) == 0:
        nx = Lx * num // (2 * den)
        spin += [(Lx // 2 - nx, Ly // 2), (Lx // 2 + nx, Ly // 2)]
        charge += [(2 * nx, 0), (Lx - 2 * nx, 0)]
    if Ly * num % (2 * den) == 0:
        ny = Ly * num // (2 * den)
        spin += [(Lx // 2, Ly // 2 - ny), (Lx // 2, Ly // 2 + ny)]
        charge += [(0, 2 * ny), (0, Ly - 2 * ny)]
    if not spin:
        raise ValueError(f"q is not on the momentum grid of {Lx}x{Ly}")
    return tuple(spin), tuple(charge)


def assemble(series, Lx, Ly, tprime=0.0):
    """The scalars and maps of one measurement from its per-sweep series (dict of arrays, first axis the sweep)."""
    geo = Torus(Lx, Ly)
    N = geo.N
    out = {}
    if "E_mean" in series:
        e, de = jackknife(lambda m: m / N, series["E_mean"])
        v, dv = jackknife(lambda m, q: (q - m ** 2) / N, series["E_mean"], series["E_sqr"])
        out.update(nsweeps=int(len(series["E_mean"])), E_per_site=float(e), E_err=float(de),
                   var_per_site=float(v), var_err=float(dv))
    if "F" in series:
        idx = PairIndex(geo, *keys_from_array(series["F_keys"], series["G_keys"]))
        dist = geo.min_image(np.array(geo.sites))
        far = dist >= dist.max() / 2 - 1e-9
        Cp, dCp = jackknife(lambda F, G: idx.pair_correlation(F, G), series["F"], series["G"])
        d2, dd2 = jackknife(lambda F, G: idx.pair_correlation(F, G)[far].mean(), series["F"], series["G"])
        out.update(C_p=Cp, C_p_err=dCp, Delta2_SC=float(d2), Delta2_SC_err=float(dd2),
                   Delta_SC=float(np.sqrt(d2)) if d2 > 0 else float("nan"), r_max=float(dist.max() / 2),
                   n_far=int(far.sum()), translation_fill=idx.translated)
    try:
        ks, qs = stripe_momenta(Lx, Ly, tprime)
    except ValueError:                      # a lattice without the stripe wavevector: the maps alone
        ks = qs = None

    def at(S, ms):
        return np.mean([S[..., a % Lx, b % Ly] for a, b in ms], axis=0) / N

    if "Tpm" in series:
        Cs_series = np.asarray(series["Czz"]) + 0.5 * (np.asarray(series["Tpm"]).mean(axis=1)
                                                       + np.asarray(series["Tmp"]).mean(axis=1))
        Cs, dCs = jackknife(lambda x: x, Cs_series)
        out.update(C_s=Cs, C_s_err=dCs)
        if ks is not None:
            m2, dm2 = jackknife(lambda x: at(structure_factor(x, Lx, Ly), ks), Cs_series)
            out.update(M2_str=float(m2), M2_str_err=float(dm2), k_alpha=np.array(ks))
    if "Cnn" in series:
        Cc, dCc = jackknife(lambda x: x, series["Cnn"])
        out.update(C_c=Cc, C_c_err=dCc)
        if qs is not None:
            out.update(q_c=np.array(qs))
    return out
