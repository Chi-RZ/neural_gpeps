"""A projected Gaussian tensor network's amplitude as ONE sparse skew matrix.

THE STATE.  Site `s` is `exp(1/2 c_s^+ Z_s c_s^+)|0>` over its modes, `Z_s`
real skew; modes are numbered globally site after site. A bond joins mode `p`
of one site to mode `q` of another through `<0_pq| (1 + sigma c_q c_p)/sqrt2`.
The amplitude of a physical configuration is
`<0| c_{n_k} ... c_{n_1} prod_bonds(...) prod_s exp(...)|0>` over the occupied
physical modes `n_1 < ... < n_k`.

THE MATRIX.  Wick's theorem gives

    psi(n) = epsilon(n) * 2^{-b/2} * Pf(M),   M = (blockdiag_s Z_s + sigma-couplings)_{SS},

with `S` every virtual mode and the occupied physical modes, in global order,
`M[p_a, q_b] = +sigma` in the bond's own orientation (`p_a` on the side the
operator names first), and `epsilon(n)` a sign of the occupied positions alone
(`configuration_sign`); the tests hold both to a brute-force Fock-space amplitude.
Every entry of `M`
is a raw site entry or a bond constant: nothing is divided, so there is no
chart, reference or pole — `Pf M` vanishes only where the amplitude does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
            # M_pq = BOND_SIGN*sigma in the bond's own orientation (a then b)
            out.append((p, q, sigma) if p < q else (q, p, -sigma))
        return out


def kept_modes(net, occupation):
    """Global modes of `M`: every non-physical mode, and the occupied physical ones.

    `occupation[s]` is a tuple of 0/1 per physical mode of site `s`.
    """
    keep = []
    for s, m in enumerate(net.modes):
        phys = dict(zip(net.physical[s], occupation[s]))
        for local in range(m):
            if local not in phys or phys[local]:
                keep.append(net.global_mode(s, local))
    return keep


BOND_SIGN = 1  # M[p_a, q_b] = BOND_SIGN * sigma, in the bond's orientation


def assemble(net, z_sites, occupation, dtype=float):
    """`(M, keep)`: the skew matrix over the kept modes, entries copied, never divided."""
    total = int(net.offsets[-1])
    full = np.zeros((total, total), dtype=dtype)
    for s, z in enumerate(z_sites):
        o = net.offsets[s]
        full[o:o+len(z), o:o+len(z)] = z
    for p, q, sigma in net.bond_pairs():
        full[p, q] += BOND_SIGN*sigma
        full[q, p] -= BOND_SIGN*sigma
    keep = kept_modes(net, occupation)
    return full[np.ix_(keep, keep)], keep


def configuration_sign(net, occupation):
    """`epsilon(n)`: psi(n) = epsilon(n) 2^{-b/2} Pf(M).

    Expanding every bond operator, `2^{b/2} psi = sum_K sigma_K chi(K) Pf(Z_{T_K})`
    over the sets K of bonds whose pair is occupied, `chi(K)` the sign of putting
    `c_{n_k}..c_{n_1} prod_{b in K} c_q c_p` in descending order (which
    `<0|c_{t_m}..c_{t_1}|Phi> = Pf Z_T` needs). `Pf M` expands into the same
    terms up to one K-independent factor, read at K = all bonds: `prod sigma`
    times chi(all).
    """
    occupied = sorted(net.global_mode(s, l) for s in range(len(net.modes))
                      for l, bit in zip(net.physical[s], occupation[s]) if bit)
    sequence = occupied[::-1]
    sign = 1
    for sa, ma, sb, mb, sigma in net.bonds:
        sequence += [net.global_mode(sb, mb), net.global_mode(sa, ma)]
        sign *= sigma
    # inversions relative to descending order
    inversions = sum(1 for i in range(len(sequence)) for j in range(i+1, len(sequence))
                     if sequence[i] < sequence[j])
    return sign*(-1)**inversions


def log_scale(net):
    """`log 2^{-b/2}`, the bonds' normalisation."""
    return -0.5*len(net.bonds)*np.log(2.0)
