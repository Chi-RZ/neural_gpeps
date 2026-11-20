"""The energy, d-wave pair correlations, stripe spin order and charge correlations of a state with fixed weights.

Chains (`vmc.u1_mix_sampler`, seed `seed`) burn in for `thermal` proposals a chain — from random configurations, or
from saved chains (`chains_in`) — and are then read `nsweeps` times, `sweeps` proposals apart, `nsamples` samples
each. Every sweep measures the selected `channels`:

* `energy` — E_loc and E_loc^2 over the samples whose local energy is finite (the others counted);
* `dwave` — F and G of the +x and +y bonds at the origin against every bond and site (`orderparams.DwaveTerms`);
* `spin` — S^z S^z by FFT and the transverse terms of `nref` reference sites on the diagonal (`TransverseTerms`);
* `charge` — the density correlation by FFT.

NON-FINITE TERMS. The sampler never stands on a configuration whose contraction overflows its delay slots (such a
proposal is rejected), so every sample's own amplitude is finite; a connected configuration's can overflow. Every
estimator here is that of psi-tilde = psi x 1[the contraction completes]: a connected term whose amplitude overflowed
adds zero and its sample is kept. In this first pass every non-finite correlator term is zero, including the rare
one whose two amplitudes are finite but whose float32 ratio psi(s')/psi(s) is not; a local energy that is not
finite is re-evaluated term by term (`psitilde_local_energy`), its overflowed amplitudes at zero and a float32 ratio
overflow at its float64 value. A sample whose OWN amplitude is not finite when re-evaluated here (the sampler's
evaluation found it finite, so this needs float32 rounding to flip a pivot at the threshold while the delayed count
sits at the slot limit) has psi-tilde = 0 and no weight: its local energy is excluded and counted
(`E_own_overflow`), never kept with its diagonal term alone. Each sweep records the counts (`E_restored`,
`E_excluded` for a sample still not finite, `E_overflow_terms`, `E_own_overflow`, `dwave_nonfinite`,
`spin_nonfinite`) and saves the configurations of every such sample (`bad_spins`,
`bad_meta`). They can be re-read afterwards: `reread` (`nngtn-measure --reread-series`) re-evaluates them term by
term — every term classified, a ratio overflow counted at its float64 value, an overflowed amplitude recomputed at
more delay slots (`--reread-slots`, where it is finite) — and reports how much each sweep's means move. A sweep with
any non-finite mean left is excluded whole. For full float32 matmul precision (no TF32)
call `set_matmul_precision("highest")` (or set `JAX_DEFAULT_MATMUL_PRECISION=highest`) before the
first array exists.

Writes, for an output prefix `P`: `P-series.npz` (the per-sweep means, rewritten every `save_every` sweeps),
`P-chains.npy` (the chains after the latest sweep read), and at the end `P.json` (the assembled scalars) and
`P-maps.npz` (C_p, C_s, C_c and their errors). `orderparams.assemble` re-derives the scalars from the series alone.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CHANNELS = ("energy", "dwave", "spin", "charge")

__all__ = ["CHANNELS", "set_matmul_precision", "psitilde_local_energy", "measure", "reread"]


def set_matmul_precision(precision):
    """The process's default matmul precision (`"highest"`: full float32, no TF32), as
    `JAX_DEFAULT_MATMUL_PRECISION` sets it; `None` leaves jax's default (TF32 for float32 on Ampere and later)."""
    import jax
    if precision:
        jax.config.update("jax_default_matmul_precision", precision)


def _one_bin(s0, s1):
    import jax.numpy as jnp
    return jnp.zeros(s0.shape[0], jnp.int32)


def psitilde_local_energy(state, H, s, stats=None):
    """The local energy of ONE configuration `s` (shape (1, 2N)) of psi-tilde: H's off-diagonal terms through
    `orderparams.classified_sum` — an overflowed connected amplitude adds zero, a float32 ratio overflow its float64
    value — plus the diagonal. `stats` collects the counts ("amplitude", "ratio", "other", "max_log_ratio", "own").

    A configuration whose OWN amplitude is not finite here has psi-tilde(s) = 0: it carries no weight, so its local
    energy is NaN (the caller excludes and counts it), never the diagonal alone. The sampler rejects such a
    configuration, so this happens only when this evaluation disagrees with the sampler's (float32 rounding flipping
    a pivot at the threshold while the delayed count sits at the slot limit)."""
    import numpy as np
    import quantax as qtx
    from quantax.sampler import Samples

    from . import orderparams as op
    psi = state(s)
    own = np.asarray(qtx.utils.to_replicated_numpy(psi.logabs), np.float64)
    if not np.all(np.isfinite(own)):
        if stats is not None:
            stats["own"] = stats.get("own", 0) + 1
        return float("nan")
    smp = Samples(s, psi=psi)
    off, _, _ = op.classified_sum(state, smp, H, _one_bin, 1, {} if stats is None else stats, psitilde=True)
    diag = np.asarray(qtx.utils.to_replicated_numpy(H.apply_diag(s)))
    return float(np.real(diag)[0] + float(off[0]))


def measure(state, H, out_prefix, *, nparticles, nsamples=8192, nsweeps=64, thermal=10240, sweeps=1024, seed=1,
            chains_in=None, tprime=0.0, nref=8, channels=CHANNELS, save_every=4, extra=None, log=None):
    """Measure `state` (a quantax `Variational`, projected or not) under `H`; returns the assembled scalars (process
    0) or `None`. `extra` is recorded in the series and the JSON (e.g. the checkpoint path, the symmetry)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import quantax as qtx
    from quantax.sampler import Samples

    from . import orderparams as op
    from .chains import load_chains
    from .vmc import u1_mix_sampler

    channels = set(channels)
    if not channels <= set(CHANNELS):
        raise ValueError(f"channels must be a subset of {CHANNELS}")
    writer = jax.process_index() == 0
    if log is None:
        def log(msg):
            if writer:
                print(msg, flush=True)
    Lx, Ly = int(state.model.Lx), int(state.model.Ly)
    N = Lx * Ly
    if qtx.get_sites().Nsites != N:
        raise ValueError("quantax's lattice is not the model's: call lattice.make_sites first")
    nbar = sum(nparticles) / N
    geo = op.Torus(Lx, Ly)
    to_np = qtx.utils.to_replicated_numpy
    if chains_in is None or isinstance(chains_in, (str, os.PathLike)):
        chains_label = str(chains_in or "")
    else:
        chains_label = f"<{np.shape(chains_in)[0]} chains given as an array>"

    series = dict(Lx=Lx, Ly=Ly, nbar=nbar, nsamples=nsamples, thermal=thermal, sweeps=sweeps, tprime=tprime,
                  chains_in=chains_label, **{k: str(v) for k, v in (extra or {}).items()})
    if "dwave" in channels:
        dw = op.DwaveTerms(geo)
        series["F_keys"], series["G_keys"] = op.keys_to_array(dw.F_keys, dw.G_keys)
    if "spin" in channels:
        refs = op.reference_sites(Lx, Ly, nref)
        tr = op.TransverseTerms(geo, refs)
        series["ref_sites"] = np.array(refs)
    log(f"{Lx}x{Ly}, {jax.device_count()} devices over {jax.process_count()} processes; "
        f"channels={sorted(channels)}; {nsamples} samples a sweep, {nsweeps} sweeps, "
        f"burn-in {thermal} from {chains_label or 'random chains'}, {sweeps} proposals between sweeps; t' = {tprime}")

    out_prefix = Path(out_prefix)
    if writer:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = Path(f"{out_prefix}-series.npz")
    chains_path = Path(f"{out_prefix}-chains.npy")
    rec, kept, excluded = {}, [], []
    bad_spins, bad_meta = [], []          # the non-finite samples' configurations and (sweep, energy, dwave, spin)
    last = {}                             # the chains after the latest sweep read, and its index

    def save():
        chains = np.asarray(to_np(last["s"]), np.int8) if last else None   # every process: a replicated gather
        if writer:
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **series, **{k: np.stack(v) for k, v in rec.items()},
                     kept=np.array(kept), excluded=np.array(excluded),
                     bad_spins=(np.concatenate(bad_spins) if bad_spins else np.zeros((0, 2 * N), np.int8)),
                     bad_meta=(np.array(bad_meta, np.int32).reshape(-1, 4)),
                     chains_sweep=np.int32(last["n"] if last else -1))
            os.replace(tmp, path)
            if chains is not None:
                tmpc = chains_path.with_suffix(".tmp.npy")
                np.save(tmpc, chains)
                os.replace(tmpc, chains_path)

    t0 = time.time()
    if chains_in is not None and not (isinstance(chains_in, str) and not chains_in):
        spins0 = (load_chains(chains_in, nsamples) if isinstance(chains_in, (str, os.PathLike))
                  else np.asarray(chains_in, np.int8))
        sampler = u1_mix_sampler(state, nsamples, thermal_steps=0, sweep_steps=sweeps, seed=seed)
        sampler.reset(nsweeps=thermal, initial_spins=jnp.asarray(spins0))
    else:
        sampler = u1_mix_sampler(state, nsamples, thermal_steps=thermal, sweep_steps=sweeps, seed=seed)
    log(f"burn-in done, {time.time() - t0:.0f} s")
    for n in range(nsweeps):
        a = time.time()
        s = sampler.sweep().spins
        last["s"], last["n"] = s, n
        samples = Samples(s, psi=state(s))
        cur = {}
        flags = {}                        # channel -> mask (nsamples,) of the samples that owned a non-finite value
        if "energy" in channels:
            e = to_np(jnp.real(H.Oloc(state, samples))).astype(np.float64)
            bad_e = ~np.isfinite(e)
            e_stats = {}
            if bad_e.any():                   # psi-tilde: the overflowed connected terms at zero, the sample kept
                s_np = np.asarray(to_np(s))
                for r in np.flatnonzero(bad_e):
                    e[r] = psitilde_local_energy(state, H, jnp.asarray(s_np[r:r + 1]), e_stats)
            ok = np.isfinite(e)
            cur["E_mean"], cur["E_sqr"], cur["E_excluded"] = e[ok].mean(), (e[ok] ** 2).mean(), int((~ok).sum())
            cur["E_restored"] = int((bad_e & ok).sum())
            cur["E_overflow_terms"] = int(e_stats.get("amplitude", 0))
            cur["E_own_overflow"] = int(e_stats.get("own", 0))    # samples excluded: their own amplitude overflowed
            flags["energy"] = bad_e
        if {"spin", "charge"} & channels:
            if "spin" in channels:
                cur["Czz"] = to_np(op.autocorr(op.sz_grid(s, Lx, Ly)))
                cur["sz_site"] = to_np(jnp.mean(op.sz_grid(s, Lx, Ly), axis=0))
            if "charge" in channels:
                g = op.dn_grid(s, Lx, Ly, nbar)
                cur["Cnn"] = to_np(op.autocorr(g))
                cur["n_site"] = to_np(jnp.mean(g, axis=0)) + nbar
        if "spin" in channels:
            T, bad, owner = tr.measure(state, samples)
            T = to_np(T)
            cur["Tpm"], cur["Tmp"], cur["spin_nonfinite"] = T[:, 0], T[:, 1], bad
            flags["spin"] = np.asarray(to_np(owner), bool)
        if "dwave" in channels:
            F, G, bad, owner = dw.measure(state, samples)
            cur["F"], cur["G"], cur["dwave_nonfinite"] = to_np(F), to_np(G), bad
            flags["dwave"] = np.asarray(to_np(owner), bool)
        any_bad = np.zeros(next(iter(flags.values())).shape[0], bool) if flags else np.zeros(0, bool)
        for v in flags.values():
            any_bad |= v
        n_bad = int(any_bad.sum())
        if n_bad:
            bad_spins.append(np.asarray(to_np(s), np.int8)[any_bad])
            bad_meta += [[n] + [int(flags[c][i]) if c in flags else 0 for c in ("energy", "dwave", "spin")]
                         for i in np.flatnonzero(any_bad)]
        if not all(np.all(np.isfinite(v)) for v in cur.values()):
            excluded.append(n)
            log(f"sweep {n}: non-finite values, excluded")
            continue
        kept.append(n)
        for k, v in cur.items():
            rec.setdefault(k, []).append(np.asarray(v))
        msg = f"sweep {n}: {time.time() - a:.1f} s"
        if "energy" in channels:
            msg += (f", E/N = {cur['E_mean'] / N:.6f}, var/N = {(cur['E_sqr'] - cur['E_mean'] ** 2) / N:.4f}, "
                    f"E restored {cur['E_restored']}, E excluded {cur['E_excluded']}")
        if "dwave" in channels:
            msg += f", dwave non-finite {cur['dwave_nonfinite']}"
        if "spin" in channels:
            msg += f", spin non-finite {cur['spin_nonfinite']}"
        msg += f", samples saved {n_bad}"
        log(msg)
        if (n + 1) % save_every == 0:
            save()
    save()
    if not (writer and kept):
        return None
    res = op.assemble({**series, **{k: np.stack(v) for k, v in rec.items()}}, Lx, Ly, tprime=tprime)
    maps = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    scalars.update(sweeps_kept=len(kept), sweeps_excluded=excluded, seconds=time.time() - t0,
                   settings=dict(nsamples=nsamples, nsweeps=nsweeps, thermal=thermal, sweeps=sweeps, seed=seed,
                                 chains_in=chains_label, tprime=tprime, nref=nref,
                                 channels=sorted(channels), **(extra or {})))
    for k in ("E_excluded", "E_restored", "E_overflow_terms", "E_own_overflow", "dwave_nonfinite", "spin_nonfinite"):
        if k in rec:
            scalars[k + "_total"] = int(np.sum(rec[k]))
    np.savez(f"{out_prefix}-maps.npz", **maps)
    Path(f"{out_prefix}.json").write_text(json.dumps(scalars, indent=1, default=str))
    log(json.dumps({k: v for k, v in scalars.items() if k != "settings"}, default=str))
    return scalars


def reread(state, state_more, H, series, *, log=None):
    """A measurement's saved non-finite configurations re-evaluated with more delay slots.

    `series` is a measurement's per-sweep series (`measure`'s, or one of the same format) with its saved
    configurations (`bad_spins`, `bad_meta`); `state` is the measured state (the delay slots it was measured with)
    and `state_more` the same weights under more slots, where the amplitudes that overflowed are finite. For every
    sweep with saved configurations, its correlator means move by (more - measured) over those configurations, both
    psi-tilde readings, times their number over the sweep's sample count; its energy samples are replaced by their
    `state_more` local energies (or put back, in a series that excluded them). Returns `(re-read series, report)`;
    the report lists, per sweep, the configurations, the largest change of each mean, the energies and the term
    counts under both plans, so the gap between psi-tilde and the wider-buffer state can be read off. The re-read
    series is that diagnostic, not a variational estimate: its samples come from `state` while the re-read terms
    use `state_more`; the measured series is psi-tilde's."""
    import time
    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import quantax as qtx
    from quantax.sampler import Samples

    from . import orderparams as op

    if log is None:
        def log(msg):
            if jax.process_index() == 0:
                print(msg, flush=True)
    if "bad_spins" not in series or "bad_meta" not in series:
        raise ValueError("the series holds no saved configurations (bad_spins, bad_meta): nothing to re-read")
    to_np = qtx.utils.to_replicated_numpy
    Lx, Ly = int(series["Lx"]), int(series["Ly"])
    N, nsamples = Lx * Ly, int(series["nsamples"])
    kept = [int(k) for k in series["kept"]]
    bad_spins, bad_meta = np.asarray(series["bad_spins"]), np.asarray(series["bad_meta"])
    geo = op.Torus(Lx, Ly)
    dw = op.DwaveTerms(geo) if "F" in series else None
    tr = op.TransverseTerms(geo, [tuple(int(v) for v in a) for a in series["ref_sites"]]) if "Tpm" in series else None
    fixed = {k: np.array(series[k], np.float64) for k in ("E_mean", "E_sqr", "F", "G", "Tpm", "Tmp") if k in series}
    excluded = np.array(series.get("E_excluded", np.zeros(len(kept))), np.int64)

    def correlators(st, s, stats):
        smp = Samples(s, psi=st(s))
        fn = partial(op.classified_sum, stats=stats, psitilde=True)
        out = {}
        if dw is not None:
            F, G, _, _ = dw.measure(st, smp, sum_fn=fn)
            out["F"], out["G"] = np.asarray(to_np(F), np.float64), np.asarray(to_np(G), np.float64)
        if tr is not None:
            T, _, _ = tr.measure(st, smp, sum_fn=fn)
            T = np.asarray(to_np(T), np.float64)
            out["Tpm"], out["Tmp"] = T[:, 0], T[:, 1]
        return out

    report = dict(nsamples=nsamples, sweeps=[], terms={"measured": {}, "more": {}})
    t0 = time.time()
    for n in sorted(set(int(x) for x in bad_meta[:, 0])):
        if n not in kept:
            report["sweeps"].append(dict(sweep=n, skipped="sweep excluded from the series"))
            continue
        i = kept.index(n)
        rows = np.flatnonzero(bad_meta[:, 0] == n)
        s = jnp.asarray(bad_spins[rows])
        m = len(rows)
        entry = dict(sweep=n, configurations=m)
        if dw is not None or tr is not None:
            st_a, st_b = {}, {}
            a, b = correlators(state, s, st_a), correlators(state_more, s, st_b)
            for k in a:
                delta = (b[k] - a[k]) * m / nsamples
                fixed[k][i] += delta
                entry[f"max_abs_delta_{k}"] = float(np.abs(delta).max())
            entry["terms_measured"], entry["terms_more"] = st_a, st_b
            for key, st in (("measured", st_a), ("more", st_b)):
                for k, v in st.items():
                    report["terms"][key][k] = (max if k == "max_log_ratio" else (lambda x, y: x + y))(
                        report["terms"][key].get(k, 0), v)
        e_rows = [r for r in rows if bad_meta[r, 1]]
        if e_rows and "E_mean" in fixed:
            e_a = [psitilde_local_energy(state, H, jnp.asarray(bad_spins[r:r + 1])) for r in e_rows]
            e_b = [psitilde_local_energy(state_more, H, jnp.asarray(bad_spins[r:r + 1])) for r in e_rows]
            # A flagged sample whose psi-tilde energy is finite was restored in the first pass and is in the
            # sweep's means: its value is replaced. One whose energy is not was excluded: it is put back. A sweep
            # can hold both kinds, so each sample is classified on its own.
            count = nsamples - int(excluded[i])
            for x, y in zip(e_a, e_b):
                if np.isfinite(x) and np.isfinite(y):
                    fixed["E_mean"][i] += (y - x) / count
                    fixed["E_sqr"][i] += (y * y - x * x) / count
            back = np.asarray([y for x, y in zip(e_a, e_b) if not np.isfinite(x) and np.isfinite(y)], np.float64)
            if back.size:
                total = count + back.size
                fixed["E_mean"][i] = (count * fixed["E_mean"][i] + back.sum()) / total
                fixed["E_sqr"][i] = (count * fixed["E_sqr"][i] + (back ** 2).sum()) / total
                excluded[i] = max(0, int(excluded[i]) - back.size)
            entry.update(E_measured_per_site=[x / N for x in e_a], E_more_per_site=[y / N for y in e_b])
        report["sweeps"].append(entry)
        log(f"reread sweep {n}: {m} configurations, {len(e_rows)} energies, {time.time() - t0:.0f} s")
    out = dict(series)
    out.update(fixed)
    out["E_excluded"] = excluded
    return out, report
