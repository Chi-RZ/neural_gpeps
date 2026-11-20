"""Variational Monte Carlo for the t-t'-U Hubbard model (`system.hamiltonian`) with quantax: the sampler, the SR
optimiser (AdamSR with an LSMR solve) and the step-norm cap, and the step loop.

The construction order is fixed (the Hamiltonian, the projected `Variational` state, the chains — built last, as
their seed resets quantax's global key — the solver and the optimiser), and so is the step (sweep, AdamSR's step,
the norm cap read at `step + cap_step_offset`, the update): a seed then names one run.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .system import diagonal_bonds, hubbard_operator, square_bonds

__all__ = ["at_precision", "u1_mix_sampler", "chains_sampler", "build_solver", "build_optimiser", "step_cap",
           "scale_step", "run"]


# ---- the local energy at a chosen precision -----------------------------------------------------------------------
def at_precision(H, precision, dual=None):
    """`H` whose `Oloc` runs under the matmul `precision` (`"highest"`), the sampler's psi dropped so psi(s) and
    every connected psi(s') are evaluated together at that precision; `Oloc` then fills the reweighting factor
    `|psi|^2 / prob`, so the energy and the force are those of the re-evaluated state. `None`/`""`: `H` unchanged.

    `dual` (a dict, or `None`): before the re-evaluation, the sampler's own estimator is read on the same samples
    with its psi kept — the local energy the run would use without the re-evaluation — and its mean (`e`),
    population variance (`var`) and non-finite count (`nonfinite`) are written into `dual`, together with the
    reweighting's effective sample fraction `ess` = (mean r)^2 / mean(r^2) once `Oloc` has filled `r`. The
    reweighted estimator is unchanged: `samples.r` is put back before the re-evaluation."""
    if not precision:
        return H
    import jax
    import jax.numpy as jnp
    oloc = H.Oloc

    def oloc_at_precision(state, samples, *a, **k):
        if dual is not None:
            r_in = samples.r
            e0 = jnp.real(oloc(state, samples, *a, **k))
            ok0 = jnp.isfinite(e0)
            n0 = jnp.count_nonzero(ok0)
            m0 = jnp.sum(jnp.where(ok0, e0, 0)) / n0
            dual["e"] = m0
            dual["var"] = jnp.sum(jnp.where(ok0, (e0 - m0) ** 2, 0)) / n0
            dual["nonfinite"] = e0.size - n0
            samples.r = r_in
        samples.psi = None
        with jax.default_matmul_precision(precision):
            out = oloc(state, samples, *a, **k)
        if dual is not None and samples.r is not None:
            r = jnp.real(jnp.asarray(samples.r))
            rv = jnp.where(jnp.isfinite(r), r, 0)
            dual["ess"] = jnp.mean(rv) ** 2 / jnp.mean(rv ** 2)
        return out

    H.Oloc = oloc_at_precision
    return H


# ---- sampler ------------------------------------------------------------------------------------------------------
def u1_mix_sampler(state, n_samples, thermal_steps, sweep_steps, seed=0):
    """Metropolis chains conserving N_up and N_dn separately: half quantax `ParticleHop`, half `SiteExchange`,
    started from random states of the lattice's particle-number sector. The seed resets quantax's global key."""
    import quantax as qtx
    if qtx.get_sites().Nparticles is None:
        raise ValueError("the u1mix sampler needs a lattice with Nparticles fixed")
    if n_samples % 2:
        raise ValueError(f"u1mix splits its chains in half; {n_samples} is odd")
    qtx.set_random_seed(seed)
    half = n_samples // 2
    return qtx.sampler.MixSampler(
        [qtx.sampler.ParticleHop(state, half, thermal_steps=0),
         qtx.sampler.SiteExchange(state, half, thermal_steps=0)],
        thermal_steps=thermal_steps, sweep_steps=sweep_steps)


def chains_sampler(state, n_samples, *, thermal, sweeps, seed=0, chains_in=None, chains_thermal=-1):
    """`u1_mix_sampler`, burnt in from random states for `thermal` proposals a chain, or started from saved chains
    (`chains_in`: a path or an array, `io.load_chains`) and re-thermalised for `chains_thermal` proposals
    (`-1`: one `sweeps`)."""
    import jax.numpy as jnp
    if chains_in is None or (isinstance(chains_in, str) and not chains_in):
        return u1_mix_sampler(state, n_samples, thermal_steps=thermal, sweep_steps=sweeps, seed=seed)
    from .io import load_chains
    if isinstance(chains_in, (str, bytes)) or hasattr(chains_in, "__fspath__"):
        spins0 = load_chains(chains_in, n_samples)
    else:
        spins0 = np.asarray(chains_in).astype(np.int8)
        if spins0.shape[0] != n_samples:
            raise ValueError(f"{spins0.shape[0]} chains given for {n_samples} samples")
    sampler = u1_mix_sampler(state, n_samples, thermal_steps=0, sweep_steps=sweeps, seed=seed)
    sampler.reset(nsweeps=sweeps if chains_thermal < 0 else chains_thermal, initial_spins=jnp.asarray(spins0))
    return sampler


# ---- SR solve and optimiser ---------------------------------------------------------------------------------------
def _fused_matvec(A, x):
    """`A @ x` as one fused multiply-and-reduce over the columns (no GEMM)."""
    import jax.numpy as jnp
    return jnp.sum(A * x[None, :], axis=1)


def _lsmr_float32(rtol=1e-4, atol=0.0, maxiter=500, fused_matvec=False):
    """quantax's `lsmr` solve for a float32 Jacobian under `jax_enable_x64` (lineax's LSMR with a norm returning
    float64 scalars, so its divisions type-check; no vector or Jacobian is widened)."""
    import jax
    import jax.numpy as jnp
    import lineax as lx
    from jax.lax import with_sharding_constraint
    from lineax._norm import two_norm
    from quantax.utils import get_replicated_sharding

    def norm(v):
        return two_norm(v).astype(jnp.float64)

    @jax.jit
    def solution(A, b, *, diag_preconditioner=None, **kwargs):
        if diag_preconditioner is not None:
            diag_preconditioner = with_sharding_constraint(diag_preconditioner, get_replicated_sharding())
        n, m = A.shape

        def M_apply(x):
            if diag_preconditioner is not None:
                x /= diag_preconditioner
            if fused_matvec:
                return _fused_matvec(A, x)
            return jnp.einsum("sk,k->s", A, x, precision="highest")

        operator = lx.FunctionLinearOperator(M_apply, jax.ShapeDtypeStruct((m,), A.dtype))
        solver = lx.LSMR(rtol=rtol, atol=atol, max_steps=maxiter, norm=norm)
        x = lx.linear_solve(operator, b, solver, options={}, throw=False).value
        if diag_preconditioner is not None:
            x /= diag_preconditioner
        return x

    return solution


def _absolute_lsmr(rtol=1e-4, atol=0.0, maxiter=500):
    """quantax's `lsmr` with absolute tolerances (quantax 0.4 divides them by the sample count)."""
    from quantax.optimizer.solver import lsmr
    solvers = {}

    def solution(A, b, *, diag_preconditioner=None, **kwargs):
        n = A.shape[0]
        if n not in solvers:
            solvers[n] = lsmr(rtol=rtol * n, atol=atol * n, maxiter=maxiter)
        return solvers[n](A, b, diag_preconditioner=diag_preconditioner, **kwargs)

    return solution


#: `minsr_chunked`'s parameters per card per round
MINSR_CHUNK = 4096


def minsr_chunked(rshift=None, ashift=1e-6, chunk=None, mesh=None):
    """quantax's `minnorm_shift_eig` (quantax before 0.4) — `x = A^T (A A^T +
    eps I)^-1 b` with quantax's own shift and Cholesky solve, in float64 —
    without ever holding a second Jacobian.

    WHY. quantax transposes `A`, reshards it over the parameters and converts a
    card's whole share to float64 before its Gram product; at a large network
    and sample count that share fills much of a card, and XLA's GEMM autotuner
    benchmarks the product on fresh buffers that size.
    Chunking only the float64 conversion is not enough: the padded transpose
    and its reshard are themselves card-sized, and the solve's temporaries
    outgrow the card.

    HOW. `A` stays sharded over the samples, as quantax hands it over. The
    Gram matrix is built in rounds: each card cuts `chunk` columns per card
    out of its own rows, one `all_to_all` gives every card all samples of its
    `chunk` columns, and `S S^T` in float64 is added to a per-card sum that one
    `psum` joins — the reshard's traffic, a slab at a time. `x = A^T y` needs
    no reshard at all: each card forms `A_d^T y_d` from its own rows, block by
    block, and a `psum` adds the cards. The last round's window is clamped to
    the end of the parameters and its columns already counted are zeroed.

    The same arithmetic in another summation order, so results move by
    round-off. `diag_preconditioner` is declared, so quantax passes `AdamSR`'s
    `V` here instead of forming `Obar / V`: the columns are divided by `V` in
    every slab, and `A_d^T y_d` by `V^2` — quantax's `solve(A / V, b) / V`. A real
    Jacobian only (the real chart, the package's only one). `mesh` is quantax's
    `("process", "device")` mesh unless a test hands another.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from jax.scipy.linalg import solve
    from quantax.utils import make_mesh

    chunk = MINSR_CHUNK if chunk is None else int(chunk)

    @jax.jit
    def solution(A, b, *, diag_preconditioner=None, **kwargs):
        if jnp.iscomplexobj(A):
            raise ValueError("minsr_chunked takes a real Jacobian")
        input_dtype = A.dtype
        mesh_ = make_mesh() if mesh is None else mesh
        axes = mesh_.axis_names
        ndev = mesh_.size
        n, m = A.shape
        if n % ndev:
            raise ValueError(f"{n} samples do not split over {ndev} cards")
        V = (jnp.ones((m,), A.dtype) if diag_preconditioner is None
             else diag_preconditioner.astype(A.dtype))
        W = chunk * ndev
        if m < W:   # one round, on columns padded with zeros (and V of one)
            A = jnp.pad(A, ((0, 0), (0, W - m)))
            V = jnp.pad(V, (0, W - m), constant_values=1)
        mw = max(m, W)
        nround = -(-mw // W)
        n_l = n // ndev

        def local(Ad, b, V):
            with jax.enable_x64():
                def window(k):
                    start = jnp.minimum(k * W, mw - W)
                    blk = lax.dynamic_slice_in_dim(Ad, start, W, 1)
                    vk = lax.dynamic_slice_in_dim(V, start, W, 0)
                    return blk, vk, start

                def gram(k, T):
                    blk, vk, start = window(k)
                    fresh = (start + jnp.arange(W)) >= k * W
                    blk = jnp.where(fresh[None, :], blk / vk[None, :], 0)
                    S = lax.all_to_all(blk, axes, 1, 0, tiled=True)
                    S = S.astype(jnp.float64)
                    return T + jnp.matmul(S, S.T, precision="highest")

                T = lax.fori_loop(0, nround, gram, jnp.zeros((n, n), jnp.float64))
                T = lax.psum(T, axes)
                T = diag_shift(T, rshift, ashift)
                y = solve(T, b.astype(jnp.float64), assume_a="pos")
                y_l = lax.dynamic_slice_in_dim(y, lax.axis_index(axes) * n_l, n_l)

                def back(k, x):
                    blk, vk, start = window(k)
                    # (A / V)^T y, divided by V once more: A^T y / V^2
                    xk = (jnp.matmul(blk.astype(jnp.float64).T, y_l, precision="highest")
                          / vk.astype(jnp.float64) ** 2)
                    return lax.dynamic_update_slice_in_dim(x, xk, start, 0)

                x = lax.fori_loop(0, nround, back, jnp.zeros((mw,), jnp.float64))
                return lax.psum(x, axes)

        # `check_vma=False`: the loops start from zeros, which the checker
        # types as replicated, and accumulate per-card values into them
        x = jax.shard_map(local, mesh=mesh_, in_specs=(P(axes, None), P(), P()),
                          out_specs=P(), check_vma=False)(A, b, V)
        return x[:m].astype(input_dtype)

    return solution


def diag_shift(T, rshift, ashift):
    """`T + (rshift Tr(T) / sqrt(n) + ashift) I` — quantax's `_diag_shift`
    (quantax before 0.4), removed in quantax 0.4. Its replacement `_shift(A)`
    reads the shift off the whole Jacobian, which `minsr_chunked` never holds;
    with fewer samples than parameters, `Tr(A A^T) = sum |A|^2` and its `n` is
    the sample count, so the shift is the same number."""
    import jax.numpy as jnp
    from quantax.optimizer.solver import _get_rtol
    n = T.shape[0]
    trace = jnp.linalg.trace(T).real
    if rshift is None:
        rshift = _get_rtol(trace.dtype)
    return T + (rshift * trace / jnp.sqrt(n) + ashift) * jnp.identity(n, T.dtype)


def build_solver(fused_matvec=True, solver="lsmr", shift_abs=None, minsr_chunk=None, **kw):
    """The SR solve AdamSR is handed. `lsmr` (the default): at the Jacobian's width — a float32 Jacobian to
    `_lsmr_float32`, a float64 one to quantax's `lsmr` with absolute tolerances; the energy vector and AdamSR's
    preconditioner cast to the Jacobian's dtype. LSMR never forms the S matrix and regularises by stopping early
    (rtol 1e-4, at most 500 iterations). `minsr_chunked`: the MinSR solve
    `A^T (A A^T + eps I)^-1 b` with quantax's shift (`shift_abs`, the relative shift at quantax's default), built a
    parameter chunk at a time."""
    if solver == "minsr_chunked":
        skw = {} if shift_abs is None else {"ashift": shift_abs}
        return minsr_chunked(**skw, chunk=minsr_chunk)
    if solver != "lsmr":
        raise ValueError(f"solver={solver!r}; 'lsmr' or 'minsr_chunked'")
    solve64 = _absolute_lsmr(**kw)
    solve32 = _lsmr_float32(**kw, fused_matvec=fused_matvec)

    def solution(A, b, *, diag_preconditioner=None, **kwargs):
        if diag_preconditioner is not None:
            diag_preconditioner = diag_preconditioner.astype(A.dtype)
        solve = solve32 if A.dtype.itemsize == 4 else solve64
        return solve(A, b.astype(A.dtype), diag_preconditioner=diag_preconditioner, **kwargs)

    return solution


def _fused_adam_updater(mu, beta):
    """quantax's `AdamUpdater` with `Obar @ mhat` as a fused reduction; the rest is quantax's, line for line."""
    import jax.numpy as jnp
    from quantax.optimizer.updater import AdamUpdater

    class FusedAdamUpdater(AdamUpdater):
        def update(self, core_solve, Obar, Ebar, buffers):
            g = core_solve(Obar, Ebar)
            m_new = self.mu * buffers["m"] + (1 - self.mu) * g
            v_new = self.beta * buffers["v"] + (1 - self.beta) * jnp.abs(g) ** 2
            finite = jnp.all(jnp.isfinite(m_new)) & jnp.all(jnp.isfinite(v_new))
            t = buffers["t"] + finite.astype(buffers["t"].dtype)
            m = jnp.where(finite, m_new, buffers["m"])
            v = jnp.where(finite, v_new, buffers["v"])
            buffers["t"] = t
            buffers["m"] = m
            buffers["v"] = v
            mhat = jnp.where(t == 0, m, m / (1 - self.mu**t).astype(m.dtype))
            vhat = jnp.where(t == 0, v, v / (1 - self.beta**t).astype(v.dtype))
            V = vhat**0.25 + 1e-8
            Ebar = Ebar - _fused_matvec(Obar, mhat)
            step = core_solve(Obar, Ebar, diag_preconditioner=V) + mhat
            return step, buffers

    return FusedAdamUpdater(mu, beta)


def build_optimiser(state, H, solver, clip=5.0, fused_matvec=True):
    """quantax's `AdamSR`: SR with first-moment momentum and the diagonal preconditioner `V = vhat**0.25`
    (momentum factors at quantax's defaults; the momentum starts at zero), local energies clipped at `clip` standard
    deviations."""
    from quantax.optimizer import AdamSR
    opt = AdamSR(state, H, solver=solver, clip=clip)
    if fused_matvec:
        opt._updater = _fused_adam_updater(opt._updater.mu, opt._updater.beta)
    return opt


# ---- step control -------------------------------------------------------------------------------------------------
def step_cap(i, c0=0.5, t0=1000, decay=1500.0):
    """The step-norm ceiling at iteration `i`: `c0` until `t0`, then `c0 / (1 + (i - t0) / decay)`."""
    return c0 / (1.0 + max(i - t0, 0) / decay)


def scale_step(raw, i, c0=0.5, t0=1000, decay=1500.0):
    """The update to apply: the raw SR step at learning rate 1, scaled down to the ceiling when its norm exceeds
    it (`min(1, c(i) / |raw|) * raw`); a step already under the ceiling is untouched."""
    import jax.numpy as jnp
    cmax = step_cap(i, c0, t0, decay)
    norm_raw = float(jnp.linalg.norm(raw))
    if not math.isfinite(norm_raw):
        raise FloatingPointError(f"step {i}: the SR step's norm is {norm_raw}; the weights were not updated")
    scale = min(1.0, cmax / norm_raw) if norm_raw > 0 else 1.0
    return scale * raw


# ---- the step loop ------------------------------------------------------------------------------------------------
def run(model, *, n_particles, t=1.0, u=8.0, tprime=0.0, mu=0.0, symm=None, steps=2000, n_samples=8192,
        thermal=4096, sweeps=256, chain_seed=0, chains_in=None, chains_thermal=-1, chains_out=None,
        max_parallel=(2048, 256), cap_c0=0.5, cap_t0=1000, cap_decay=1500.0, cap_step_offset=0,
        lsmr_rtol=1e-4, lsmr_atol=0.0, lsmr_maxiter=500, solver="lsmr", shift_abs=None,
        clip=5.0, fused_matvec=True, eloc_precision=None, eloc_dual=False, on_step=None):
    """Optimise `model` by sampled VMC on its own lattice (`model.Lx x model.Ly`); returns `(rows, model)`.

    Every step: `sweeps` Metropolis proposals a chain for the `n_samples` chains, AdamSR's step from the samples'
    local energies (clipped at `clip` sigma in the gradient only) and log-derivatives (LSMR), the step scaled under
    the ceiling `step_cap(step + cap_step_offset)`, the update applied.

    * `symm` projects the state (`system.build_symm`); `max_parallel` counts configurations, each evaluated at
      every image of the group.
    * `chains_in` starts the chains from saved ones (path or array) re-thermalised for `chains_thermal` proposals
      (`-1`: one `sweeps`) instead of a `thermal` burn-in from random states; `chains_out` writes the chains (`.npy`
      or packed `.npz`) after every step — a continuation passes both, with `cap_step_offset` the steps already
      taken so the cap's decay carries on. AdamSR's momentum is not carried across a continuation (it restarts
      at zero).
    * `eloc_precision` (`"highest"`) evaluates the local energies' amplitudes at full float32 matmul precision while
      the sampler and the Jacobian keep the trunk's TF32 (`at_precision`). `eloc_dual` (with `eloc_precision` set)
      also reads the sampler's own TF32 estimator on the same samples every step and records it beside the
      reweighted one (`e_sampler_precision`, `var_sampler_precision`, `nonfinite_sampler_precision`,
      `rw_ess_frac`); the reweighted energy, force and step are unchanged.
    * `solver`: `"lsmr"` (the default) or `"minsr_chunked"` with `shift_abs`;
      `fused_matvec` makes AdamSR's and LSMR's Jacobian products fused reductions (the default).

    `rows` holds per step the sampled energy, its error bar `sqrt(VarE / n_samples)`, VarE, the applied step's norm
    and the seconds. `on_step(row, model, sampler)` is called after every step."""
    if eloc_dual and not eloc_precision:
        raise ValueError("eloc_dual compares against eloc_precision's re-evaluation; set both")
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.utils import to_replicated_numpy

    from .io import save_chains
    from .state import make_state

    Lx, Ly = model.Lx, model.Ly
    sites = qtx.get_sites()
    if (sites.Nsites, tuple(sites.Nparticles or ())) != (Lx * Ly, tuple(n_particles)):
        raise ValueError(f"quantax's lattice ({sites.Nsites} sites, {sites.Nparticles}) is not the model's "
                         f"{Lx}x{Ly} at {tuple(n_particles)}: call make_sites first")
    n_sites = Lx * Ly
    bonds = square_bonds(Lx, Ly)
    bonds2 = diagonal_bonds(Lx, Ly) if tprime else ()
    dual = {} if eloc_dual else None
    H = at_precision(hubbard_operator(bonds, n_sites, t=t, u=u, mu=mu, bonds2=bonds2, t2=tprime), eloc_precision,
                     dual=dual)
    state = make_state(model, symm=symm, max_parallel=max_parallel)
    # the chains are built last: their seed resets quantax's global key
    sampler = chains_sampler(state, n_samples, thermal=thermal, sweeps=sweeps, seed=chain_seed,
                             chains_in=chains_in, chains_thermal=chains_thermal)
    if solver == "lsmr":
        solver = build_solver(fused_matvec=fused_matvec, rtol=lsmr_rtol, atol=lsmr_atol, maxiter=lsmr_maxiter)
    else:
        solver = build_solver(fused_matvec=fused_matvec, solver=solver, shift_abs=shift_abs)
    opt = build_optimiser(state, H, solver, clip=clip, fused_matvec=fused_matvec)
    rows = []
    for step in range(steps):
        s0 = time.perf_counter()
        samples = sampler.sweep(sweeps)
        raw = opt.get_step(samples)
        if chains_out:
            save_chains(chains_out, sampler._spins)
        update = scale_step(raw, step + cap_step_offset, c0=cap_c0, t0=cap_t0, decay=cap_decay)
        energy, var_e = opt.energy, opt.VarE
        state.update(update)
        var = float(np.real(to_replicated_numpy(var_e)))
        row = {"step": step, "U": u, "energy": float(np.real(to_replicated_numpy(energy))),
               "energy_err": float(np.sqrt(var / n_samples)), "var_e": var,
               "step_norm": float(jnp.linalg.norm(update)), "raw_norm": float(jnp.linalg.norm(raw)),
               "cap": step_cap(step + cap_step_offset, cap_c0, cap_t0, cap_decay),
               "seconds": round(time.perf_counter() - s0, 4)}
        if dual is not None:
            row.update(e_sampler_precision=float(np.real(to_replicated_numpy(dual["e"]))),
                       var_sampler_precision=float(np.real(to_replicated_numpy(dual["var"]))),
                       nonfinite_sampler_precision=int(to_replicated_numpy(dual["nonfinite"])),
                       rw_ess_frac=float(np.real(to_replicated_numpy(dual["ess"]))) if "ess" in dual else None)
        rows.append(row)
        if on_step is not None:
            on_step(row, state.model, sampler)
    return rows, state.model
