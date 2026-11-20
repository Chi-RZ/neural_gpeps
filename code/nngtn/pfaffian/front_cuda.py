"""The CUDA elimination of one tree level's fronts (`front_cuda.cu`) as a jax function.

`fused_fn(...)` gives a level's function: the fronts assemble themselves in the kernel from the free vector and the
child levels' contribution blocks, are eliminated, and come back as the contribution blocks for the parent, the
pivots and log|psi|'s share. Batched by `vmap` (`expand_dims`); its `custom_vjp` runs the backward kernel on the
multipliers the forward recorded, from the adjoints of `C` and `logabs`. Built with nvcc on first use and cached under
`$XDG_CACHE_HOME/nngtn_kernels` (default `~/.cache/nngtn_kernels`), keyed by the source, the compile flags, the
compiler's path and the jax version.

Two libraries, one scalar each: `double` (the default) and `float` (`-DPIVOTED_REAL=float`). `fused_fn` takes
`dtype` and runs the library of that scalar — every front, contribution block, multiplier and
record in it, log|psi| and sign too (accumulated in double inside the kernels, stored in the scalar). The float
library closes its global-memory panels on the CUDA cores, or on the fp64 tensor cores when built with
`PIVOTED_GCLOSE_DMMA` (`dmma=True`, see `recipe.py`).
"""
from __future__ import annotations

import ctypes
import functools
import fcntl
import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np

# ---- the build lock ------------------------------------------------------------------------------------------------------
# Compile a CUDA library once across every process asking for it (a file lock, a temporary file renamed into
# place).

def compile_locked(lib, command, needed=None, *, prepare=None, run=None):
    """Compile `lib` once across every process asking for it: `command` is the
    compiler's argument list WITHOUT its output (`-o` goes in after the compiler's
    name, naming a temporary file beside `lib` that is renamed into place when
    the compiler exits 0, so no reader ever sees a partial library; the source
    stays the last argument); `needed()` says whether
    a build is due (default: `lib` missing) and is asked again under the lock,
    so a process that waited while another built finds nothing left to do;
    `prepare()` runs under the lock before the compiler (a generated header).
    The lock is `flock` on `<lib>.lock`; a filesystem that refuses it still gets
    the temporary-and-rename, which alone leaves only the double compile."""
    lib = Path(lib)
    needed = needed or (lambda: not lib.exists())
    if not needed():
        return lib
    run = run or subprocess.run
    lib.parent.mkdir(parents=True, exist_ok=True)
    with open(lib.with_name(lib.name + ".lock"), "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            locked = True
        except OSError:
            locked = False
        try:
            if locked and not needed():
                return lib
            if prepare is not None:
                prepare()
            tmp = lib.with_name(f"{lib.name}.{os.getpid()}.tmp")
            try:
                run([command[0], "-o", str(tmp), *command[1:]], check=True)
                os.replace(tmp, lib)
            finally:
                tmp.unlink(missing_ok=True)
        finally:
            if locked:
                fcntl.flock(fh, fcntl.LOCK_UN)
    return lib


_SOURCE = Path(__file__).with_suffix(".cu")


REALS = ("double", "float")


def _real(dtype):
    """The library scalar of a jax dtype: float32 -> "float", float64 -> "double"."""
    return "float" if np.dtype(dtype) == np.float32 else "double"


def library_path(real="double"):
    import jax
    from jax import ffi
    if real not in REALS:
        raise ValueError(f"real={real!r}; expected one of {REALS}")
    # the caller's own NNGTN_NVCC_FLAGS, else this library's recipe flags (recipe.apply)
    flags = os.environ.get("NNGTN_NVCC_FLAGS", os.environ.get(f"NNGTN_NVCC_FLAGS_{real.upper()}", ""))
    defs = ["-DPIVOTED_REAL=float", "-DPIVOTED_REAL_FLOAT"] if real == "float" else []
    sources = _SOURCE.read_bytes()
    nvcc = os.environ.get("NNGTN_NVCC", "nvcc")
    key = hashlib.sha256(sources+(flags+"\n"+" ".join(defs)+"\n"+str(ffi.include_dir())+"\n"+jax.__version__
                                  + "\n"+nvcc).encode())
    checkout = hashlib.sha256(str(_SOURCE.resolve()).encode()).hexdigest()[:12]
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache")))
    lib = root/"nngtn_kernels"/checkout/key.hexdigest()[:16]/f"front_{real}.so"
    # one process of a job compiles, the rest wait and load
    return compile_locked(lib, [nvcc, "-O3", "-std=c++17", "-arch=sm_80", "-shared",
                                "-Xcompiler", "-fPIC", f"-I{ffi.include_dir()}", *defs, *shlex.split(flags), str(_SOURCE)])


@functools.lru_cache(None)
def _load(real="double"):
    import jax
    from jax import ffi
    from jax._src.lib import xla_client
    path = library_path(real)
    handle = ctypes.cdll.LoadLibrary(str(path))
    tag = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    names = ("pivoted_front_fused_"+tag, "pivoted_front_global_"+tag, "pivoted_parity_"+tag,
             "pivoted_front_bwds_"+tag)
    for name, symbol in zip(names, (handle.FrontFused, handle.FrontGlobal, handle.FrontParity,
                                    handle.FrontBackwardS)):
        ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA",
                                traits=xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE)
    jax.devices()       # a context exists before the attribute is set
    code = handle.front_cuda_init()
    if code != 0:   # 1000 * (instance + 1) + cudaError_t, forward; 10000 * ..., backward
        raise RuntimeError(f"front_cuda_init could not raise the shared-memory limit (code {code})")
    return names


@functools.lru_cache(None)
def parity_fn(real="double"):
    """`parity(alive (n_modes,) int8, *pivots) -> () int32`: the sign's permutation parity of contract._parity_modes
    — the alive modes in increasing id sent, in order, to the valid entries (mode ids, -1 absent) of the levels'
    pivot arrays taken in order — as one kernel a configuration, no concatenation. Batches by vmap."""
    import jax
    from jax import ffi
    name = _load(real)[2]
    call = ffi.ffi_call(name, jax.ShapeDtypeStruct((), jax.numpy.int32), vmap_method="broadcast_all")
    return lambda alive, *pivots: call(alive, *pivots)


@functools.lru_cache(None)
def parity_fits(real, n_modes):
    """Whether the parity kernel's four arrays a mode for `n_modes` modes fit the shared memory this device grants it
    (cards with 99 KB a block do not take 32x32 lattices); `contract` uses its _parity_modes otherwise."""
    _load(real)
    return bool(ctypes.cdll.LoadLibrary(str(library_path(real))).front_cuda_parity_fits(int(n_modes)))


@functools.lru_cache(None)
def max_width(real="double"):
    """The widest front the shared-memory kernels of the `real` library take on this device; `contract` sends wider
    ones to the global-memory kernels (`front_gfwd_kernel`, `front_gbwd_kernel`)."""
    _load(real)
    return int(ctypes.cdll.LoadLibrary(str(library_path(real))).front_cuda_max_width())


def fused_fn(key, tables, n, steps, u, root, d_max, u_max, fs, child_shape, n_phys, inv_pos, inv_sgn, panel=1,
             glob=False, search_window=0, warp_max=96, dtype=None, gbwd=False, gpanel=1, bwduu=False):
    """A level whose fronts assemble themselves in the kernel (`front_fwd_fused` a warp a front up to 96 slots, the
    block kernel's fused form above; `panel`: pivot pairs whose updates the block kernel holds back and applies as
    one, 1 = each at once): `(free_flat, childC, childV, childM, occ, acc) -> (C, cvalid, cmode, count, alive, logabs,
    sign, pivots, growth, acc)`: per front the contribution block for the parent, `(d_max + u_max)^2` — the leftover
    candidates (at most `slots_here`, in slot order, padded to `d_max`) then the update slots (padded to `u_max`) —
    with its rows' validity and mode ids, the number of leftover candidates (more than `slots_here` is an overflow),
    the alive flags after, the log of the product of pivots and its sign, the pivot pairs' mode ids `(steps, 2)` (-1
    where none), the largest entry the scan met; then `acc` unchanged. `acc`: the free vector's gradient accumulator
    threaded through the levels — the backward writes this level's free-source gradients into its cotangent's own
    buffer (the FFI output aliased to it) and returns that as `acc`'s cotangent, None for `free_flat`'s. `tables`: the
    numpy arrays pc_off (nf + 1,), pc_desc (P, 4), pc_map (M,) — the assembly pieces, `contract._pieces` — bd_off (nf +
    1,), bd_pq (NB, 2), bd_val (NB,) — the bonds' constants — asrc (nf, n), msrc (nf, n), counts (nf, 3), ijtab, ijoff
    (the packed (i, j) per width). The adjoint runs the backward kernel, which writes the sources' gradients on
    `free_flat` and `childC` itself through a per-front table of `inv_pos` / `inv_sgn` — the one front entry each
    source feeds (a flat position in nf * n * n, nf * n * n = none) over [0, free..., child...] (FrontBackwardS).
    `key` is a hashable identity of the level for the cache. `glob`: the global-memory kernels, the fronts in a
    workspace slab of n x n each, for fronts wider than `max_width()` (or any, when asked). `search_window`: candidate
    columns per round (0 preserves the original block-width window); fewer columns use the same threshold,
    continuing to later candidates when none passes. `warp_max`: the widest front the one-warp kernel takes; wider
    ones (up to the shared-memory bound) go to the block kernel."""
    import jax.numpy as jnp
    rt = jnp.float64 if dtype is None else jnp.dtype(dtype)
    return _fused_cached(key, n, steps, u, root, d_max, u_max, fs, child_shape, n_phys, panel, glob, search_window,
                         warp_max, _real(rt), bool(gbwd), int(gpanel), int(bwduu))(tables, inv_pos, inv_sgn)


@functools.lru_cache(None)
def _fused_cached(key, n, steps, u, root, d_max, u_max, fs, child_shape, n_phys, panel, glob, search_window,
                  warp_max, real, gbwd=False, gpanel=1, bwduu=False):
    # bwduu: the shared backward's triangle holds the rows
    # below the update boundary only (`buuent`), the update rows' products over the update
    # columns coming from bwd_uu_kernel (W's update block is Cbar's for the whole replay); shared levels only
    # the block kernel's shared triangle holds the rows below the update boundary only; `uuent` is
    # the level's largest such prefix (the widest front, every slot alive), 0 when a front has more than three
    # children (the gather's inverse maps) or the level runs a panel
    # gbwd: the backward on the global-memory kernel (`front_gbwd_kernel`, its replay panel `gpanel`) from the
    # shared-memory forward's records — the two forwards record the same slot-indexed multipliers
    def build(tables, inv_pos, inv_sgn):
        import jax
        import jax.numpy as jnp
        from jax import ffi
        names = _load(real)
        rt = jnp.float32 if real == "float" else jnp.float64
        fused_name = names[1] if glob else names[0]
        pc_off, pc_desc, pc_map, bd_off, bd_pq, bd_val, asrc, msrc, counts, ijtab, ijoff = tables
        nfront = len(pc_off)-1
        cd = d_max+u_max
        cs = sum(nfc*cdc*cdc for nfc, cdc in child_shape)      # the source levels' blocks, concatenated flat
        vs = sum(nfc*cdc for nfc, cdc in child_shape)
        attrs = dict(nfront=int(nfront), n=int(n), fs=int(fs), cs=int(cs), vs=int(vs),
                     np=int(n_phys), steps=int(steps), u=float(u), root=int(root), d_max=int(d_max), u_max=int(u_max),
                     panel=int(panel), search_window=int(search_window))
        if not glob:
            attrs["warp_max"] = int(warp_max)     # fronts wider than this take the block kernel (the warp kernel's own bound is 96)
            uuent = 0
            if panel == 1:
                nu = np.asarray(counts)[:, 2]
                nchild = np.array([int(np.sum(np.asarray(pc_desc)[pc_off[i]:pc_off[i+1], 3] < 0)) for i in range(nfront)])
                if nfront and nchild.max() <= 3:
                    nfc = int(n)-nu
                    uuent = int(np.max(nfc*(2*int(n)-nfc-1)//2))
            attrs["uuent"] = int(uuent)
        f64, s8, s32 = rt, jnp.int8, jnp.int32   # f64: the library's scalar
        out = (jax.ShapeDtypeStruct((nfront, cd, cd), f64), jax.ShapeDtypeStruct((nfront, cd), s8),
               jax.ShapeDtypeStruct((nfront, cd), s32), jax.ShapeDtypeStruct((nfront,), s32),
               jax.ShapeDtypeStruct((nfront, n), s8), jax.ShapeDtypeStruct((nfront,), f64),
               jax.ShapeDtypeStruct((nfront,), f64), jax.ShapeDtypeStruct((nfront, steps, 2), s32),
               jax.ShapeDtypeStruct((nfront,), f64))
        rec = (jax.ShapeDtypeStruct((nfront, steps, 2), s32), jax.ShapeDtypeStruct((nfront, steps), f64),
               jax.ShapeDtypeStruct((nfront, steps, n), f64), jax.ShapeDtypeStruct((nfront, steps, n), f64))
        small = (jax.ShapeDtypeStruct((nfront, 1, 2), s32), jax.ShapeDtypeStruct((nfront, 1), f64),
                 jax.ShapeDtypeStruct((nfront, 1, 1), f64), jax.ShapeDtypeStruct((nfront, 1, 1), f64))
        ws = (jax.ShapeDtypeStruct((nfront, n, n), f64),) if glob else ()     # the global kernel's slabs
        scr = (jax.ShapeDtypeStruct((nfront, steps, max(u_max, 1)), jnp.float64),)*2    # update rows' c, tau a step: double in both libraries, the tensor-core writer's operands
        forward_only = ffi.ffi_call(fused_name, out+small+scr+ws, vmap_method="expand_dims")
        forward_rec = ffi.ffi_call(fused_name, out+rec+scr+ws, vmap_method="expand_dims")
        ypre_shape = jax.ShapeDtypeStruct((nfront, steps, 2, max(u_max, 1)), f64)
        buuent = 0
        # bwduu 2: the block kernel's levels only (n > warp_max)
        if bwduu and not glob and not gbwd and 0 < u_max <= 128 and nfront and (bwduu == 1 or n > warp_max):
            nu = np.asarray(counts)[:, 2]
            nfc = int(n)-nu
            buuent = int(np.max(nfc*(2*int(n)-nfc-1)//2))
        ip = np.asarray(inv_pos); sgn = np.asarray(inv_sgn); NN = n*n
        # no child memset where the level's pieces cover its child buffer (every entry the child's backward reads — a
        # valid-row pair's upper entry — is one the table writes), the backward reading the cotangent's upper entry
        # alone; an unbalanced tree's level reads only some of a source level's blocks, the rest zero for another
        # level to fill, and then the memset stays
        cfull = 0
        if cs:
            cip = ip[1+fs:1+fs+cs]
            covered = np.zeros(cs, bool)
            pd_ = np.asarray(pc_desc); pof_ = np.asarray(pc_off)
            for f_ in range(nfront):
                for q_ in range(pof_[f_], pof_[f_+1]):
                    if pd_[q_, 3] < 0:
                        b_, w_ = int(pd_[q_, 0]), int(pd_[q_, 1])
                        covered[b_:b_+w_*w_] = True
                        seg = cip[b_:b_+w_*w_]   # a mapped child entry must belong to the front that zeroes the rest
                        assert np.all((seg >= nfront*NN) | (seg // NN == f_)), "child piece mapped to another front"
            cfull = int(bool(covered.all()))
        nomemset = cfull
        # the per-front table: every source
        idx = np.nonzero(ip < nfront*NN)[0]
        fr = ip[idx]//NN
        order = np.argsort(fr, kind="stable")
        csr_s = idx[order].astype(np.int32)
        csr_loc = (ip[idx] % NN)[order].astype(np.int32)
        csr_sg = sgn[idx][order]
        csr_off = np.searchsorted(fr[order], np.arange(nfront+1)).astype(np.int32)
        bs_out = (jax.ShapeDtypeStruct((nfront, n, n) if (glob or gbwd) else (1,), f64),
                  jax.ShapeDtypeStruct((fs,), f64), jax.ShapeDtypeStruct((max(cs, 1),), f64), ypre_shape)
        backward = ffi.ffi_call(names[3], bs_out, vmap_method="expand_dims", input_output_aliases={12: 1})

        def statics():   # the bonds' constants in the library's scalar, the index tables as they are
            return tuple(jnp.asarray(x, rt) if i == 5 else jnp.asarray(x) for i, x in enumerate(tables))

        # acc: the free vector's gradient accumulator threaded through the levels — the primal passes it through
        # unchanged; the backward writes this level's free-source gradients into the accumulator cotangent's own
        # buffer (the FFI output aliased to it) and returns that as the accumulator's cotangent, None for free_flat's
        @jax.custom_vjp
        def level(free_flat, childC, childV, childM, occ, acc):
            return forward_only(free_flat, childC, childV, childM, occ, *statics(), record=0, **attrs)[:9]+(acc,)

        def fwd(free_flat, childC, childV, childM, occ, acc):
            res = forward_rec(free_flat, childC, childV, childM, occ, *statics(), record=1, **attrs)
            return res[:9]+(acc,), (res[4],)+tuple(res[9:13])

        def bwd(saved, cot):
            alive_out, kr, a, c, tau = saved
            _, dfree, dchild, _ = backward(alive_out, kr, a, c, tau, cot[0], cot[5], jnp.asarray(counts),
                                           jnp.asarray(csr_off), jnp.asarray(csr_s), jnp.asarray(csr_loc),
                                           jnp.asarray(csr_sg, f64), cot[9], steps=int(steps), n=int(n),
                                           d_max=int(d_max), u_max=int(u_max), panel=int(gpanel),
                                           glob=int(glob or gbwd), fs=int(fs), cs=int(cs), uuent=buuent,
                                           nomemset=nomemset)
            dchild = dchild if cs else jnp.zeros(1, f64)
            return None, dchild, None, None, None, dfree

        level.defvjp(fwd, bwd)
        return level
    return build
