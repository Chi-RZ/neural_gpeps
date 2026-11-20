"""The neural Gaussian tensor-network state as a quantax model.

    psi(s) = epsilon(n) 2^(-b/2) Pf(M[free(s)])

`free(s)` is the network's output: a `ResConvTrunk` and a 1x1 head emit, for every site, the strict upper triangle
of a real skew pairing matrix `Z_s` (`m(m-1)/2` numbers, `m = 2 + 4 Nv` modes a site). `M` is the sparse skew matrix
of those site entries and the bond constants over the virtual modes and the occupied physical modes; its Pfaffian
is taken by the multifrontal threshold-pivoted elimination with CUDA kernels (`pfaffian/`), forward and backward.

`__call__` is written for ONE configuration (quantax vmaps it): `s` is `+-1` over `[up sites..., down sites...]`;
it returns quantax's `LogArray(sign, log|psi|)`, the sign including the reordering from quantax's spin-block mode
order to the site order the contraction uses.

The symmetry projection is not part of the model: quantax's `Variational` applies it (`make_state(..., symm=...)`).
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import recipe
from .network import ResConvTrunk, scale_residual_branches
from .pfaffian.layout import peps_layout

__all__ = ["NETWORKS", "NeuralGTN", "build_model", "make_state", "load_weights", "save_weights",
           "config_of_spins", "polar_pair"]

#: two network presets by Nv: residual blocks, channels, the padded width the convolutions run at, and the scale of
#: every residual branch's last convolution at construction (the head is a 1x1 convolution with bias, no per-site
#: offset)
NETWORKS = {
    3: dict(nblocks=16, channels=91, pad_channels=92, res_scale=0.1),
    4: dict(nblocks=24, channels=153, pad_channels=160, res_scale=0.02),
}

_PLANS = {}


def _contraction(Lx, Ly, nv, d, threshold, delay_slots, leaf_sites):
    """`fast.log_amplitude` of the Lx x Ly torus: built once per setting and process."""
    key = (Lx, Ly, nv, d, float(threshold), int(delay_slots), int(leaf_sites))
    if key in _PLANS:
        return _PLANS[key]
    from .pfaffian import fast, layout as lay, plan as pl, tree as tr
    _raise_shared_memory_on_every_device()
    network = lay.network_of_layout(peps_layout(Lx, Ly, d, nv, nv, True))
    roots = tr.lattice_dissection(network, Lx, Ly, leaf_sites)
    plan = pl.build(network, roots, int(delay_slots))
    # the first call comes inside a trace; the executor's constant tables must not become its tracers
    with jax.ensure_compile_time_eval():
        _PLANS[key] = fast.log_amplitude(plan, float(threshold))
    return _PLANS[key]


def _raise_shared_memory_on_every_device():
    """`front_cuda_init` raises the kernels' dynamic shared-memory limit on the CURRENT device only, and the
    attribute is per device; this makes each local card current in turn and calls the init there, for both
    libraries (float and double)."""
    import ctypes
    from .pfaffian import front_cuda
    libs = []
    for real in front_cuda.REALS:
        front_cuda._load(real)
        libs.append((real, ctypes.cdll.LoadLibrary(str(front_cuda.library_path(real)))))
    cu = ctypes.CDLL("libcuda.so.1")
    prev = ctypes.c_void_p()
    if cu.cuCtxGetCurrent(ctypes.byref(prev)):
        raise RuntimeError("cuCtxGetCurrent failed")
    try:
        for d in jax.local_devices(backend="gpu"):
            dev, ctx = ctypes.c_int(), ctypes.c_void_p()
            if cu.cuDeviceGet(ctypes.byref(dev), d.local_hardware_id) or \
                    cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) or cu.cuCtxSetCurrent(ctx):
                raise RuntimeError(f"could not make {d} current")
            codes = [(real, lib.front_cuda_init()) for real, lib in libs]
            cu.cuDevicePrimaryCtxRelease(dev)
            for real, code in codes:
                if code:
                    raise RuntimeError(f"front_cuda_init ({real}) on {d} could not raise the shared-memory limit "
                                       f"(code {code})")
    finally:
        cu.cuCtxSetCurrent(prev)


def config_of_spins(s, n_sites, d=2):
    """quantax's `+-1` spins over `[up..., dn...]` to the per-site index `2 up_i + dn_i`."""
    occ = (jnp.asarray(s) > 0).astype(jnp.int32)
    blocks = [occ[j * n_sites:(j + 1) * n_sites] for j in range(d)]
    return sum(b * (2 ** (d - 1 - j)) for j, b in enumerate(blocks))


def polar_pair(log_psi, s):
    """The complex logarithm as quantax's `LogArray(sign, log|psi|)`: a vanishing amplitude (`-inf`) has sign 0,
    and the sign carries the parity between quantax's spin-block mode order and the site order."""
    from quantax.nn import fermion_reorder_sign
    from quantax.utils import LogArray
    logabs = jnp.real(log_psi)
    live = jnp.isfinite(logabs)
    phase = jnp.where(jnp.cos(jnp.imag(log_psi)) >= 0.0, 1.0, -1.0)
    sign = jnp.where(live, phase, 0.0).astype(jnp.float64)
    return LogArray(sign * fermion_reorder_sign(s), logabs)


class NeuralGTN(eqx.Module):
    """`s -> network -> free vectors -> pivoted Pfaffian -> (sign, log|psi|)` on the periodic Lx x Ly torus.

    Leaves: `net` (the trunk) and `head` (a 1x1 convolution, channels -> m(m-1)/2), float32. Static: the lattice and
    the contraction's settings — `contraction_dtype` (float32 or float64), the pivot `threshold`, the `delay_slots`
    a child front may pass up, the dissection's `leaf_sites`."""

    net: eqx.Module
    head: eqx.Module
    Lx: int = eqx.field(static=True)
    Ly: int = eqx.field(static=True)
    nv: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    n_sites: int = eqx.field(static=True)
    emit_len: int = eqx.field(static=True)
    contraction_dtype: str = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    delay_slots: int = eqx.field(static=True)
    leaf_sites: int = eqx.field(static=True)
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, net, head, Lx, Ly=None, nv=3, d=2, contraction_dtype="float32", threshold=0.05,
                 delay_slots=20, leaf_sites=2):
        if contraction_dtype not in ("float32", "float64"):
            raise ValueError(f"contraction_dtype={contraction_dtype!r}; float32 or float64")
        Ly = Lx if Ly is None else Ly
        self.net, self.head = net, head
        self.Lx, self.Ly, self.nv, self.d = int(Lx), int(Ly), int(nv), int(d)
        self.n_sites = self.Lx * self.Ly
        m = d + 4 * nv
        self.emit_len = m * (m - 1) // 2
        self.contraction_dtype = contraction_dtype
        self.threshold, self.delay_slots, self.leaf_sites = float(threshold), int(delay_slots), int(leaf_sites)
        self.holomorphic = False

    def with_contraction(self, **kw):
        """The same weights under other contraction settings (`contraction_dtype`, `threshold`, `delay_slots`,
        `leaf_sites`) — e.g. more delay slots for a measurement."""
        args = dict(contraction_dtype=self.contraction_dtype, threshold=self.threshold,
                    delay_slots=self.delay_slots, leaf_sites=self.leaf_sites)
        unknown = set(kw) - set(args)
        if unknown:
            raise ValueError(f"unknown contraction settings {sorted(unknown)}")
        args.update(kw)
        return NeuralGTN(self.net, self.head, self.Lx, self.Ly, nv=self.nv, d=self.d, **args)

    def free_vectors(self, s):
        """`(n_sites, m(m-1)/2)`, row `i * Ly + j` the free vector of site `(i, j)`."""
        emitted = self.head(self.net(s))
        return emitted.reshape(self.emit_len, self.n_sites).T

    def log_psi(self, config, free):
        """The complex `log psi` of one configuration (per-site index) from its free vectors."""
        f = _contraction(self.Lx, self.Ly, self.nv, self.d, self.threshold, self.delay_slots, self.leaf_sites)
        d = self.d
        bits = jnp.stack([(config >> (d - 1 - j)) & 1 for j in range(d)], axis=-1).reshape(-1)
        dtype = jnp.float32 if self.contraction_dtype == "float32" else jnp.float64
        sign, logabs, stats = f(free.astype(dtype), bits.astype(jnp.int32))
        if dtype == jnp.float64:
            sign, logabs = sign.astype(jnp.float64), logabs.astype(jnp.float64)
        # an overflowed front loses its delayed pivots and the root is left incomplete (sign 0): the overflow is
        # decided last, so such a configuration reads NaN, not a true zero. The sampler rejects it; the measurement
        # counts it as zero amplitude in a connected term (psi-tilde) and drops a sample whose own amplitude
        # overflows; training drops any sample with a non-finite local energy
        logabs = jnp.where(sign == 0, -jnp.inf, logabs)
        logabs = jnp.where(stats["overflow"], jnp.nan, logabs)
        return logabs + 1j * np.pi * (sign < 0)

    def __call__(self, s):
        config = config_of_spins(s, self.n_sites, self.d)
        free = self.free_vectors(s).astype(jnp.float64)
        return polar_pair(self.log_psi(config, free), s)


def build_model(Lx, Ly=None, nv=3, nblocks=None, channels=None, pad_channels=None, kernel=(3, 3), res_scale=None,
                head_scale=1.0, net_seed=0, dtype=jnp.float32, contraction_dtype="float32", threshold=0.05,
                delay_slots=20, leaf_sites=2, card=None, dmma=False):
    """The model, its weights drawn as the original runs draw them: quantax's key reset to `net_seed`, the trunk
    (residual branches scaled by `res_scale`), then the head (lecun-normal weight times `head_scale`, zero bias).

    The network's size defaults to the preset for `nv` (`NETWORKS[nv]`); any of `nblocks`, `channels`,
    `pad_channels` (0 for none), `res_scale` overrides it. `card` (`"hopper"`, `"a100"`, `None` = the first device's)
    picks the kernel recipe (`recipe.apply`); `dmma=True` takes its float32 variant with the global close on the fp64
    tensor cores (off by default; see `recipe`). Needs quantax's lattice (`lattice.make_sites`) set first."""
    import quantax as qtx
    from quantax import get_subkeys
    from quantax.nn import Conv
    Ly = Lx if Ly is None else Ly
    size = dict(NETWORKS.get(nv, {}))
    for k, v in (("nblocks", nblocks), ("channels", channels), ("pad_channels", pad_channels),
                 ("res_scale", res_scale)):
        if v is not None:
            size[k] = v
    missing = {"nblocks", "channels", "res_scale"} - set(size)
    if missing:
        raise ValueError(f"no preset network for nv={nv}; give {sorted(missing)}")
    pad = size.get("pad_channels") or None
    recipe.apply(contraction_dtype, card, dmma)
    qtx.set_random_seed(net_seed)
    net = scale_residual_branches(ResConvTrunk(size["nblocks"], size["channels"], kernel, sublattice=None,
                                               dtype=dtype, pad_channels=pad), size["res_scale"])
    m = 2 + 4 * nv
    head = Conv(in_channels=net.channels, out_channels=m * (m - 1) // 2, kernel_size=1, use_bias=True, dtype=dtype,
                key=get_subkeys())
    head = eqx.tree_at(lambda h: h.weight, head, jnp.asarray(head_scale, dtype) * head.weight)
    head = eqx.tree_at(lambda h: h.bias, head, jnp.zeros_like(head.bias))
    return NeuralGTN(net, head, Lx, Ly, nv=nv, contraction_dtype=contraction_dtype, threshold=threshold,
                     delay_slots=delay_slots, leaf_sites=leaf_sites)


def make_state(model, symm=None, max_parallel=(2048, 256)):
    """quantax's `Variational` state of `model`, projected onto `symm` (`symmetry.build_symm` names, e.g.
    `"flip0@flip1@spininverse"`; `None`: no projection). `max_parallel` is the forward (and backward) chunk per
    device, counted in configurations BEFORE the symmetry's images: quantax expands each configuration of a chunk
    into its images, so a projected state evaluates the network on `max_parallel x |group|` inputs a chunk."""
    import quantax as qtx
    from .symmetry import build_symm
    if isinstance(max_parallel, (list, tuple)) and len(max_parallel) == 1:
        max_parallel = max_parallel[0]
    elif isinstance(max_parallel, list):
        max_parallel = tuple(max_parallel)
    return qtx.state.Variational(model, symm=build_symm(symm), max_parallel=max_parallel)


def load_weights(model, path):
    """Weights saved by `save_weights` (equinox's serialisation) into a model of the same shape (the trunk's
    leaves, then the head's); a leaf of another shape raises."""
    def spec(f, x):
        if not isinstance(x, (jax.Array, np.ndarray)):
            return eqx.default_deserialise_filter_spec(f, x)
        arr = np.load(f)
        if arr.shape != x.shape:
            raise RuntimeError(f"leaf of shape {x.shape} has shape {arr.shape} on disk in {path}")
        return jnp.asarray(arr, dtype=x.dtype)

    return eqx.tree_deserialise_leaves(path, model, filter_spec=spec)


def save_weights(model, path):
    """The model's leaves to `path` (equinox's serialisation), through a temporary file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    eqx.tree_serialise_leaves(tmp, model)
    os.replace(tmp, path)
