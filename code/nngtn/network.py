"""The configuration-reading body: quantax's `ResConv` trunk with its scalar tail dropped.

`s -> (channels, Lx, Ly)` features, one column per site; the model's 1x1 head turns them into one free vector per
site. Translation covariant (circular convolutions, one shared embedding).
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

__all__ = ["ResConvTrunk", "scale_residual_branches"]


class ResConvTrunk(eqx.Module):
    """quantax's `ResConv` with its scalar tail dropped: `s -> (channels, Lx, Ly)` features, one column per site,
    divided by `sqrt(nblocks + 1)`. Each block is RMSNorm -> gelu -> conv -> gelu -> conv, plus the residual.

    `pad_channels` runs the convolutions at a wider channel count than the parameters hold (zero-padded weights
    and feature map, the extra channels sliced off at the end; the RMSNorm divides by the real element count), so
    cuDNN's TF32 kernels see a friendlier shape. The parameters and the output are those of the `channels`-wide
    trunk."""

    nblocks: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    pad_channels: int | None = eqx.field(static=True)
    layers: Tuple[Callable, ...]

    def __init__(self, nblocks: int, channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 sublattice: Sequence[int] | None = None,
                 dtype: DTypeLike = jnp.float64,
                 pad_channels: int | None = None):
        from quantax.model import ResConv
        if pad_channels is not None and int(pad_channels) < int(channels):
            raise ValueError(f"pad_channels={pad_channels} is narrower than "
                             f"the trunk's {channels} channels")
        resconv = ResConv(nblocks, channels, kernel_size, sublattice=sublattice,
                          use_rmsnorm=True, dtype=dtype)
        self.layers = tuple(resconv.layers[:-2])
        self.nblocks = int(nblocks)
        self.channels = int(channels)
        self.pad_channels = None if pad_channels is None else int(pad_channels)

    def __call__(self, s: jax.Array) -> jax.Array:
        if self.pad_channels is not None:
            return self._padded(s)
        x = s
        for layer in self.layers:
            x = layer(x)
        return x / jnp.sqrt(self.nblocks + 1)

    def _padded(self, s: jax.Array) -> jax.Array:
        """The same trunk with every convolution run at `pad_channels`."""
        c, p = self.channels, self.pad_channels
        embedding, to_nbr, *blocks, to_orig = self.layers
        x = to_nbr(embedding(s))
        x = jnp.pad(x, ((0, p - c),) + ((0, 0),) * (x.ndim - 1))
        n_real = c * (x.size // p)
        for block in blocks:
            x = _padded_block(block, x, c, p, n_real)
        x = to_orig(x[:c])
        return x / jnp.sqrt(self.nblocks + 1)


def _padded_conv(conv, x: jax.Array, c: int, p: int) -> jax.Array:
    """`conv` (c -> c channels) applied to a `p`-channel map whose channels
    beyond `c` are zero, its weight and bias zero-padded to `p`."""
    w = conv.weight
    w = jnp.pad(w, ((0, p - c), (0, p - c)) + ((0, 0),) * (w.ndim - 2))
    if conv.bias is None:
        wide = eqx.tree_at(lambda m: m.weight, conv, w)
    else:
        b = jnp.pad(conv.bias, ((0, p - c),) + ((0, 0),) * (conv.bias.ndim - 1))
        wide = eqx.tree_at(lambda m: (m.weight, m.bias), conv, (w, b))
    return wide(x)


def _padded_block(block, x: jax.Array, c: int, p: int, n_real: int) -> jax.Array:
    """quantax's `_ConvBlock` (RMSNorm, gelu, conv1, gelu, conv2, residual) on
    the padded map. The norm is equinox's `RMSNorm` written out — the same
    float32-or-wider compute dtype and `eps` — with the mean of squares taken
    over the `n_real` real elements, the padded zeros adding nothing to the sum."""
    norm = block.norm
    if norm.use_weight or norm.use_bias:
        raise NotImplementedError("the padded block writes out a weightless, "
                                  "biasless RMSNorm, as ResConv builds it")
    residual = x
    orig = x.dtype
    with jax.numpy_dtype_promotion("standard"):
        dt = jnp.result_type(x.dtype, jnp.float32)
    h = x.astype(dt)
    h = (jax.lax.rsqrt(jnp.sum(h ** 2) / n_real + norm.eps) * h).astype(orig)
    h = jax.nn.gelu(h)
    h = _padded_conv(block.conv1, h, c, p)
    h = jax.nn.gelu(h)
    h = _padded_conv(block.conv2, h, c, p)
    return h + residual


def scale_residual_branches(net, scale: float):
    """`net` with the last convolution of every residual block (weight and bias) multiplied by `scale`: an
    initialisation move that makes the trunk start close to its embedding alone."""
    if scale == 1.0:
        return net

    def scaled(conv):
        conv = eqx.tree_at(lambda c: c.weight, conv, scale * conv.weight)
        if getattr(conv, "bias", None) is not None:
            conv = eqx.tree_at(lambda c: c.bias, conv, scale * conv.bias)
        return conv

    if isinstance(net, ResConvTrunk):
        layers = tuple(eqx.tree_at(lambda b: b.conv2, l, scaled(l.conv2))
                       if hasattr(l, "conv2") else l for l in net.layers)
        return eqx.tree_at(lambda n: n.layers, net, layers)
    raise TypeError(f"scale_residual_branches knows ResConvTrunk, not {type(net).__name__}")
