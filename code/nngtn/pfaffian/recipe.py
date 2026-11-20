"""The contraction kernels' tuned settings, per contraction dtype and GPU generation.

The settings live in `recipes/*.env`, one file a dtype and card:

| file                | library | card                                   | used for                          |
|---------------------|---------|----------------------------------------|-----------------------------------|
| `gh200_nv3_f32.env` | float   | Hopper (H100, H200, GH200; 227 KB smem) | float32 contraction on Hopper     |
| `a100_nv3_f32.env`  | float   | Ampere (A100; 164 KB smem)              | float32 contraction on an A100    |
| `gh200_nv3.env`     | double  | every card                              | float64 contraction               |

`dmma=True` (float32 only) appends one compile flag to the card's float32 recipe, `DMMA_FLAG`
(`-DPIVOTED_GCLOSE_DMMA=1`): the global-memory kernels' panel close runs on the fp64 tensor cores, its float operands
converted in registers, instead of on the CUDA cores (identical products, summed by the tensor core). It acts only
on levels wide enough for the global kernels (32x32 lattices and larger) and is off by default; the paper's timings
ran with it.

The A100 float recipe differs from the Hopper one in one compile flag: two warp fronts a block instead of four (four
do not fit the float gradient's shared memory on an A100 once a level is 96 wide); both float recipes take the block
kernel's residency floor 6, the float64 recipe 3. The kernel ARITHMETIC is the same on both cards; only the launch
geometry differs.

`apply` reads the file for the dtype and card and exports its `PIVOTED_*` knobs (the same in every recipe) into the
environment, keeping any knob the caller has already set. Each kernel library is compiled with its own recipe's
flags: the float library with the card's float32 recipe (`NNGTN_NVCC_FLAGS_FLOAT`), the double library with the
float64 recipe (`NNGTN_NVCC_FLAGS_DOUBLE`), so a process may build both in either order; a `NNGTN_NVCC_FLAGS` set by
the caller is used for both instead.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

__all__ = ["RECIPES", "DMMA_FLAG", "recipe_file", "read_env", "settings", "is_hopper", "apply"]

RECIPES = Path(__file__).resolve().parent / "recipes"
#: the compile flag `dmma=True` appends to a float32 recipe
DMMA_FLAG = "-DPIVOTED_GCLOSE_DMMA=1"
#: `(card, dmma)` of this process's first `apply`: a library is compiled once a process, so these cannot change
_applied = None



def is_hopper(device=None):
    """Whether `device` (default: the first jax device) is a Hopper card (H100, H200, GH200, H800, H20)."""
    import jax
    kind = getattr(device or jax.devices()[0], "device_kind", "")
    return any(k in kind for k in ("H100", "H200", "GH200", "H800", "H20"))


def recipe_file(contraction_dtype="float32", card=None):
    """The recipe file for a contraction dtype and a card (`"hopper"`, `"a100"`, or `None`: the first device's).
    Cards other than Hopper take the A100 recipe."""
    if contraction_dtype not in ("float32", "float64"):
        raise ValueError(f"contraction_dtype={contraction_dtype!r}; float32 or float64")
    if contraction_dtype == "float64":
        return RECIPES / "gh200_nv3.env"
    if card is None:
        card = "hopper" if is_hopper() else "a100"
    if card not in ("hopper", "a100"):
        raise ValueError(f"card={card!r}; 'hopper' or 'a100'")
    return RECIPES / ("gh200_nv3_f32.env" if card == "hopper" else "a100_nv3_f32.env")


def read_env(path):
    """`{name: value}` of the `export NAME=VALUE` lines of a recipe file, quotes removed, trailing comments dropped."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        words = shlex.split(line[len("export "):], comments=True)
        if not words:
            continue
        name, sep, value = words[0].partition("=")
        if not sep:
            raise ValueError(f"{path}: cannot read {line!r}")
        out[name] = value
    return out


def settings(contraction_dtype="float32", card=None, dmma=False):
    """`(file, {name: value})`: the recipe for this dtype and card as `read_env` reads it, with `DMMA_FLAG` appended to
    its compile flags when `dmma` (float32 only). Nothing is exported."""
    if dmma and contraction_dtype != "float32":
        raise ValueError("dmma acts on the float32 contraction only")
    path = recipe_file(contraction_dtype, card)
    env = read_env(path)
    if dmma:
        env["NNGTN_NVCC_FLAGS"] = f"{env['NNGTN_NVCC_FLAGS']} {DMMA_FLAG}"
    return path, env


def apply(contraction_dtype="float32", card=None, dmma=False):
    """Export the recipe for this dtype and card (knobs already set are kept); returns `(file, settings in force)`.
    Must run before the first contraction is built (the kernels compile and read their knobs then). One process
    builds each kernel library once, so its first call fixes the card and `dmma`: a later call asking for another
    raises instead of silently running the first one's kernels."""
    global _applied
    if card is None:
        card = "hopper" if is_hopper() else "a100"
    if _applied is not None and _applied != (card, bool(dmma)):
        raise RuntimeError(f"this process's kernels follow card={_applied[0]!r}, dmma={_applied[1]}; "
                           f"card={card!r}, dmma={bool(dmma)} needs a process of its own")
    path, wanted = settings(contraction_dtype, card, dmma)
    out = {}
    for name, value in wanted.items():
        if name == "NNGTN_NVCC_FLAGS":
            continue
        os.environ.setdefault(name, value)
        out[name] = os.environ[name]
    if "NNGTN_NVCC_FLAGS" in os.environ:            # the caller's own flags, for both libraries
        out["NNGTN_NVCC_FLAGS"] = os.environ["NNGTN_NVCC_FLAGS"]
    else:
        for real, dtype in (("FLOAT", "float32"), ("DOUBLE", "float64")):
            name = f"NNGTN_NVCC_FLAGS_{real}"
            flags = settings(dtype, card, dmma and dtype == "float32")[1]["NNGTN_NVCC_FLAGS"]
            os.environ.setdefault(name, flags)
            out[name] = os.environ[name]
    _applied = (card, bool(dmma))
    return path, out
