"""The symmetry a state is projected onto, by name — quantax's symmetry classes joined by `@`.

`"flip0@flip1@spininverse"` is quantax's `Flip(axis=0) @ Flip(axis=1) @
SpinInverse()`, every generator in its trivial sector, a group of 8. quantax evaluates the model on all 8 images of
every configuration and sums them with their fermionic signs, so each forward pass costs 8 times as much.
"""
from __future__ import annotations

__all__ = ["SYMMETRY_TOKENS", "build_symm"]

SYMMETRY_TOKENS = ("flip0", "flip1", "spininverse")


def build_symm(spec=None):
    """The quantax symmetry `spec` names (`@`-joined tokens from `SYMMETRY_TOKENS`: `flip0` and `flip1` the
    lattice's mirrors `Flip(axis=0)` and `Flip(axis=1)`, `spininverse` quantax's `SpinInverse()`, each in its trivial
    sector), or `None` for `None`/`"none"` (no projection).
    Needs the lattice defined (`lattice.make_sites`): quantax reads the coordinates from it."""
    if spec is None or spec == "none":
        return None
    from quantax.symmetry import Flip, SpinInverse
    table = {"flip0": lambda: Flip(axis=0), "flip1": lambda: Flip(axis=1),
             "spininverse": SpinInverse}
    names = spec.split("@")
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown symmetry {unknown}; expected '@'-joined names from {SYMMETRY_TOKENS}")
    symm = table[names[0]]()
    for n in names[1:]:
        symm = symm @ table[n]()
    return symm


