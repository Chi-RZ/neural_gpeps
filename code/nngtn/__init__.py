"""nngtn: ground states of the t-t'-U Hubbard model on periodic Lx x Ly tori with a neural Gaussian
tensor-network state, contracted by a sparse threshold-pivoted Pfaffian on the GPU, optimised by variational Monte
Carlo (quantax), and measured (energy, d-wave pairing, stripe spin and charge order).

    import nngtn as ng
    ng.make_sites(16, 16, n_particles=(112, 112))
    model = ng.load_weights(ng.build_model(16, 16, nv=3), "state.eqx")
    state = ng.make_state(model, symm="flip0@flip1@spininverse")
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .io import load_chains, read_chains, save_chains, write_chains  # noqa: E402
from .state import NETWORKS, NeuralGTN, build_model, load_weights, make_state, save_weights  # noqa: E402
from .system import build_symm, diagonal_bonds, hamiltonian, hubbard_operator, make_sites, square_bonds  # noqa: E402
from .vmc import run  # noqa: E402

__version__ = "0.1.0"

__all__ = ["NETWORKS", "NeuralGTN", "build_model", "load_weights", "save_weights", "make_state", "make_sites",
           "square_bonds", "diagonal_bonds", "hubbard_operator", "hamiltonian", "build_symm", "run",
           "load_chains", "read_chains", "save_chains", "write_chains"]
