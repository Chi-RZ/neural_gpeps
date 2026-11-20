# nngtn

Neural-network Gaussian PEPS (NN-GPEPS) for the t-t'-U Hubbard model on periodic `Lx x Ly` tori: a convolutional
network emits a fermionic Gaussian PEPS for each configuration, its amplitude is the Pfaffian of a sparse skew
matrix computed on the GPU, and the network is optimised by variational Monte Carlo. The paper's states and
measurements are in `../data`.

## The method in brief

- **State** (paper, Eqs. 2-4 and Appendix A). For a configuration `s` the network gives each site a real skew
  pairing matrix over its `m = 2 + 4 Nv` modes; the sites are joined by bond pairs, and
  `psi(s) = epsilon(n) 2^(-b/2) Pf(M[Z(s)])`. A state can be projected onto the 8-image group
  `"flip0@flip1@spininverse"` (two reflections and spin inversion).
- **Network** (Appendix B). Residual CNN with circular padding and a 1x1 head; presets in `nngtn.NETWORKS`:
  Nv = 3: 16 blocks, 91 channels (2.4 M parameters); Nv = 4: 24 blocks, 153 channels (10.1 M).
- **Contraction** (Appendix A). Multifrontal threshold-pivoted Parlett-Reid elimination along a nested dissection,
  CUDA kernels for the forward pass and its gradient; pivot threshold 0.05, up to `delay_slots` delayed pivots a
  front (default 20), float32 or float64.
- **Optimisation** (Appendix D). quantax's AdamSR with an LSMR solve under a decaying step-norm cap; samples from
  Metropolis chains at fixed `(N_up, N_dn)`.
- **Measurement** (Appendix D, G). Energy, variance, pair, spin and charge correlations at fixed weights, block
  jackknife over sweeps. A contraction that overflows its delay slots gives zero amplitude (psi-tilde): an overflowed
  connected term counts as zero, and a sample whose own amplitude overflows is excluded and counted.

## Install

An NVIDIA GPU (A100, H100, H200 or GH200) with a CUDA 12 driver, `nvcc` 12.1 or newer, g++ 11 or newer, and
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra test
uv run python -m pytest tests -q      # the first run compiles the CUDA kernels (a few minutes, cached)
```

Without uv: `pip install -r requirements.txt && pip install --no-deps -e .` in a Python 3.12 environment. If `nvcc` is
not on `PATH`, set `NNGTN_NVCC`. The lock pins every library, cuDNN and cuBLAS included, because they choose the
network's numerical algorithms.

## Package layout

```
nngtn/
    system.py         the problem: periodic lattice and bonds, the t-t'-U Hamiltonian, symmetry projections by name
    state.py          the NN-GPEPS wavefunction: the residual CNN trunk, NeuralGTN, build_model, make_state,
                      load_weights / save_weights
    vmc.py            sampler, AdamSR + LSMR, step-norm cap, the training loop `run`
    measure.py        the measurement loop
    obs.py            the observables and their jackknife assembly
    io.py             chains in and out, training configs (TOML)
    pfaffian/         the contraction: tree.py builds the contraction tree once (layout, nested dissection,
                      plan), contract.py evaluates each amplitude on it, front_cuda.py/.cu the GPU kernels,
                      recipe.py their tuned settings per card (recipes/*.env)
    cli/              the commands nngtn-train and nngtn-measure, and distributed.py (several nodes, one
                      process a node)
examples/             a 4x4 training from scratch
tests/                the package tests
```

## Training

A training is a TOML file of stages (`nngtn.io.load_config`), each with its lattice, start (random or an earlier stage's
weights), chains, step-norm cap, samples and contraction. The weights carry over to any lattice size.

```bash
uv run nngtn-train examples/hubbard_4x4.toml --stage cold --out runs/4x4
uv run nngtn-train examples/hubbard_4x4.toml --stage cont --out runs/4x4
```

A stage writes `trajectory.json`, `weights.eqx` and `chains.npy`. Several GPUs and nodes run one process a node
(`srun --ntasks-per-node=1 ... nngtn-train ...`): `samples` is the total over all cards, `max_parallel` the
configurations a chunk per card (a projected state evaluates each at its 8 images). Runs are bitwise reproducible on
one card with `--xla_gpu_deterministic_ops=true` added to the default `XLA_FLAGS` of `nngtn/cli/train.py`
(an `XLA_FLAGS` set beforehand replaces those defaults, so set the whole list). Without it, GPU reductions whose
summation order varies from run to run change the last bits of the amplitudes, so two runs differ from the first
sampling sweep on and then behave like independent runs of the same training, as for any GPU VMC code.

## Measuring

```bash
cd ../data && uv run nngtn-measure --config measurements/proj16x4-t0-nv4/config.toml --out runs/x   # paths relative to data/
```

or the same keys as options (`--weights`, `--Lx`, `--Ly`, `--nparticles`, `--symm`, `--nsamples`, `--nsweeps`, ...).
The default `--precision highest` runs every matmul in full float32, without TF32.

## Using it in your own code

```python
import nngtn as ng

ng.make_sites(16, 16, n_particles=(112, 112))                 # quantax's lattice; set it first
model = ng.load_weights(ng.build_model(16, 16, nv=3), "state.eqx")
state = ng.make_state(model, symm="flip0@flip1@spininverse", max_parallel=128)
psi = state(ng.read_chains("chains.npz")[:64])                # psi.sign, psi.logabs
H = ng.hamiltonian(16, 16, u=8.0, tprime=0.0)
```

A configuration is `+-1` occupations, spin-up sites first, site `(i, j)` at index `i * Ly + j`.

## Tests

`tests/test_nngtn.py`. CPU: lattice bonds, recipe files, chain formats, configs, the padded CNN trunk against quantax's blocks, the psi-tilde re-read. GPU, each in its own process: the contraction against the dense Pfaffian and the
amplitude against the Gaussian PEPS built by brute force in Fock space, gradients against finite differences,
float32 against float64, the Hamiltonian against quantax's, overflow handling, a few VMC steps. Run them with one
card visible (quantax uses every visible device): `CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests`, about 15 minutes
on an A100.
