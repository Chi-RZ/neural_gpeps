# Scalable Simulation of Strongly Correlated Fermions with Neural Gaussian Tensor Networks

Runze Chi, Ao Chen, Garnet Kin-Lic Chan (California Institute of Technology)

**Paper:** [arXiv:2611.26604](https://arxiv.org/abs/2611.26604)

Neural-network Gaussian projected entangled-pair states (NN-GPEPS) let a convolutional network choose the local
Gaussian tensors of a fermionic PEPS for each configuration. Each amplitude stays an exact contraction: the
Pfaffian of a sparse matrix, eliminated along a nested dissection at a cost of O(N^{3/2}) on a two-dimensional
lattice. For the Hubbard model at U/t = 8 and 1/8 hole doping, the paper reports variational energies on the
periodic 16 x 4 and 16 x 16 lattices and transfers the 16 x 16 networks unchanged to 32 x 32. This repository holds
the method's code and every trained state and measurement behind the paper's tables and figures.

## Layout

```
nngtn/
├── code/   the Python package: the network, the GPU Pfaffian contraction (CUDA), VMC training and measurement
└── data/   the trained states, their measurements, the 32 x 32 transfer reads, the timings, and the scripts
            that reproduce the paper's numbers from them
```

[code/README.md](code/README.md) describes the package and [data/README.md](data/README.md) the states and
measurements.

## Installation

Linux, Python 3.12, an NVIDIA A100, H100, H200 or GH200 with a CUDA 12 driver, `nvcc` 12.1 or newer and g++ 11 or
newer. The environment is locked with [uv](https://docs.astral.sh/uv/); `code/requirements.txt` gives the same pins
for pip.

```bash
cd code && uv sync --frozen --extra test && uv run python -m pytest tests -q
cd ../data && uv sync --frozen --extra test && uv run python -m pytest tests -q
```

The first GPU use compiles the CUDA kernels, which takes a few minutes; they are cached afterwards. Reproducing the
paper's numbers from the stored measurements needs only a CPU.

## Usage

Evaluate a released state, from `data/`:

```python
import nngtn as ng

ng.make_sites(16, 4, (28, 28))                                   # the lattice comes first
model = ng.load_weights(ng.build_model(16, 4, nv=4), "weights/proj16x4-t0-nv4.eqx")
state = ng.make_state(model, symm="flip0@flip1@spininverse")
spins = ng.load_chains("weights/proj16x4-t0-nv4_spins.npz")      # 8192 equilibrated configurations
psi = state(spins[:16])                                          # psi.sign, psi.logabs
```

Common commands, from `data/`:

```bash
# the paper's energies and order parameters from the stored measurements (CPU, seconds)
uv run python scripts/assemble.py proj16x16-t0-nv4
uv run python scripts/transfer.py

# a short measurement from a state's stored chains (errors need at least 8 sweeps);
# drop the last two options for the full 64 x 8192
uv run nngtn-measure --config measurements/proj16x4-t0-nv4/config.toml --out runs/x \
    --chains-in weights/proj16x4-t0-nv4_spins.npz --thermal 0 --nsamples 256 --nsweeps 2
```

and a training from scratch, from `code/`:

```bash
uv run nngtn-train examples/hubbard_4x4.toml --stage cold --out runs/4x4
```

## Released states

| state | lattice | N_v | projection | in the paper |
|---|---|---:|---|---|
| `proj16x4-t0-nv3` | 16 x 4, 28 + 28 electrons | 3 | 8 images | Table G1 |
| `proj16x4-t0-nv4` | 16 x 4, 28 + 28 | 4 | 8 images | Table G1, Fig. G1 |
| `proj16x16-t0-nv3` | 16 x 16, 112 + 112 | 3 | 8 images | Table I |
| `proj16x16-t0-nv4` | 16 x 16, 112 + 112 | 4 | 8 images | Table I, Fig. 3 |
| `unproj16x16-t0-nv3` | 16 x 16, 112 + 112 | 3 | none | Table II (read on 32 x 32) |
| `unproj16x16-t0-nv4` | 16 x 16, 112 + 112 | 4 | none | Table II (read on 32 x 32) |

Each state has its weights and 8192 equilibrated configurations in `data/weights/`. The measured values, the
32 x 32 reads and the file checksums are in [data/README.md](data/README.md) and `data/states.json`.

## Citation

```bibtex
@article{Chi2026NeuralGaussian,
  title   = {Scalable Simulation of Strongly Correlated Fermions with Neural Gaussian Tensor Networks},
  author  = {Chi, Runze and Chen, Ao and Chan, Garnet Kin-Lic},
  journal = {arXiv preprint arXiv:2611.26604},
  eprint  = {2611.26604},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
