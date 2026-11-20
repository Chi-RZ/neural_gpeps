# Data

The paper's trained states and their measurements. Every state is a neural Gaussian tensor-network state
of the Hubbard model at U = 8, t = 1, t' = 0 and 1/8 hole doping on a periodic torus. The `proj` states were trained
under the 8-image symmetry projection `flip0@flip1@spininverse` (Tables I and G1); the two `unproj` states are the 16x16
states the projected training continued from, the ones the paper transfers to 32x32 (Table II). The code is
`../code` (`nngtn`).

### The projected states (Tables I, G1 and G3, Figs. 3 and G1)

| state | lattice | Nv | E/N | sigma^2/N | Delta^2_SC | M^2_str |
|---|---|---:|---:|---:|---:|---:|
| `proj16x16-t0-nv3` | 16x16, 112 + 112 | 3 | -0.759835(14) | 0.0149(7) | 1.23(13)e-4 | 0.0138749(78) |
| `proj16x16-t0-nv4` | 16x16, 112 + 112 | 4 | -0.760115(11) | 0.01287(24) | 1.37(20)e-4 | 0.013725(10) |
| `proj16x4-t0-nv3` | 16x4, 28 + 28 | 3 | -0.765883(26) | 0.0133(7) | 8.18(33)e-4 | 0.032464(31) |
| `proj16x4-t0-nv4` | 16x4, 28 + 28 | 4 | -0.766152(16) | 0.01095(42) | 8.43(41)e-4 | 0.032403(27) |

Every number is a fixed-weight measurement: 64 sweeps of 8192 samples, float32 throughout (no TF32), the delay slots
of each `measurements/<state>/config.toml`. Errors are one standard error. Each `results.json` also gives
`Delta_SC` (the square root of Delta^2_SC), the distance window of Delta^2_SC (separations r >= `r_max` = d_max / 2,
`n_far` of them), the momenta of the spin peaks (`k_alpha`) and charge peaks (`q_c`) in units of 2 pi / L, and the
counts of excluded sweeps and samples and of non-finite terms, all zero in these measurements. The series were
written before `nngtn.measure` counted `E_restored`, `E_overflow_terms` and `E_own_overflow`; all three are zero,
because every sample with a
non-finite local energy is counted in `E_excluded`, which is zero in every sweep.

### The transfer to 32x32 (Table II, Table E1)

The unprojected 16x16 states read on the 32x32 torus (448 + 448) at exactly their weights, 1024 samples a reading:

| state | projection at evaluation | sampling | estimator | readings | E/N | sigma^2/N |
|---|---|---|---|---:|---:|---:|
| `unproj16x16-t0-nv3` | none | float32 | float32 | 64 | -0.757451(21) | 0.02407(51) |
| `unproj16x16-t0-nv4` | none | float32 | float32 | 64 | -0.758019(22) | 0.02013(42) |
| `unproj16x16-t0-nv3` | 8 images | float32 | float32 | 32 | -0.758414(27) | 0.01647(64) |
| `unproj16x16-t0-nv4` | 8 images | float32 | float32 | 32 | -0.758727(33) | 0.01606(82) |
| `unproj16x16-t0-nv3` | none | TF32 | float32 reweighted | 64 | -0.757429(27) | 0.02398(76) |
| `unproj16x16-t0-nv3` | none | TF32 | TF32 | 64 | -0.757396(36) | 0.05573(81) |
| `unproj16x16-t0-nv4` | none | TF32 | float32 reweighted | 64 | -0.758021(18) | 0.02051(68) |
| `unproj16x16-t0-nv4` | none | TF32 | TF32 | 64 | -0.757979(28) | 0.04781(75) |

The two TF32 rows of a state are one read: both estimators on the same TF32-sampled configurations. The energy and
variance pool each reading's moments with equal weight; the error is the largest block-jackknife standard error over
blocks of 1, 2, 4 and 8 readings that leave at least eight blocks (`scripts/transfer.py`).

`states.json` holds every number with its source, and the sha256 of every weight, chain, measurement and 32x32
read file (the timings and scripts are not hashed).

## Layout

```
states.json                 one row a state: lattice, network, contraction, numbers, sha256 of each of its files
weights/<state>.eqx         the network weights
weights/<state>_spins.npz   8192 equilibrated configurations (packed bits)
measurements/<state>/       config.toml (the measurement's settings, nngtn-measure input), series.npz (per-sweep
                            raw output), results.json (the numbers), chains_in.npz (start chains)
measurements/transfer32x32/ the 32x32 reads of the unprojected states: readings.csv (every reading's energy and
                            variance), results.json (the numbers), nv3_f32.toml, nv4_f32.toml, nv3_proj.toml,
                            nv4_proj.toml (the float32 reads' settings, nngtn-measure input)
timing/                     the paper's timings (A100, NERSC Perlmutter): amplitude.csv (Tables C1-C2, Fig. C1),
                            precision.csv (Table C3, Fig. C2)
scripts/assemble.py         results.json from series.npz (CPU)
scripts/transfer.py         measurements/transfer32x32/results.json from readings.csv (CPU)
scripts/timing.py           the paper's timings with this package
tests/                      checksums, assembly, chains, configs; run_gpu_checks.sh
```

## Install

```bash
uv sync --frozen --extra test
uv run python -m pytest tests -q
```

## Using a state

```python
import nngtn as ng                                     # run from data/
ng.make_sites(16, 4, (28, 28))
model = ng.load_weights(ng.build_model(16, 4, nv=4), "weights/proj16x4-t0-nv4.eqx")
state = ng.make_state(model, symm="flip0@flip1@spininverse")
spins = ng.load_chains("weights/proj16x4-t0-nv4_spins.npz")
psi = state(spins[:16])                                   # psi.sign, psi.logabs
```

## Reproducing the numbers

```bash
uv run python scripts/assemble.py proj16x4-t0-nv4                     # the published numbers from the series (CPU)

uv run nngtn-measure --config measurements/proj16x4-t0-nv4/config.toml --out runs/x \
    --chains-in weights/proj16x4-t0-nv4_spins.npz --thermal 0     # a new measurement from the stored chains
uv run nngtn-measure --config measurements/proj16x4-t0-nv4/config.toml --out runs/x   # the original procedure

uv run python scripts/transfer.py                                     # Table II and E1 from the readings (CPU)
uv run nngtn-measure --config measurements/transfer32x32/nv3_proj.toml --out runs/t   # a new 32x32 read

uv run python -u scripts/timing.py amplitude --out runs/timing          # the timings
```

A new measurement agrees with `results.json` statistically. A full one is 64 sweeps of 8192 samples, which at 16x16
needs several GPU nodes; add `--nsamples 256 --nsweeps 2` for a short check.
