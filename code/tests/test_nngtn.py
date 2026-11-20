"""Package tests. CPU: the lattice's bond lists, the Hamiltonian's matrix elements, the recipe files, the chain
formats, the training-config loader, the stripe momenta, the padded CNN trunk against quantax's own blocks,
the re-evaluation at full precision, the psi-tilde re-read and the refusal of a non-finite step.
GPU, each in its own process: the contraction against the dense Pfaffian (shared-memory and global kernels, with
and without DMMA), the amplitude against the Gaussian PEPS built by brute force in Fock space, gradients against
finite differences, float32 against float64, the Hamiltonian against quantax's, overflow handling, and a few VMC
steps on a 4x4 torus with and without projection and t', with a continuation."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# the XLA flags the CLI sets (autotuners and command buffers off)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner="
                      "false --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=")

import nngtn as ng  # noqa: E402
from nngtn import io as nio  # noqa: E402
from nngtn.pfaffian import recipe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _gpu():
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


gpu = pytest.mark.skipif(not _gpu(), reason="the pivoted-Pfaffian kernels need a GPU")


# ---- CPU ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("Lx,Ly", [(2, 2), (4, 4), (16, 4), (3, 5)])
def test_bonds(Lx, Ly):
    nn, nnn = ng.square_bonds(Lx, Ly), ng.diagonal_bonds(Lx, Ly)
    assert len(nn) == 2 * Lx * Ly and len(nnn) == 2 * Lx * Ly
    assert (np.bincount(np.array(nn).ravel(), minlength=Lx * Ly) == 4).all()
    assert (np.bincount(np.array(nnn).ravel(), minlength=Lx * Ly) == 4).all()
    # the enumeration: per site i * Ly + j, +x then +y; diagonals +x+y then +x-y
    assert nn[:2] == [(0, Ly), (0, 1)]
    assert nnn[:2] == [(0, Ly + 1), (0, Ly + Ly - 1)]


def test_recipe_files_are_the_tuned_ones():
    """Every recipe's settings, as `recipe.read_env` reads them, are the tuned ones: the shared knobs and compile
    flags, and the cards' one difference (warp fronts a block)."""
    knobs = {"PIVOTED_PANEL": "16", "PIVOTED_SEARCH_WINDOW": "1", "PIVOTED_LEAF_CUTOFF": "0",
             "PIVOTED_GLOBAL_BWD_FROM": "176", "PIVOTED_GBWD_PANEL": "8"}

    def flags(wpb, minblocks, dmma=False):
        return ("-DPIVOTED_GLIVE=7 -DPIVOTED_BSELECT_WARP=1 -DPIVOTED_WARP_COMPACT=24 -DPIVOTED_BLOCK_THREADS=256 "
                f"-DPIVOTED_BLOCK_COMPACT=0 -DPIVOTED_GFWD_THREADS=512 -DPIVOTED_WPB={wpb} -DPIVOTED_BWD_WORK=25 "
                f"-DPIVOTED_BLOCK_MINBLOCKS={minblocks} -DPIVOTED_GUU=1 -DPIVOTED_GBWD_LIVE=1 -DPIVOTED_BWD_MINBLOCKS=3 "
                "-DPIVOTED_GBWD_MINBLOCKS=2" + (" -DPIVOTED_GCLOSE_DMMA=1" if dmma else ""))
    want = {"gh200_nv3.env": flags(2, 3), "gh200_nv3_f32.env": flags(4, 6), "a100_nv3_f32.env": flags(2, 6)}
    assert sorted(p.name for p in recipe.RECIPES.glob("*.env")) == sorted(want)
    for name, nvcc in want.items():
        assert recipe.read_env(recipe.RECIPES / name) == {**knobs, "NNGTN_NVCC_FLAGS": nvcc}, name
    a100 = recipe.read_env(recipe.recipe_file("float32", "a100"))
    hopper = recipe.read_env(recipe.recipe_file("float32", "hopper"))
    assert "-DPIVOTED_WPB=2" in a100["NNGTN_NVCC_FLAGS"] and "-DPIVOTED_WPB=4" in hopper["NNGTN_NVCC_FLAGS"]
    assert a100["PIVOTED_PANEL"] == "16"
    assert recipe.recipe_file("float64", "a100").name == "gh200_nv3.env"
    # dmma: the float32 recipe plus DMMA_FLAG, the flags the paper's timings were built with
    for card, wpb in (("hopper", 4), ("a100", 2)):
        assert recipe.settings("float32", card, dmma=True)[1] == {**knobs, "NNGTN_NVCC_FLAGS": flags(wpb, 6, True)}



def test_dmma_adds_only_the_dmma_flag():
    """`dmma` takes the card's float32 recipe with `-DPIVOTED_GCLOSE_DMMA=1` appended to the compile flags, every
    other setting equal; `dmma` is refused for the float64 contraction."""
    for card in ("a100", "hopper"):
        plain = recipe.settings("float32", card)[1]
        dmma = recipe.settings("float32", card, dmma=True)[1]
        assert dmma.keys() == plain.keys()
        for k in plain:
            if k == "NNGTN_NVCC_FLAGS":
                assert dmma[k] == plain[k] + " -DPIVOTED_GCLOSE_DMMA=1"
            else:
                assert dmma[k] == plain[k], k
    with pytest.raises(ValueError):
        recipe.settings("float64", "a100", dmma=True)

def test_chains_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    s = np.where(rng.random((10, 2 * 12)) < 0.5, 1, -1).astype(np.int8)
    assert np.array_equal(nio.unpack(**nio.pack(s)), s)
    for name in ("c.npy", "c.npz"):
        nio.write_chains(tmp_path / name, s)
        assert np.array_equal(nio.read_chains(tmp_path / name), s)


def test_load_chains_takes_the_first_n_and_checks_the_sector(tmp_path):
    """A chains file with more chains than the run needs gives its first `n_samples`; fewer, or a chain outside the
    lattice's particle sector, is an error."""
    import nngtn as ng
    ng.make_sites(2, 2, (1, 1))                       # 4 sites, one up and one down electron
    rng = np.random.default_rng(1)
    s = -np.ones((10, 8), np.int8)
    for row in s:
        row[rng.integers(4)] = 1
        row[4 + rng.integers(4)] = 1
    nio.write_chains(tmp_path / "c.npz", s)
    assert np.array_equal(nio.load_chains(tmp_path / "c.npz", 4), s[:4])
    assert np.array_equal(nio.load_chains(tmp_path / "c.npz"), s)
    with pytest.raises(ValueError):
        nio.load_chains(tmp_path / "c.npz", 11)
    bad = s.copy()
    bad[0, :4] = 1
    nio.write_chains(tmp_path / "bad.npz", bad)
    with pytest.raises(ValueError):
        nio.load_chains(tmp_path / "bad.npz", 4)


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.toml")), ids=lambda p: p.stem)
def test_configs_load(path):
    cfg = nio.load_config(path)
    assert cfg["stage"] and cfg["stage"][-1]["lattice"] == {**cfg["lattice"], **cfg["stage"][-1].get("lattice", {})}
    for st in cfg["stage"]:
        assert st["contraction"]["dtype"] in ("float32", "float64")
        assert st["solver"] in ("lsmr", "minsr_chunked")


def _dense_pfaffian(A):
    """`(sign, log|Pf A|)` by completely pivoted skew Parlett-Reid on a dense copy (a reference)."""
    A = A.copy()
    n = A.shape[0]
    if n % 2:
        return 0, None
    sign, logabs = 1, 0.0
    for j in range(0, n, 2):
        block = np.abs(A[j:, j:])
        flat = int(np.argmax(block.reshape(-1)))
        r, c = divmod(flat, n-j)
        r, c = r+j, c+j
        if block.reshape(-1)[flat] == 0:
            return 0, None
        if r != j:
            A[[r, j]] = A[[j, r]]
            A[:, [r, j]] = A[:, [j, r]]
            sign = -sign
            if c == j:                  # the partner moved with the swap
                c = r
        if c != j+1:
            A[[c, j+1]] = A[[j+1, c]]
            A[:, [c, j+1]] = A[:, [j+1, c]]
            sign = -sign
        a = A[j, j+1]
        if a < 0:
            sign = -sign
        logabs = logabs+np.log(abs(a))
        if j+2 < n:
            tau = A[j, j+2:]/a
            column = A[j+2:, j+1]
            A[j+2:, j+2:] -= np.outer(column, tau)-np.outer(tau, column)
    return sign, logabs






# ---- GPU ----------------------------------------------------------------------------------------------------------


def _isolate(request):
    """Run this test alone in a child process and report its outcome; True in the parent, False in the child.

    quantax keeps the lattice (and what it compiles against it) in process-wide globals, so two lattices in one
    process interfere; every GPU test that builds a lattice runs in its own process."""
    import subprocess
    import sys
    if os.environ.get("NNGTN_TEST_CHILD"):
        return False
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", request.node.nodeid],
                       cwd=ROOT, env={**os.environ, "NNGTN_TEST_CHILD": "1"}, capture_output=True, text=True)
    tail = (r.stdout[-6000:] + r.stderr[-2000:]).strip()
    if r.returncode == 0 and " skipped" in r.stdout and " passed" not in r.stdout:
        pytest.skip("in the child process: " + r.stdout.strip().splitlines()[-2][:300])
    assert r.returncode == 0, tail
    return True


def _configs(L, n, seed=0):
    rng = np.random.default_rng(seed)
    N = L * L
    s = -np.ones((n, 2 * N), np.int8)
    for b in range(n):
        s[b, rng.choice(N, N // 2, replace=False)] = 1
        s[b, N + rng.choice(N, N // 2, replace=False)] = 1
    return s


@gpu
def test_float32_and_float64_contractions_agree(request):
    if _isolate(request):
        return
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    ng.make_sites(4, 4, (8, 8))
    m32 = ng.build_model(4, 4, nblocks=2, channels=16, pad_channels=0, contraction_dtype="float32")
    m64 = m32.with_contraction(contraction_dtype="float64")
    s = jnp.asarray(_configs(4, 32))
    f = lambda m: eqx.filter_jit(jax.vmap(lambda mm, x: mm(x), in_axes=(None, 0)))(m, s)  # noqa: E731
    a, b = f(m32), f(m64)
    assert np.isfinite(np.asarray(b.logabs)).all()
    np.testing.assert_array_equal(np.asarray(a.sign), np.asarray(b.sign))
    assert np.max(np.abs(np.asarray(a.logabs, np.float64) - np.asarray(b.logabs))) < 1e-3


@gpu
@pytest.mark.parametrize("symm,tprime", [(None, 0.0), ("flip0@flip1@spininverse", -0.2)])
def test_a_few_vmc_steps_run(tmp_path, symm, tprime, request):
    if _isolate(request):
        return
    ng.make_sites(4, 4, (7, 7))
    model = ng.build_model(4, 4, nblocks=2, channels=16, pad_channels=0)
    rows, model = ng.run(model, n_particles=(7, 7), tprime=tprime, symm=symm, steps=3, n_samples=64, thermal=16,
                         sweeps=4, max_parallel=(64, 64), chains_out=str(tmp_path / "c.npy"))
    assert len(rows) == 3
    assert all(np.isfinite(r["energy"]) and np.isfinite(r["var_e"]) for r in rows)
    # a continuation from the written chains and weights
    rows2, _ = ng.run(model, n_particles=(7, 7), tprime=tprime, symm=symm, steps=1, n_samples=64, sweeps=4,
                      max_parallel=(64, 64), chains_in=str(tmp_path / "c.npy"), chains_thermal=4, cap_step_offset=3)
    assert np.isfinite(rows2[0]["energy"])




def test_stripe_momenta():
    """The stripe wavevector q = pi/8 at t' = 0 and pi/4 at t' = -0.2: on a square torus the spin peaks sit at
    (pi -+ q, pi), (pi, pi -+ q) and the charge's at +-(2q, 0), +-(0, 2q); on a 16x4 strip only the x direction fits
    the momentum grid; a t' with no recorded wavevector, or a lattice where neither direction fits, raises."""
    from nngtn import obs
    spin, charge = obs.stripe_momenta(16, 16, 0.0)
    assert spin == ((7, 8), (9, 8), (8, 7), (8, 9)) and charge == ((2, 0), (14, 0), (0, 2), (0, 14))
    spin, charge = obs.stripe_momenta(16, 16, -0.2)
    assert spin == ((6, 8), (10, 8), (8, 6), (8, 10)) and charge == ((4, 0), (12, 0), (0, 4), (0, 12))
    spin, charge = obs.stripe_momenta(16, 4, 0.0)
    assert spin == ((7, 2), (9, 2)) and charge == ((2, 0), (14, 0))
    with pytest.raises(ValueError):
        obs.stripe_momenta(16, 16, -0.3)
    with pytest.raises(ValueError):
        obs.stripe_momenta(12, 4, 0.0)
    for odd in ((16, 5), (15, 16)):                       # pi is not on an odd side's momentum grid
        with pytest.raises(ValueError):
            obs.stripe_momenta(*odd, 0.0)


def test_dwave_terms_refuse_a_two_wide_torus():
    """On a side of 2 two bonds join the same pair of sites, which the d-wave bookkeeping cannot tell apart."""
    from nngtn import obs
    for shape in ((2, 4), (4, 2)):
        with pytest.raises(ValueError):
            obs.DwaveTerms(obs.Torus(*shape))
    assert len(obs.DwaveTerms(obs.Torus(4, 4)).F_keys) == 2 * 2 * 64     # 2 left bonds x 2 orientations x 64 right


def test_config_requires_the_keys_training_reads(tmp_path):
    """A config missing a key the training reads stops at load, not with a KeyError mid-run."""
    text = (ROOT / "examples" / "hubbard_4x4.toml").read_text()
    for drop in ("tprime = 0.0\n", "delay_slots = 20\n", "net_seed = 0\n"):
        assert drop in text, drop
        path = tmp_path / "short.toml"
        path.write_text(text.replace(drop, "", 1))
        with pytest.raises(ValueError, match="lacks"):
            nio.load_config(path)


def test_apply_gives_each_library_its_recipe_and_fixes_card_and_dmma(monkeypatch):
    """`recipe.apply`: the float library takes the card's float32 recipe flags and the double library the float64
    recipe's, whichever dtype is asked first; a second call with another card or dmma raises."""
    for k in [k for k in os.environ if k.startswith(("PIVOTED_", "NNGTN_NVCC_FLAGS"))]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(recipe, "_applied", None)
    recipe.apply("float64", "a100")
    assert os.environ["NNGTN_NVCC_FLAGS_FLOAT"] == recipe.settings("float32", "a100")[1]["NNGTN_NVCC_FLAGS"]
    assert os.environ["NNGTN_NVCC_FLAGS_DOUBLE"] == recipe.settings("float64", "a100")[1]["NNGTN_NVCC_FLAGS"]
    assert os.environ["PIVOTED_PANEL"] == "16"
    recipe.apply("float32", "a100")                          # same card and dmma: fine
    for card, dmma in (("hopper", False), ("a100", True)):
        with pytest.raises(RuntimeError):
            recipe.apply("float32", card, dmma)


@gpu
def test_overflow_is_psitilde_and_reread_clears_it(tmp_path, request):
    """With one delay slot a small random state overflows its contraction on some connected configurations: every
    energy sample stays finite (the overflowed term at zero, `E_restored` counts them), the configurations are saved,
    and a re-read with 64 slots finds no overflowed term left."""
    if _isolate(request):
        return
    import numpy as np

    from nngtn import measure as meas
    ng.make_sites(4, 4, (7, 7))
    model = ng.build_model(4, 4, nblocks=2, channels=16, pad_channels=0, delay_slots=1, net_seed=3)
    state = ng.make_state(model, None, 64)
    H = ng.hamiltonian(4, 4, u=8.0)
    meas.measure(state, H, tmp_path / "m", nparticles=(7, 7), nsamples=64, nsweeps=2, thermal=32, sweeps=16,
                 channels=("energy", "dwave", "spin"), save_every=1)
    z = dict(np.load(tmp_path / "m-series.npz"))
    assert np.isfinite(z["E_mean"]).all() and int(z["E_excluded"].sum()) == 0
    if len(z["bad_spins"]) == 0:
        pytest.skip("no connected configuration overflowed one delay slot with this draw")
    assert int(z["E_restored"].sum()) + int(z["dwave_nonfinite"].sum()) + int(z["spin_nonfinite"].sum()) > 0
    more = ng.make_state(model.with_contraction(delay_slots=64), None, 64)
    out, report = meas.reread(state, more, H, z)
    assert report["terms"]["more"].get("amplitude", 0) == 0
    assert np.isfinite(out["E_mean"]).all()

@pytest.mark.parametrize("Lx,Ly,nup,tprime", [(16, 4, 28, 0.0), (6, 4, 10, -0.2)])
def test_hamiltonian_matrix_elements_match_quantax_hubbard(Lx, Ly, nup, tprime):
    """`hamiltonian` against quantax's own `Hubbard` (bonds built independently): for random configurations, the
    same diagonal elements and, row by row, the same connected configurations with the same hopping elements (the
    slots of hops that cannot happen are NaN in both and skipped by `Oloc`)."""
    import jax.numpy as jnp
    import quantax as qtx
    ng.make_sites(Lx, Ly, (nup, nup))
    N = Lx * Ly
    rng = np.random.default_rng(0)
    s = -np.ones((4, 2 * N), np.int8)
    for b in range(4):
        s[b, rng.choice(N, nup, replace=False)] = 1
        s[b, N + rng.choice(N, nup, replace=False)] = 1
    s = jnp.asarray(s)
    ours = ng.hamiltonian(Lx, Ly, t=1.0, u=8.0, tprime=tprime)
    ref = (qtx.operator.Hubbard(U=8.0, t=[1.0, tprime], n_neighbor=[1, 2]) if tprime
           else qtx.operator.Hubbard(U=8.0, t=1.0, n_neighbor=1))
    np.testing.assert_array_equal(np.asarray(ours.apply_diag(s)), np.asarray(ref.apply_diag(s)))

    def rows(op):
        out = [dict() for _ in range(s.shape[0])]
        for _, sc, h in op.apply_off_diag(s):
            sc, h = np.asarray(sc).reshape(s.shape[0], -1, 2 * N), np.asarray(h).reshape(s.shape[0], -1)
            for b in range(s.shape[0]):
                for c, v in zip(sc[b], h[b]):
                    if np.isfinite(v) and v != 0:
                        key = c.astype(np.int8).tobytes()
                        out[b][key] = out[b].get(key, 0.0) + float(v)
        return out
    a, b = rows(ours), rows(ref)
    assert all(len(r) > 0 for r in a)
    assert a == b


# ---------------------------------------------------------------- correctness of the GPU contraction

# ---- the dense network matrix: the exact reference the GPU contraction is checked against ----------------------
# A projected Gaussian tensor network's amplitude as ONE sparse skew matrix.
#
# THE STATE.  Site `s` is `exp(1/2 c_s^+ Z_s c_s^+)|0>` over its modes, `Z_s`
# real skew; modes are numbered globally site after site. A bond joins mode `p`
# of one site to mode `q` of another through `<0_pq| (1 + sigma c_q c_p)/sqrt2`.
# The amplitude of a physical configuration is
# `<0| c_{n_k} ... c_{n_1} prod_bonds(...) prod_s exp(...)|0>` over the occupied
# physical modes `n_1 < ... < n_k`.
#
# THE MATRIX.  Wick's theorem gives
#
#     psi(n) = epsilon(n) * 2^{-b/2} * Pf(M),   M = (blockdiag_s Z_s + sigma-couplings)_{SS},
#
# with `S` every virtual mode and the occupied physical modes, in global order,
# `M[p_a, q_b] = +sigma` in the bond's own orientation (`p_a` on the side the
# operator names first), and `epsilon(n)` a sign of the occupied positions alone
# (`configuration_sign`); the tests hold both to a brute-force Fock-space amplitude.
# Every entry of `M`
# is a raw site entry or a bond constant: nothing is divided, so there is no
# chart, reference or pole — `Pf M` vanishes only where the amplitude does.

def kept_modes(net, occupation):
    """Global modes of `M`: every non-physical mode, and the occupied physical ones.

    `occupation[s]` is a tuple of 0/1 per physical mode of site `s`.
    """
    keep = []
    for s, m in enumerate(net.modes):
        phys = dict(zip(net.physical[s], occupation[s]))
        for local in range(m):
            if local not in phys or phys[local]:
                keep.append(net.global_mode(s, local))
    return keep


BOND_SIGN = 1  # M[p_a, q_b] = BOND_SIGN * sigma, in the bond's orientation


def assemble(net, z_sites, occupation, dtype=float):
    """`(M, keep)`: the skew matrix over the kept modes, entries copied, never divided."""
    total = int(net.offsets[-1])
    full = np.zeros((total, total), dtype=dtype)
    for s, z in enumerate(z_sites):
        o = net.offsets[s]
        full[o:o+len(z), o:o+len(z)] = z
    for p, q, sigma in net.bond_pairs():
        full[p, q] += BOND_SIGN*sigma
        full[q, p] -= BOND_SIGN*sigma
    keep = kept_modes(net, occupation)
    return full[np.ix_(keep, keep)], keep


def configuration_sign(net, occupation):
    """`epsilon(n)`: psi(n) = epsilon(n) 2^{-b/2} Pf(M).

    Expanding every bond operator, `2^{b/2} psi = sum_K sigma_K chi(K) Pf(Z_{T_K})`
    over the sets K of bonds whose pair is occupied, `chi(K)` the sign of putting
    `c_{n_k}..c_{n_1} prod_{b in K} c_q c_p` in descending order (which
    `<0|c_{t_m}..c_{t_1}|Phi> = Pf Z_T` needs). `Pf M` expands into the same
    terms up to one K-independent factor, read at K = all bonds: `prod sigma`
    times chi(all).
    """
    occupied = sorted(net.global_mode(s, l) for s in range(len(net.modes))
                      for l, bit in zip(net.physical[s], occupation[s]) if bit)
    sequence = occupied[::-1]
    sign = 1
    for sa, ma, sb, mb, sigma in net.bonds:
        sequence += [net.global_mode(sb, mb), net.global_mode(sa, ma)]
        sign *= sigma
    # inversions relative to descending order
    inversions = sum(1 for i in range(len(sequence)) for j in range(i+1, len(sequence))
                     if sequence[i] < sequence[j])
    return sign*(-1)**inversions


def log_scale(net):
    """`log 2^{-b/2}`, the bonds' normalisation."""
    return -0.5*len(net.bonds)*np.log(2.0)



def _contraction_case(L=4, nv=1, n=6, seed=0):
    """A small torus' contraction `f(free, bits)` and random inputs: site entries N(0, 0.5^2) (float64) and
    half-filled configurations as per-site bits; with the network for the dense reference."""
    import jax
    from nngtn.state import _contraction, config_of_spins
    from nngtn.pfaffian import tree as lay
    ng.make_sites(L, L, (L * L // 2, L * L // 2))
    f = _contraction(L, L, nv, 2, 0.05, 20, 2)
    m = 2 + 4 * nv
    rng = np.random.default_rng(seed)
    free = 0.5 * rng.normal(size=(n, L * L, m * (m - 1) // 2))
    spins = _configs(L, n, seed)
    config = np.asarray(jax.vmap(lambda x: config_of_spins(x, L * L))(spins))
    bits = np.stack([(config >> 1) & 1, config & 1], axis=-1).reshape(n, -1).astype(np.int32)
    net = lay.network_of_layout(lay.peps_layout(L, L, 2, nv, nv, True))
    return f, free, bits, net, m


def _dense_reference(net, m, free, bits):
    """`(sign, log|psi|)` of one configuration: psi = epsilon(n) 2^(-b/2) Pf(M), M assembled densely from the site
    entries and the bonds, its Pfaffian by completely pivoted Parlett-Reid in float64."""
    iu = np.triu_indices(m, 1)
    z_sites = []
    for fv in free:
        z = np.zeros((m, m))
        z[iu] = fv
        z_sites.append(z - z.T)
    occupation = [tuple(int(b) for b in bits[2 * s:2 * s + 2]) for s in range(len(free))]
    M, _ = assemble(net, z_sites, occupation)
    sign, logabs = _dense_pfaffian(M)
    return sign * configuration_sign(net, occupation), logabs + log_scale(net)


def _check_against_dense(dtype, tol, global_from=None, dmma=False, L=4, nv=1, n=6, env=None):
    import jax
    import jax.numpy as jnp
    from nngtn.pfaffian import recipe
    if global_from is not None:
        os.environ["PIVOTED_GLOBAL_FROM"] = str(global_from)     # every level at least this wide on the global kernels
    os.environ.update(env or {})                                 # other path knobs (read once a process)
    recipe.apply(dtype, dmma=dmma)
    f, free, bits, net, m = _contraction_case(L=L, nv=nv, n=n)
    sign, logabs = jax.vmap(lambda x, o: f(x, o)[:2])(jnp.asarray(free, dtype), jnp.asarray(bits))
    for b in range(len(free)):
        rs, rl = _dense_reference(net, m, free[b].astype(dtype).astype(np.float64), bits[b])
        assert float(sign[b]) == rs, (b, float(sign[b]), rs)
        assert abs(float(logabs[b]) - rl) < tol, (b, float(logabs[b]), rl)


@gpu
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-9), ("float32", 2e-3)])
def test_contraction_matches_the_dense_pfaffian(dtype, tol, request):
    """The GPU contraction (shared-memory kernels) of random site entries on a 4x4 torus against the dense
    float64 Pfaffian of the same network matrix: the same sign, log|psi| to round-off."""
    if _isolate(request):
        return
    _check_against_dense(dtype, tol)


@gpu
@pytest.mark.parametrize("dtype,dmma,tol", [("float64", False, 1e-9), ("float32", False, 2e-3), ("float32", True, 2e-3)])
def test_global_kernels_match_the_dense_pfaffian(dtype, dmma, tol, request):
    """Every front forced onto the global-memory kernels (`PIVOTED_GLOBAL_FROM=1`), the float32 close with and
    without DMMA: the same dense-Pfaffian check."""
    if _isolate(request):
        return
    _check_against_dense(dtype, tol, global_from=1, dmma=dmma)



# the kernel paths the 4x4, nv = 1 case never reaches (its fronts are 18 to 52 wide, all on the warp kernel): an 8x8
# torus at the production Nv = 3 has levels 100 and 112 wide (the block kernel), and two knobs force the rest. Each
# knob was checked to change the float32 result, so the path it names is the one that runs.
_PATHS = {"8x8-nv3": dict(L=8, nv=3, n=3), "block": dict(env={"PIVOTED_WARP_MAX": "0"}),
          "8x8-nv3-block": dict(L=8, nv=3, n=3, env={"PIVOTED_WARP_MAX": "0"})}


@gpu
@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-9), ("float32", 2e-3)])   # measured: 2e-13, 6e-5
def test_block_kernel_and_production_widths_match_the_dense_pfaffian(path, dtype, tol, request):
    """The block kernel (fronts wider than 96, or every front with `PIVOTED_WARP_MAX=0`) and the production Nv = 3
    widths of an 8x8 torus against the dense float64 Pfaffian: the same sign, log|psi| to round-off."""
    if _isolate(request):
        return
    _check_against_dense(dtype, tol, **_PATHS[path])

def _gradient_errors(dtype, global_from=None, n_entries=12, L=4, nv=1, env=None):
    """`(relative errors, entries)`: d log|psi| / d(site entries) from the GPU backward pass at `dtype` against
    central finite differences of the float64 contraction at the same (dtype-rounded) entries."""
    import jax
    import jax.numpy as jnp
    from nngtn.pfaffian import recipe
    if global_from is not None:
        os.environ["PIVOTED_GLOBAL_FROM"] = str(global_from)
    os.environ.update(env or {})
    recipe.apply(dtype)
    f, free, bits, net, m = _contraction_case(L=L, nv=nv, n=1)
    o = jnp.asarray(bits[0])
    x = jnp.asarray(free[0], dtype)
    g = np.asarray(jax.grad(lambda v: f(v, o)[1])(x), np.float64)
    x64 = np.asarray(x, np.float64)
    logabs64 = jax.jit(lambda v: f(v, o)[1])
    rng = np.random.default_rng(3)
    h = 1e-5
    errs = []
    for _ in range(n_entries):
        i, k = int(rng.integers(x64.shape[0])), int(rng.integers(x64.shape[1]))
        e = np.zeros(x64.shape)
        e[i, k] = h
        fd = (float(logabs64(jnp.asarray(x64 + e))) - float(logabs64(jnp.asarray(x64 - e)))) / (2 * h)
        errs.append(abs(g[i, k] - fd) / max(1.0, abs(fd)))
    return np.array(errs)


@gpu
@pytest.mark.parametrize("path", ["shared", "global"])
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-6), ("float32", 1e-4)])   # measured: 1e-8, 7e-6
def test_contraction_gradient_matches_finite_differences(path, dtype, tol, request):
    """d log|psi| / d(site entries) from the GPU backward kernels against central finite differences of the float64
    contraction, on the shared-memory kernels and with every front on the global-memory kernels
    (`PIVOTED_GLOBAL_FROM=1`), float64 and float32."""
    if _isolate(request):
        return
    errs = _gradient_errors(dtype, global_from=1 if path == "global" else None)
    print(f"{path} {dtype}: max relative error {errs.max():.2e}, median {np.median(errs):.2e}")
    assert errs.max() < tol, errs



_GRAD_PATHS = {"block": dict(env={"PIVOTED_WARP_MAX": "0"}), "global-backward": dict(env={"PIVOTED_GLOBAL_BWD_FROM": "1"}),
               "8x8-nv3": dict(L=8, nv=3)}


@gpu
@pytest.mark.parametrize("path", sorted(_GRAD_PATHS))
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-6), ("float32", 1e-4)])   # measured: 1e-8, 2e-5
def test_block_and_global_backward_gradients_match_finite_differences(path, dtype, tol, request):
    """The backward of the block kernel (`PIVOTED_WARP_MAX=0`), of the global-memory backward behind the
    shared-memory forward (`PIVOTED_GLOBAL_BWD_FROM=1`, the production hybrid's backward from width 176), and at the
    production Nv = 3 widths of an 8x8 torus, against central finite differences of the float64 contraction."""
    if _isolate(request):
        return
    errs = _gradient_errors(dtype, **_GRAD_PATHS[path])
    print(f"{path} {dtype}: max relative error {errs.max():.2e}, median {np.median(errs):.2e}")
    assert errs.max() < tol, errs


@gpu
@pytest.mark.parametrize("Lx,Ly,symm", [(3, 3, None), (3, 3, "flip0@flip1@spininverse"), (4, 2, "flip0@flip1@spininverse")])
def test_sampled_energy_matches_exact_enumeration(Lx, Ly, symm, tmp_path, request):
    """`measure.measure`'s sampled energy (32 sweeps of 1024 samples, jackknife error) against the exact energy of the
    same state, sum |psi|^2 E_loc / sum |psi|^2 over every configuration of the (4, 4) sector, unprojected and
    projected (measured: within 0.85 to 1.2 standard errors). Every non-finite amplitude must be an exact zero (the
    projection's images cancelling), never an overflow, and every finite amplitude must have a finite E_loc."""
    if _isolate(request):
        return
    import itertools
    import jax.numpy as jnp
    from quantax.sampler import Samples
    from nngtn import measure
    measure.set_matmul_precision("highest")
    N, nup, ndn = Lx * Ly, 4, 4
    ng.make_sites(Lx, Ly, (nup, ndn))
    model = ng.build_model(Lx, Ly, nv=1, nblocks=2, channels=16, pad_channels=0, res_scale=0.3, head_scale=3.0,
                           delay_slots=8, net_seed=4)
    state = ng.make_state(model, symm=symm, max_parallel=512)
    H = ng.hamiltonian(Lx, Ly, t=1.0, u=8.0, tprime=0.0)
    spins = []
    for up in itertools.combinations(range(N), nup):
        for dn in itertools.combinations(range(N), ndn):
            s = -np.ones(2 * N, np.int8)
            s[list(up)] = 1
            s[[N + k for k in dn]] = 1
            spins.append(s)
    spins = np.array(spins)
    logabs, eloc = [], []
    for i in range(0, len(spins), 2048):
        s = jnp.asarray(spins[i:i + 2048])
        psi = state(s)
        logabs.append(np.asarray(psi.logabs, np.float64))
        eloc.append(np.asarray(jnp.real(H.Oloc(state, Samples(s, psi=psi))), np.float64))
    logabs, eloc = np.concatenate(logabs), np.concatenate(eloc)
    assert not np.isnan(logabs).any()                                   # no overflow
    live = np.isfinite(logabs)
    assert np.isneginf(logabs[~live]).all() and np.isfinite(eloc[live]).all()
    w = np.exp(2 * (logabs[live] - logabs[live].max()))
    exact = float(np.sum(w * eloc[live]) / np.sum(w)) / N
    res = measure.measure(state, H, str(tmp_path / "m"), nparticles=(nup, ndn), nsamples=1024, nsweeps=32,
                          thermal=200 * N, sweeps=8 * N, seed=3, channels=("energy",), save_every=32)
    print(f"{Lx}x{Ly} symm={symm}: exact E/N {exact:.6f}, sampled {res['E_per_site']:.6f} +- {res['E_err']:.6f}")
    assert res["E_err"] > 0 and abs(res["E_per_site"] - exact) < 4 * res["E_err"]

@gpu
def test_hamiltonian_matches_quantax_hubbard(request):
    """`hamiltonian(t' = -0.2, U = 8)` against quantax's own `Hubbard(t=[1, t'], n_neighbor=[1, 2])` (bonds built
    independently): the same local energies of a state on a 4x4 torus held wholly in float64 (quantax's dtype, the
    network and the contraction), to float64 round-off — any term that differs shows."""
    if _isolate(request):
        return
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.sampler import Samples
    ng.make_sites(4, 4, (8, 8), dtype=jnp.float64)            # psi and the local energies in 64 bits
    model = ng.build_model(4, 4, nblocks=2, channels=16, pad_channels=0, contraction_dtype="float64",
                           dtype=jnp.float64)
    state = ng.make_state(model)
    s = jnp.asarray(_configs(4, 16, seed=5))
    smp = Samples(s, psi=state(s))
    ours = np.asarray(ng.hamiltonian(4, 4, t=1.0, u=8.0, tprime=-0.2).Oloc(state, smp))
    ref = np.asarray(qtx.operator.Hubbard(U=8.0, t=[1.0, -0.2], n_neighbor=[1, 2]).Oloc(state, smp))
    assert ours.dtype == np.float64 and np.isfinite(ours).all()
    # the two operators sum their terms in different orders: measured 3e-15 relative, with or without
    # --xla_gpu_deterministic_ops
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))


def test_at_precision_dual_records_the_sampler_estimator():
    """`eloc_dual`: the sampler's own estimator is read on the same samples first (psi kept), `r` is put back,
    and the re-evaluation that follows is the plain `eloc_precision` one — its output and the samples untouched."""
    import jax.numpy as jnp
    from nngtn.vmc import at_precision

    class FakeH:
        def __init__(self):
            self.psi_seen = []

        def Oloc(self, state, samples):
            self.psi_seen.append(samples.psi)
            samples.r = jnp.array([1.0, 1.0, 2.0])          # quantax fills |psi|^2 / prob in place
            if samples.psi is not None:                      # the sampler's precision: one non-finite term
                return jnp.array([1.0, 3.0, jnp.inf])
            return jnp.array([1.5, 2.5, 3.5])                 # the re-evaluation

    class FakeSamples:
        psi = "sampler psi"
        r = None

    plain = at_precision(FakeH(), "highest")
    s_plain = FakeSamples()
    out_plain = plain.Oloc(None, s_plain)
    assert plain.psi_seen == [None] and out_plain.tolist() == [1.5, 2.5, 3.5]

    dual = {}
    h = at_precision(FakeH(), "highest", dual=dual)
    s = FakeSamples()
    out = h.Oloc(None, s)
    assert out.tolist() == out_plain.tolist()
    assert h.psi_seen == ["sampler psi", None]
    assert float(dual["e"]) == 2.0 and float(dual["var"]) == 1.0 and int(dual["nonfinite"]) == 1
    assert abs(float(dual["ess"]) - (4 / 3) ** 2 / 2) < 1e-12
    untouched = FakeH()
    assert at_precision(untouched, None, dual={}) is untouched           # no precision: H returned unchanged


def test_eloc_dual_requires_eloc_precision():
    from nngtn.vmc import run
    with pytest.raises(ValueError, match="eloc_dual"):
        run(None, n_particles=(1, 1), eloc_dual=True)


def test_reread_energy_replaces_restored_and_puts_back_excluded(monkeypatch):
    """A sweep whose flagged samples are of both kinds: one restored in the first pass (finite psi-tilde energy, in
    the sweep's means) and one still excluded. The re-read replaces the first's value and puts the second back;
    neither is counted twice (CPU: the energy channel only, the local energies stubbed)."""
    from nngtn import measure as meas
    nsamples, others = 10, [-1.0] * 8
    restored = (-3.0, -3.5)                           # (measured, more slots)
    excluded = (float("nan"), -4.0)
    first = np.asarray(others + [restored[0]])      # the 9 samples the first pass kept
    series = dict(Lx=np.asarray(2), Ly=np.asarray(2), nsamples=np.asarray(nsamples), kept=np.asarray([0]),
                  E_mean=np.asarray([first.mean()]), E_sqr=np.asarray([(first ** 2).mean()]),
                  E_excluded=np.asarray([1]),
                  bad_spins=np.asarray([[1, -1, -1, -1, 1, -1, -1, -1], [-1, 1, -1, -1, -1, 1, -1, -1]], np.int8),
                  bad_meta=np.asarray([[0, 1, 0, 0], [0, 1, 0, 0]], np.int32))

    def energy(state, H, s):
        row = restored if int(np.asarray(s)[0, 0]) > 0 else excluded
        return row[0] if state == "measured" else row[1]

    monkeypatch.setattr(meas, "psitilde_local_energy", energy)
    out, _ = meas.reread("measured", "more", None, series, log=lambda msg: None)
    final = np.asarray(others + [restored[1], excluded[1]])
    assert np.isclose(out["E_mean"][0], final.mean()) and np.isclose(out["E_sqr"][0], (final ** 2).mean())
    assert int(out["E_excluded"][0]) == 0


def _fock_amplitudes(Lx, Ly, nv, Zs):
    """The Gaussian PEPS by brute force, from its definition alone: `<n| prod_bonds <0_pq|(1 + c_q c_p)/sqrt2
    prod_s exp(1/2 c+ Z_s c+)|0>` in a Jordan-Wigner basis, each site state by matrix exponential (no Pfaffian
    identity), the bonds applied operator by operator and their modes dropped once in the vacuum. Returns
    `{occupied physical modes in global order: amplitude}` and the physical modes `[[up, dn] per site]`. Only the
    layout's facts are read from the package: the local mode order and which leg modes a bond joins."""
    from scipy.linalg import expm
    from nngtn.pfaffian import tree as lay
    N, m = Lx * Ly, 2 + 4 * nv
    phys, ends = [], {}
    for s, (i, j) in enumerate((i, j) for i in range(Lx) for j in range(Ly)):
        sizes, names = lay.site_modes(i, j, Lx, Ly, 2, nv, nv, periodic=True)
        inds, off = lay.bond_names(i, j, Lx, Ly, periodic=True), 0
        for size, name in zip(sizes, names):
            if name == "phys":
                phys.append([s * m + off, s * m + off + 1])
            else:
                ends.setdefault(inds[name], []).append((s, name, off, size))
            off += size
    bonds = []
    for (sa, na, oa, w), (sb, nb, ob, _) in ends.values():
        if not lay.KET_SIDE[na]:
            (sa, oa), (sb, ob) = (sb, ob), (sa, oa)
        bonds += [(sa * m + oa + t, sb * m + ob + t) for t in range(w)]
    dim = 2 ** m
    cdag = []                                    # mode a is bit (m - 1 - a); c+_a carries (-1)^(occupied before a)
    for a in range(m):
        M = np.zeros((dim, dim))
        for st in range(dim):
            if not (st >> (m - 1 - a)) & 1:
                M[st | (1 << (m - 1 - a)), st] = (-1) ** bin(st >> (m - a)).count("1")
        cdag.append(M)
    live, vec, pending = [], np.ones(1), list(bonds)
    for s in range(N):
        A = sum(Zs[s][a, b] * cdag[a] @ cdag[b] for a in range(m) for b in range(a + 1, m))
        vec = np.kron(vec, expm(A)[:, 0])
        live += list(range(s * m, (s + 1) * m))
        for p, q in [b for b in pending if b[0] in live and b[1] in live]:
            pending.remove((p, q))
            L, ip, iq = len(live), live.index(p), live.index(q)
            idx = np.arange(2 ** L, dtype=np.uint64)
            bit = lambda k, x=idx: (x >> np.uint64(L - 1 - k)) & np.uint64(1)          # noqa: E731
            before = lambda k, x: np.bitwise_count(x >> np.uint64(L - k)) if k else 0 * x  # noqa: E731
            keep = (bit(ip) == 0) & (bit(iq) == 0)
            out = np.where(keep, vec, 0.0)
            both = idx[(bit(ip) == 1) & (bit(iq) == 1)]                # (1 + c_q c_p): c_p first, then c_q
            s1 = 1 - 2 * (before(ip, both).astype(np.int64) & 1)
            one = both & ~np.uint64(1 << (L - 1 - ip))
            s2 = 1 - 2 * (before(iq, one).astype(np.int64) & 1)
            np.add.at(out, (one & ~np.uint64(1 << (L - 1 - iq))).astype(np.int64), s1 * s2 * vec[both.astype(np.int64)])
            rest = [k for k in range(L) if k not in (ip, iq)]
            sel = idx[keep]
            new = np.zeros(len(sel), np.uint64)
            for pos, k in enumerate(rest):
                new |= ((sel >> np.uint64(L - 1 - k)) & np.uint64(1)) << np.uint64(len(rest) - 1 - pos)
            vec = np.zeros(2 ** len(rest))
            vec[new.astype(np.int64)] = out[sel.astype(np.int64)] / np.sqrt(2)
            live = [live[k] for k in rest]
    assert not pending
    L = len(live)
    return {tuple(live[k] for k in range(L) if (st >> (L - 1 - k)) & 1): vec[st] for st in range(2 ** L)}, phys


@gpu
@pytest.mark.parametrize("Lx,Ly,nv,nup,ndn", [(2, 2, 1, 2, 2), (3, 2, 1, 3, 2), (2, 3, 1, 2, 3), (2, 2, 2, 2, 1)])
def test_amplitude_is_the_gaussian_peps_by_brute_force(Lx, Ly, nv, nup, ndn, request):
    """Every configuration's signed amplitude (float64 contraction, random site matrices) equals the Gaussian PEPS
    built in Fock space from its definition, in the configuration's own mode order (spin-up block first, site
    i * Ly + j): sign, magnitude and the 2^(-b/2) normalisation (measured: 1e-14 relative)."""
    if _isolate(request):
        return
    import itertools
    import jax
    import jax.numpy as jnp
    from nngtn.state import config_of_spins, polar_pair
    N, m = Lx * Ly, 2 + 4 * nv
    rng = np.random.default_rng(3)
    iu = np.triu_indices(m, 1)
    free = rng.normal(0, 0.7, (N, len(iu[0])))
    Zs = []
    for f in free:
        Z = np.zeros((m, m))
        Z[iu] = f
        Zs.append(Z - Z.T)
    ref, phys = _fock_amplitudes(Lx, Ly, nv, Zs)
    ng.make_sites(Lx, Ly, (nup, ndn))
    model = ng.build_model(Lx, Ly, nv=nv, nblocks=1, channels=4, pad_channels=0, res_scale=0.1,
                           contraction_dtype="float64", delay_slots=8)
    spins = []
    for up in itertools.combinations(range(N), nup):
        for dn in itertools.combinations(range(N), ndn):
            s = -np.ones(2 * N, np.int8)
            s[list(up)] = 1
            s[[N + k for k in dn]] = 1
            spins.append(s)
    spins = np.array(spins)
    psi = jax.vmap(lambda s: polar_pair(model.log_psi(config_of_spins(s, N, 2), jnp.asarray(free)), s))(
        jnp.asarray(spins))
    got = np.asarray(psi.sign) * np.exp(np.asarray(psi.logabs, np.float64))
    want = []
    for s in spins:
        order = [phys[k][0] for k in range(N) if s[k] > 0] + [phys[k][1] for k in range(N) if s[N + k] > 0]
        inversions = sum(order[a] > order[b] for a in range(len(order)) for b in range(a + 1, len(order)))
        want.append(ref.get(tuple(sorted(order)), 0.0) * (-1) ** inversions)
    want = np.array(want)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


def test_psitilde_excludes_a_sample_whose_own_amplitude_overflows():
    """psi-tilde(s) = 0 when s's own contraction overflows: its local energy is NaN (excluded and counted as
    "own"), not the diagonal term alone with every connected term zeroed (CPU: a stub state, the Hamiltonian
    never read)."""
    import jax.numpy as jnp
    from quantax.utils import LogArray
    from nngtn.measure import psitilde_local_energy

    def overflowed(s):
        return LogArray(jnp.ones(s.shape[0]), jnp.full(s.shape[0], jnp.nan))

    stats = {}
    e = psitilde_local_energy(overflowed, None, jnp.ones((1, 8), jnp.int8), stats)
    assert np.isnan(e) and stats == {"own": 1}



def test_float32_network_stays_float32_under_x64():
    """jax_enable_x64 only allows float64 arrays and makes them the default for NEW ones; a float32 operation turns
    float64 only when a strong float64 array enters it. The production Nv = 3 network, forward and backward as
    compiled, must hold every convolution in float32 operands, at DEFAULT precision (TF32 on the GPU) by default
    and HIGHEST under "highest" — the matmul precision sets that attribute and never a dtype."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    assert jax.config.jax_enable_x64
    ng.make_sites(4, 4, (8, 8))
    model = ng.build_model(4, 4, nv=3)
    leaves = [x for x in jax.tree_util.tree_leaves(model) if hasattr(x, "dtype")]
    assert leaves and all(x.dtype == jnp.float32 for x in leaves)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    s = jnp.asarray(_configs(4, 2))
    grad = jax.grad(lambda p, x: jnp.sum(jax.vmap(eqx.combine(p, static).free_vectors)(x)))
    for precision, attribute in (("default", "DEFAULT"), ("highest", "HIGHEST")):
        with jax.default_matmul_precision(precision):
            text = jax.jit(grad).lower(params, s).as_text()
        convs = [line for line in text.splitlines() if "stablehlo.convolution" in line]
        assert len(convs) >= 3 * 16, len(convs)                  # forward + backward of 16 blocks' convolutions
        assert all("xf32>" in c and "xf64>" not in c for c in convs), [c for c in convs if "xf64>" in c][:1]
        assert all(attribute in c for c in convs), precision
    assert jax.vmap(model.free_vectors)(s).dtype == jnp.float32


@pytest.mark.parametrize("c,p", [(13, 16), (91, 92), (153, 160)])           # a small width and both presets'
def test_padded_trunk_is_the_unpadded_trunk(c, p):
    """The presets run every convolution at `pad_channels` (`ResConvTrunk._padded`: zero-padded weights, biases and
    feature map, the RMSNorm over the real elements only); with the same parameters it is the same function as
    quantax's own blocks (`pad_channels=None`): outputs and parameter gradients, in float64, to round-off. Every
    parameter, biases included, is perturbed away from its initial value first."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import quantax as qtx
    from nngtn.state import ResConvTrunk, scale_residual_branches
    ng.make_sites(4, 4, (8, 8))

    def trunk(pad):
        qtx.set_random_seed(7)
        net = scale_residual_branches(ResConvTrunk(3, c, (3, 3), dtype=jnp.float64, pad_channels=pad), 0.5)
        params, static = eqx.partition(net, eqx.is_inexact_array)
        leaves, tdef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
        leaves = [x + 0.3 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
        return jax.tree_util.tree_unflatten(tdef, leaves), static

    (p0, s0), (p1, s1) = trunk(None), trunk(p)
    l0, l1 = jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(p1)
    assert len(l0) == len(l1) and all(np.array_equal(a, b) for a, b in zip(l0, l1))     # the same parameters
    assert sum(x.size for x in l0) == sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(ResConvTrunk(3, c, (3, 3), dtype=jnp.float64), eqx.is_inexact_array)))  # stored c wide, not p
    s = jnp.asarray(_configs(4, 3))

    def out(params, static):
        return jax.vmap(eqx.combine(params, static))(s)

    def loss(params, static):
        return jnp.sum(jnp.sin(out(params, static)))

    with jax.default_matmul_precision("highest"):
        y0, y1 = out(p0, s0), out(p1, s1)
        g0, g1 = jax.grad(loss)(p0, s0), jax.grad(loss)(p1, s1)
    assert y0.shape == y1.shape == (3, c, 4, 4) and y0.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(y1), np.asarray(y0), rtol=1e-11, atol=1e-12)
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-10, atol=1e-11)

def test_config_rejects_unknown_keys(tmp_path):
    """A typo or an option the package does not have stops the load instead of being ignored."""
    text = (ROOT / "examples" / "hubbard_4x4.toml").read_text()
    for bad in (text.replace("steps = 100", "steps = 100\nlsmr_rtol = 1e-3", 1),
                text.replace("delay_slots = 20", "delay_slots = 20\nleaf_sites = 3", 1),
                text.replace("cap = { c0 = 0.5,", "cap = { c00 = 0.5,", 1)):
        path = tmp_path / "bad.toml"
        path.write_text(bad)
        with pytest.raises(ValueError, match="unknown keys"):
            nio.load_config(path)


def test_a_non_finite_step_stops_training():
    import jax.numpy as jnp
    from nngtn.vmc import scale_step
    with pytest.raises(FloatingPointError):
        scale_step(jnp.array([1.0, jnp.nan]), 0)
    assert float(jnp.linalg.norm(scale_step(jnp.array([3.0, 4.0]), 0, c0=0.5))) == pytest.approx(0.5)
