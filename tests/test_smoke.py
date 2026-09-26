"""
Smoke tests on tiny FAKE data (tests/fake_data.py), no real dataset needed, under a minute on CPU.

    python3 -m pytest -q

1. the library: loaders on every layout, split, POD, NS operators (sliced = rebuilt), RK4,
   screening, the arms, the training loop, the CSV helpers;
2. every study script in its dry-run mode (--plan / --dry-run / --assess-only / --epochs 1),
   run in-process on the fake data with ROM_DATA_DIR / ROM_MODELS_DIR / ROM_RESULTS_DIR.
"""
import os, sys, csv, runpy, tempfile, importlib, atexit, shutil
import numpy as np
import pytest

# ---- fake folders, set BEFORE rom is imported (rom.paths reads the variables once) ----
_TMP = tempfile.mkdtemp(prefix="rom_smoke_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
DATA, MODELS, RESULTS = (os.path.join(_TMP, d) for d in ("DATA", "bestModels", "convergence"))
os.environ.update(ROM_DATA_DIR=DATA, ROM_MODELS_DIR=MODELS, ROM_RESULTS_DIR=RESULTS, MPLBACKEND="Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
import fake_data                                           # noqa: E402

fake_data.make(DATA, nt_alpha0=300, nt_re100=300, nt_channel=60)
os.makedirs(MODELS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

from rom import paths, data, split, pod, vae, augment, results      # noqa: E402
from rom import galerkin_ns as gns, galerkin_data as gd              # noqa: E402

RE100 = os.path.join(DATA, "2PlatesGap", "Data2PlatesGap1Re100.mat")
RE50 = os.path.join(DATA, "Alpha0", "dataRe50Alpha0_2.mat")
CHAN = os.path.join(DATA, "CHANNEL", "channel_fake_UVW.mat")


# ------------------------------------------------------------------ library

def test_paths_follow_env():
    assert (paths.DATA, paths.MODELS, paths.RESULTS) == (DATA, MODELS, RESULTS)
    assert paths.AUGMENTED == os.path.join(DATA, "AUGMENTED")


def test_loaders_every_layout():
    d, names = data.load_data(RE50, None, 1, 120)
    assert d.shape == (120, 2, 16, 24) and names == ["u", "v"] and d.dtype == np.float32
    d, _ = data.load_data(RE100)
    assert d.shape == (300, 2, 12, 20)
    d, names = data.load_data(CHAN)
    assert d.shape == (60, 2, 16, 16) and names == ["u", "w"]
    for name in data.DATASETS:
        d, re_, dt, path = data.load(name)
        assert d.shape[1] == 2 and re_ == float(name[2:])
    assert gns.grid_spacing(RE50) == (0.25, 0.25)
    d, re_, dt, path = data.load_channel("chan")
    assert d.shape == (200, 2, 16, 32) and re_ == 20000 and abs(dt - 0.1) < 1e-12
    dx, dy = gns.grid_spacing(path)
    assert abs(dx - 8 * np.pi / 32) < 1e-12 and abs(dy - 2 / 15) < 1e-12


def test_split_and_draws():
    val, pool, n_val = split.split_indices(1000, None, "random", split.SPLIT_SEED)
    assert n_val == 100 and len(pool) == 900 and not set(val) & set(pool)
    val, pool, n_val = split.split_indices(1000, None, "tail5", 7)
    assert list(val) == list(range(950, 1000))
    rng = np.random.default_rng([split.DRAW_SEED, 50, 40, 0])
    idx = split.draw_blocks(set(int(i) for i in pool), 1000, 40, rng)
    assert len(idx) == 40 and set(idx) <= set(pool)
    assert len(split.draw_subset("contig", set(pool), pool, 1000, 25, rng)) == 25
    assert len(split.draw_pool_subset(900, 60, "blocks", np.random.default_rng(11))) == 60


def test_pod_ns_operators_and_arms():
    d, re_, dt, path = data.load("Re50")
    val, pool, _ = split.split_indices(len(d), None, "random", 7)
    tr = np.sort(pool[:60])
    P = pod.subset_pod(d[tr])
    K = pod.k_for(P, 0.99)
    rec = pod.reconstruct(P["x_mean"], P["Xc"], P["Wp"], P["A"], P["shape"])
    assert np.allclose(rec, d[tr], atol=1e-3)                      # full POD rebuilds the snapshots
    ceiling, _ = pod.ceiling_and_project(P, K, d[val], [])
    assert 50 < ceiling <= 100
    # operators built once at K+2 and sliced == built at K
    l8, q8, kap = gns.build_ops(P, K + 2, re_, path)
    lK, qK, _ = gns.build_ops(P, K, re_, path)
    assert np.allclose(l8[:K, :K + 1], lK) and np.allclose(q8[:K, :K + 1, :K + 1], qK)
    A = np.asarray(P["A"][:, :K], dtype=np.float64)
    step, s, to_b, from_b = gns.model_at(l8, q8, kap, A, K, dt, augment.SUBSTEPS)
    assert np.all(np.isfinite(step(to_b(A[0]))))
    aug, st = augment.synthetic_arm(step, s, to_b, from_b, P, A, K, 20, 5, np.random.default_rng(0),
                                    max_traj=4000)
    assert aug.shape == (20, 2, 16, 24) and 0 < st["keep_frac"] <= 1
    assert augment.jitter_arm(P, A, K, 10, np.random.default_rng(1)).shape == (10, 2, 16, 24)
    got, take = augment.real_arm(P, K, d, pool, tr, 10, np.random.default_rng(2))
    assert got.shape == (10, 2, 16, 24) and take == 10
    # data-identified model, fitted on 200 consecutive snapshots (central differences need them)
    Pc = pod.subset_pod(d[:200])
    Ac = np.asarray(Pc["A"][:, :4], dtype=np.float64)
    beta, s2, R2, n_fit = gd.fit_galerkin(Ac, np.arange(200), dt, 1e-6)
    assert n_fit == 198 and np.all(R2 > 0.9)
    assert np.all(np.isfinite(gd.make_step(beta, "ode", dt, 20)(Ac[0] / s2)))
    assert augment.n_aug_for(0.5, 25) == 25 and augment.n_aug_for(0.8, 25) == 100


def test_training_loop_runs():
    d, _ = data.load_data(RE100)
    ek, detR, vl, _ = vae.train(d[:40], d[40:50], d[250:270], tag="smoke", latent=3, epochs=1)
    assert np.isfinite(ek) and np.isfinite(vl) and 0 <= detR <= 1


def test_results_helpers(tmp_path):
    p = str(tmp_path / "x.csv")
    results.append_row(p, ["a", "b"], {"a": 1})
    results.append_row(p, ["a", "b"], {"a": 2, "b": 3})
    assert results.done_rows(p, lambda r: r["a"]) == {"1", "2"}
    results.rewrite_with_row(p, ["a", "b"], {"b": 4})
    assert [r["b"] for r in csv.DictReader(open(p))] == ["", "3", "4"]
    q = str(tmp_path / "last.csv")
    results.update_row(q, "ds1", {"Re": 50, "NN_Ek": 90.0})
    results.update_row(q, "ds1", {"POD95_n": 25})
    (row,) = list(csv.DictReader(open(q)))
    assert row["NN_Ek"] == "90.0" and row["POD95_n"] == "25"


# ------------------------------------------------------------------ every study, dry-run

def run_script(rel, *args):
    """Run a study script in-process, as `python3 <rel> <args>` would."""
    old = sys.argv
    sys.argv = [os.path.join(REPO, rel)] + [str(a) for a in args]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        sys.argv = old


S = "studies/"
DRY_RUNS = [
    (S + "1_vae_vs_pod/pod.py", "--data", CHAN),
    (S + "1_vae_vs_pod/beta_vae.py", "--data", CHAN, "--epochs", 1),
    (S + "2_convergence/assess_pod_modes.py", "tail5"),
    (S + "2_convergence/pod_convergence.py", RE100, 1500, 5, 95, "tail5"),
    (S + "2_convergence/nn_convergence.py", RE100, 1500, 3, "tail5", "--epochs", 1),
    (S + "2_convergence/make_alpha0_excel.py", "tail5"),
    (S + "2_convergence/nn_latent_test.py", "tail5", "Re100=3", "--epochs", 1),
    (S + "2_convergence/run_alpha0_convergence.py", "--plan"),
    (S + "3_re100_fraction/re100_fraction_sweep.py", "galerkin_ns", "--fractions", "0,0.25", "--dry-run"),
    (S + "3_re100_fraction/re100_conv_fraction_sweep.py", "galerkin_ns", "--modes", 6, "--assess-only"),
    (S + "3_re100_fraction/run_re100_grid.py", "--modes", 6, "--tols", "0.05", "--plan"),
    (S + "4_region_map/run_region_map.py", "--plan"),
    (S + "4_region_map/run_region_map.py", "--datasets", "Re100", "--n", 20, "--subsets", 1, "--epochs", 1),
    (S + "4_region_map/run_round1.py", "--plan"),
    (S + "4_region_map/run_round2.py", "--plan"),
    (S + "5_low_data/run_lowdata.py", "--plan"),
    (S + "5_low_data/run_lowdata.py", "--stage", "A", "--datasets", "Re50", "--n", 25, "--subsets", 1),
    (S + "5_low_data/run_lowdata_subsets.py", "--plan"),
    (S + "6_capacity/run_capacity.py", "--plan"),
    (S + "7_channel/run_channel.py", "--stage", "A", "--n", 20, "--subsets", 1),
    (S + "7_channel/run_channel.py", "--stage", "B", "--n", 20, "--subsets", 1, "--epochs", 1),
]


@pytest.mark.parametrize("case", DRY_RUNS, ids=lambda c: " ".join(map(str, (os.path.basename(c[0]),) + c[1:3])))
def test_study_dry_run(case):
    run_script(*case)


def test_analysis_after_pod_and_vae():
    """analysis.py needs the two files the first two dry runs wrote."""
    pt = [f for f in os.listdir(MODELS) if f.startswith("model_betaVAE_channel_fake")]
    npz = os.path.join(MODELS, "pod_channel_fake_UVW.npz")
    if not pt or not os.path.exists(npz):
        run_script(S + "1_vae_vs_pod/pod.py", "--data", CHAN)
        run_script(S + "1_vae_vs_pod/beta_vae.py", "--data", CHAN, "--epochs", 1)
        pt = [f for f in os.listdir(MODELS) if f.startswith("model_betaVAE_channel_fake")]
    run_script(S + "1_vae_vs_pod/analysis.py", "--vae", os.path.join(MODELS, pt[0]), "--pod", npz, "--data", CHAN)


def test_capacity_headroom_reads_results():
    """capacity_headroom.py on result files in the real schemas (from the region-map dry run
    above, plus empty round / low-data files)."""
    for name, fields in (("experiments_results.csv", ["kind"]), ("experiments2_results.csv", ["kind"]),
                         ("lowdata_results.csv", results.LOWDATA_FIELDS)):
        p = os.path.join(RESULTS, name)
        if not os.path.exists(p):
            with open(p, "w") as f:                  # header only
                f.write(",".join(fields) + "\n")
    if not os.path.exists(os.path.join(RESULTS, "region_map_results.csv")):
        run_script(S + "4_region_map/run_region_map.py", "--datasets", "Re100", "--n", 20, "--subsets", 1, "--epochs", 1)
    run_script(S + "6_capacity/capacity_headroom.py")
    assert os.path.exists(os.path.join(RESULTS, "headroom_corrected_cells.csv"))


def test_every_module_imports():
    """rom/, studies/ and tools/ import without running anything heavy (scripts with
    run_name != '__main__')."""
    for m in ("paths", "data", "split", "pod", "vae", "galerkin_ns", "galerkin_data", "augment", "results"):
        importlib.import_module("rom." + m)
    old = sys.argv
    try:
        for folder in ("studies", "tools"):
            for root, _, files in os.walk(os.path.join(REPO, folder)):
                for f in sorted(files):
                    if f.endswith(".py"):
                        sys.argv = [f]
                        runpy.run_path(os.path.join(root, f), run_name="imported")
    finally:
        sys.argv = old


# ------------------------------------------------------------------ fixed bugs

def test_study1_scripts_read_flat_plate_data():
    """pod.py / analysis.py / inspect_vae.py used their own loader, which could not read the
    2-plates or Alpha0 files; they now use rom.data.load_data."""
    run_script(S + "1_vae_vs_pod/pod.py", "--data", RE100)
    assert os.path.exists(os.path.join(MODELS, "pod_Data2PlatesGap1Re100.npz"))


def test_beta_vae_is_reproducible():
    """beta_vae.py now seeds torch: two runs give the same weights."""
    import torch
    out = []
    for _ in range(2):
        run_script(S + "1_vae_vs_pod/beta_vae.py", "--data", RE100, "--epochs", 1)
        ck = torch.load(os.path.join(MODELS, "model_betaVAE_Data2PlatesGap1Re100_lat5_b8e-04_ep1.pt"),
                        weights_only=False)
        out.append(ck["enc_state"])
    assert all(torch.equal(out[0][k], out[1][k]) for k in out[0])


def test_show_dataset_default_file_and_argument():
    import matplotlib
    matplotlib.use("Agg")
    g = runpy.run_path(os.path.join(REPO, "tools/show_dataset.py"), run_name="imported")
    assert g["FILE"] == os.path.join(DATA, "Alpha0", "dataRe50Alpha0_2.mat")
    g = runpy.run_path(os.path.join(REPO, "tools/show_dataset.py"), run_name="imported")["main"].__globals__
    g["SNAP"] = 10                                   # the fake record is shorter than snapshot 2500
    old, sys.argv = sys.argv, ["show_dataset.py", RE50]
    try:
        g["main"]()
    finally:
        sys.argv = old
    assert os.path.exists(os.path.join(DATA, "Alpha0", "dataRe50Alpha0_2_preview.png"))


def test_headroom_t_value_matches_old_table_and_student_t():
    from scipy.stats import t
    table = {n: round(float(t.ppf(0.975, n - 1)), 3) for n in range(2, 11)}
    assert (table[3], table[5], table[8], table[10]) == (4.303, 2.776, 2.365, 2.262)   # unchanged cells
    assert (table[2], table[4], table[6], table[7]) == (12.706, 3.182, 2.571, 2.447)   # were 2.262


def test_assess_pod_keeps_the_matplotlib_backend():
    import matplotlib
    matplotlib.use("pdf")
    d, _ = data.load_data(RE100)
    pod.assess_pod(d[:120], d[250:270], 4, n_step=60)
    assert matplotlib.get_backend().lower() == "pdf"
    assert os.path.exists(os.path.join(RESULTS, "re100_pod_convergence_k4.png"))
    matplotlib.use("Agg")
