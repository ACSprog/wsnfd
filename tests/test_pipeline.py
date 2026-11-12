import numpy as np, pandas as pd, pytest
from wsnfd.config import Config
from wsnfd import data as D, change as C, faults as Fa, features as Fe, metrics as M
from wsnfd.prequential import run_stream
import wsnfd.models as Mo

def base(T=6000, N=8, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(T)
    temp = 20 + 3 * np.sin(2 * np.pi * t / 288)[:, None] + rng.normal(0, .2, (T, N))
    hum = 40 - 1.5 * (temp - 20) + rng.normal(0, .5, (T, N))
    return np.stack([temp, hum, rng.normal(100, 5, (T, N)), np.full((T, N), 2.7)], axis=2)

def cfg(**k): return Config(grid_s=300, **k)

def test_fault_rate_labels_and_stuck():
    X = base(); c = cfg(); te = int(c.train_frac * len(X)); valid = np.isfinite(X).all(axis=2)
    sg = Fa.node_sigma(X, te); rng = np.random.default_rng(1)
    yb, yt = Fa.inject(X, valid, sg, (te, len(X)), rng, c)
    assert abs(yb[te:].mean() - c.fault_rate) < 0.03          # rate near the 10% target
    assert yb[:te].sum() == 0                                    # nothing outside the segment
    assert set(np.unique(yt)) <= set(range(6)) and ((yt > 0) == (yb > 0)).all()
    t0, n = np.argwhere(yt == 5)[0]                              # first stuck-at observation (time, node)
    end = t0
    while end + 1 < len(X) and yt[end + 1, n] == 5: end += 1
    assert end > t0                                              # an episode spans several epochs
    held = [np.ptp(X[t0:end + 1, n, ch]) < 1e-9 for ch in (0, 1)]
    assert any(held)                                             # value held constant on the faulty channel
    assert np.isfinite(X).all()

def test_change_scenarios():
    X = base(); c = cfg(); te = int(c.train_frac * len(X))
    sd = X[:te, :, 0].std()
    X0, on = C.apply_change(X, "S0", te, c); assert np.array_equal(X0, X)
    X2, on = C.apply_change(X, "S2", te, c)
    assert np.allclose(X2[:on], X[:on])                          # nothing before onset
    shift = (X2[on:, :, 0] - X[on:, :, 0]).mean(); assert abs(shift - 3 * sd) < 0.05 * sd + 1e-6
    dh = (X2[on:, :, 1] - X[on:, :, 1]).mean(); assert dh < 0     # humidity moves with the negative T-H correlation
    X4, on = C.apply_change(X, "S4", te, c); L = int(c.transition_days * c.epochs_per_day())
    assert abs((X4[on + 1, :, 0] - X[on + 1, :, 0]).mean()) < 0.1 * sd   # gradual: starts near zero
    assert abs((X4[min(on + L + 5, len(X) - 1), :, 0] - X[min(on + L + 5, len(X) - 1), :, 0]).mean() - 3 * sd) < 0.05 * sd + 1e-6

def test_features_are_causal():
    X = base(2000); te = 600; F1, _ = Fe.build_features(X, te, "FS-A", 10)
    Y = X.copy(); Y[1200:, :, :] += 50                           # alter the future only
    F2, _ = Fe.build_features(Y, te, "FS-A", 10)
    assert np.allclose(F1[:1200], F2[:1200], equal_nan=True)
    FB, names = Fe.build_features(X, te, "FS-B", 10); assert FB.shape[2] == 4

class Probe(Mo.Base):
    name = "M1"
    def __init__(self, c, s): super().__init__(c, s); self.events = []
    def predict(self, X): self.events.append(("p", len(X))); return np.zeros(len(X), int)
    def observe(self, when, n_epochs, X, y, yhat): self.events.append(("o", when))

@pytest.mark.parametrize("delay", [0.0, 1.0])
def test_prequential_order_and_delay(delay, monkeypatch):
    T, N = 3000, 4; c = cfg(batch_epochs=96, label_delay_days=delay); te = 800
    F = np.random.default_rng(0).normal(size=(T, N, 3)); y = (np.random.default_rng(1).random((T, N)) < .1).astype(np.int8)
    ok = np.ones((T, N), bool); probe = {}
    def mk(name, cfg_, seed): m = Probe(cfg_, seed); m.init = lambda X, yy: None; probe["m"] = m; return m
    monkeypatch.setattr("wsnfd.prequential.make_model", mk)
    run_stream(F, y, y, ok, te, c, ["M1"], 0)
    ev = probe["m"].events; d = int(round(delay * c.epochs_per_day() / c.batch_epochs))
    kinds = [k for k, _ in ev]; npred = kinds.count("p"); nobs = kinds.count("o")
    assert kinds[0] == "p" and nobs == npred - d                 # d batches of labels are still pending at the end
    seen = 0
    for i, k in enumerate(kinds):
        if k == "p": seen += 1
        else: assert seen >= (kinds[:i].count("o") + 1) + d      # k-th label reveal follows the (k+d)-th prediction

def test_selection_rule_fails_with_few_nodes():
    X = base(T=30 * 288, N=5); assert D.select_nodes_period(X, 300, 10, 28, 0.8) is None
    X = base(T=30 * 288, N=12); assert D.select_nodes_period(X, 300, 10, 28, 0.8) is not None

def test_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0]); p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    s = M.scores(y, p); assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 4)
    assert abs(s["f1"] - 2 / 3) < 1e-9 and abs(s["mcc"] - (8 - 1) / np.sqrt(3 * 3 * 5 * 5)) < 1e-9
    assert abs(s["accuracy"] - 6 / 8) < 1e-9 and abs(M.scores(y, np.zeros(8, int))["accuracy"] - 5 / 8) < 1e-9   # a detector that flags nothing still scores well on accuracy

def test_loader_skips_malformed_rows(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("2004-03-01 00:00:10 1 1 20.1 40.0 100 2.7\n2004-03-01 garbage\n2004-03-01 00:00:41 2 1 20.2 40.1 101 2.7\n")
    df = D.load_intel(str(f)); assert len(df) == 2 and df["moteid"].iloc[0] == 1

def test_cleaning_removes_implausible_and_dead_nodes():
    df = pd.DataFrame({"ts": pd.date_range("2004-03-01", periods=5, freq="31s"), "moteid": 1,
                       "temperature": [20, 122, 20, 20, 20.], "humidity": 40., "light": 100., "voltage": [2.7, 2.7, 2.7, 2.0, 2.7]})
    c = D.clean(df); assert np.isnan(c.loc[1, "temperature"]) and c.loc[3:, "temperature"].isna().all()

def test_m4b_has_detector_disabled():
    """M4 must detect drift on a label flip; M4b (NoDrift) must not."""
    rng = np.random.default_rng(0); c = Config()
    X1 = rng.normal(size=(2000, 3)); y1 = (X1[:, 0] > 0).astype(int)
    X2 = rng.normal(size=(2000, 3)); y2 = 1 - (X2[:, 0] > 0).astype(int)
    found = {}
    for det in (True, False):
        m = Mo.StreamARF(c, 0, det); m.init(X1, y1)
        for i in range(0, 2000, 100):
            p = m.predict(X2[i:i + 100]); m.observe(i, 100, X2[i:i + 100], y2[i:i + 100], p)
        found[det] = m.rf.n_drifts_detected()
    assert found[True] > 0 and found[False] == 0

def test_simulator_spatial_correlation_and_anomalies():
    df, co = D.simulate(n_nodes=16, days=14, grid_s=300, seed=3, return_coords=True)
    cl = D.clean(df); X, nodes, _ = D.align(cl, 300)
    ids = list(nodes); xy = np.array([co[int(n)] for n in ids]); d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    with np.errstate(all='ignore'): T0 = np.nan_to_num(X[:, :, 0] - np.nanmean(X[:, :, 0], axis=1, keepdims=True))
    # remove the shared daily cycle, then compare residual correlation of near and far pairs
    R = np.corrcoef(T0.T); iu = np.triu_indices(len(ids), 1)
    near = R[iu][d[iu] < np.percentile(d[iu], 25)].mean(); far = R[iu][d[iu] > np.percentile(d[iu], 75)].mean()
    assert near > far                                            # neighbours co-vary more than distant nodes
    raw_bad = ((df.temperature > 50) | (df.humidity < 0) | (df.humidity > 100)).sum()
    assert raw_bad > 0                                           # brown-out anomalies exist in the raw data ...
    assert ((cl.temperature > 50) | (cl.humidity < 0) | (cl.humidity > 100)).sum() == 0   # ... and cleaning removes them

def test_simulator_missingness_is_bursty_not_uniform():
    df = D.simulate(n_nodes=6, days=10, grid_s=300, seed=4)
    X, nodes, _ = D.align(df, 300); miss = np.isnan(X[:, :, 0])
    gaps = []
    for n in range(X.shape[1]):
        m = miss[:, n].astype(int); run = 0
        for v in m:
            if v: run += 1
            elif run: gaps.append(run); run = 0
    gaps = np.array(gaps); p = miss.mean()                       # independent loss at the same rate would rarely exceed
    indep_max = np.log(len(miss.ravel())) / -np.log(p)           # about ln(N)/-ln(p) consecutive misses
    assert gaps.max() > 2 * indep_max                            # bursts far longer than independent loss allows

def test_knn_and_neighbour_features():
    coords = {1: (0, 0), 2: (1, 0), 3: (10, 0), 4: (11, 0)}
    nb = D.knn_neighbours(coords, [1, 2, 3, 4], k=1); assert nb == [[1], [0], [3], [2]]
    X = base(1500, 4); F, names = Fe.build_features(X, 400, "FS-A", 10, neighbours=nb)
    assert np.isfinite(F[200:, :, names.index("temperature_nbdev")]).all()

def test_locations_loader(tmp_path):
    f = tmp_path / "loc.txt"; f.write_text("1 21.5 23\n2 24.5 20\nbad line\n")
    assert D.load_locations(str(f)) == {1: (21.5, 23.0), 2: (24.5, 20.0)}

# ---------- engine: analysis, report, export ----------
import os
HERE_ = os.path.dirname(os.path.abspath(__file__))
def _fixture():
    import pandas as pd
    return pd.read_csv(os.path.join(HERE_, "fixtures", "runs.csv")), pd.read_csv(os.path.join(HERE_, "fixtures", "blocks.csv"))

def test_analyse_multi_folder_keeps_experiments_apart(tmp_path):
    from wsnfd import analyse as A
    r, b = _fixture()
    for name in ("a", "b"):
        d = tmp_path / name; d.mkdir(); r.to_csv(d / "runs.csv", index=False); b.to_csv(d / "blocks.csv", index=False)
    runs, blocks = A.load_folders([str(tmp_path / "a"), str(tmp_path / "b")])
    assert set(runs.run_label) == {"a", "b"}
    S = A.summary(runs); assert "run_label" in S.columns and S.n_seeds.eq(3).all()
    assert len(A.tests(runs)) == 2 * len(A.tests(r))               # tests run separately inside each experiment
    assert {"cohen_dz", "n_pairs"} <= set(A.tests(r).columns)
    assert len(A.find_result_folders(str(tmp_path))) == 2

def test_explore_and_sensitivity():
    from wsnfd import analyse as A
    r, _ = _fixture()
    e = A.explore(r, "f1", ["model"], ("mean", "sd", "ci95", "median", "n"), where={"scenario": ["S2"]})
    assert set(e.columns) == {"model", "mean", "sd", "ci95", "median", "n"} and (e.n == 3).all()
    assert abs(e[e.model == "M1"]["mean"].iloc[0] - r[(r.scenario == "S2") & (r.model == "M1")].f1.mean()) < 1e-12
    assert len(A.explore(r, "f1", [], ("mean",))) == 1
    assert "f1" in A.numeric_columns(r) and "seed" not in A.numeric_columns(r)

def test_report_writes_all_outputs(tmp_path):
    import zipfile
    from wsnfd import report
    r, b = _fixture(); out = str(tmp_path / "rep")
    files = report.write_all(r, b, out)
    names = {os.path.basename(f) for f in files}
    assert {"summary.csv", "tests_post_change_f1.csv", "results.xlsx", "bundle.zip", "blocks_S2.png", "cost.png"} <= names
    assert all(os.path.getsize(f) > 0 for f in files)
    assert any(n.endswith("png") for n in zipfile.ZipFile(out + "/bundle.zip").namelist())

def test_dataset_export_matches_what_models_see():
    from wsnfd.config import Config
    from wsnfd import run as R, export as E
    cfg = Config(grid_s=600, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=0))
    df = E.build_dataset(X, info, cfg, "S2", 0, features=True, neighbours=info.get("neighbours"))
    Xs, yb, yt, onset, te, valid = R.make_stream(X, cfg, "S2", 0)
    assert len(df) == int(valid.sum()) and int(df.y_fault.sum()) == int(yb[valid].sum())
    t = df[df.node == df.node.iloc[0]].iloc[0]; n = list(info["nodes"]).index(t.node)
    assert abs(t.temperature - Xs[int(t.epoch), n, 0]) < 1e-9
    assert set(df.fault_type) <= {"normal", "spike", "noise", "bias", "drift", "stuck"} and "temperature_nbdev" in df.columns
    assert df[df.segment == "train"].after_change.eq(0).all() and df[df.epoch >= onset].after_change.eq(1).all()
    base = E.build_dataset(X, info, cfg, "baseline", 0); assert base.y_fault.sum() == 0
    assert (E.build_dataset(X, info, cfg, "S2", 0, test_only=True).segment == "test").all()
    df2 = E.build_dataset(X, info, cfg, "S2", 0); assert df2.y_fault.sum() == df.y_fault.sum()    # deterministic

def test_one_run_is_deterministic():
    from wsnfd.config import Config
    from wsnfd import run as R
    cfg = Config(grid_s=900, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    a, _ = R.one_run(X, cfg, "S1", 3, ["M1", "M2"]); b, _ = R.one_run(X, cfg, "S1", 3, ["M1", "M2"])
    assert [x["f1"] for x in a] == [x["f1"] for x in b]

# --- hand-over to other software
def test_mat_roundtrip_preserves_numbers_and_text(tmp_path):
    import io
    from scipy.io import loadmat
    from wsnfd import interop as IO
    df = pd.DataFrame({"model": ["M1", "M2"], "post_change_f1": [0.5, np.nan], "2bad name": [1, 2], "flag": [True, False]})
    b = io.BytesIO(); IO.tables_to_mat({"raw_runs": df}, b); b.seek(0)
    m = loadmat(b, squeeze_me=True, struct_as_record=False)["raw_runs"]
    assert list(m.model) == ["M1", "M2"] and m.post_change_f1[0] == 0.5 and np.isnan(m.post_change_f1[1])
    assert list(m.t_2bad_name) == [1, 2] and list(m.flag) == [1, 0]

def test_dataset_formats_agree_and_report_ships_matlab_files(tmp_path):
    import io, os
    from scipy.io import loadmat
    from wsnfd import interop as IO, report as RP, run as R, export as EX
    cfg = Config(grid_s=600, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=0))
    df = EX.build_dataset(X, info, cfg, "S2", 0, False, True, info.get("neighbours"))
    csv = pd.read_csv(io.BytesIO(IO.dataset_bytes(df, "csv")[0])); pq = pd.read_parquet(io.BytesIO(IO.dataset_bytes(df, "parquet")[0]))
    mat = loadmat(io.BytesIO(IO.dataset_bytes(df, "mat")[0]), squeeze_me=True, struct_as_record=False)["dataset"]
    assert len(csv) == len(pq) == len(mat.y_fault) == len(df) and int(mat.y_fault.sum()) == int(df.y_fault.sum())
    runs = pd.read_csv(os.path.join(os.path.dirname(__file__), "fixtures", "runs.csv")); blocks = pd.read_csv(os.path.join(os.path.dirname(__file__), "fixtures", "blocks.csv"))
    files = [os.path.basename(p) for p in RP.write_all(runs, blocks, str(tmp_path))]
    assert {"results.mat", "results.xlsx", "load_in_matlab.m", "load_in_R.R", "load_in_python.py", "analysis.ipynb"} <= set(files)

def test_notebook_is_valid_and_executes_on_exported_tables(tmp_path):
    import json, os, subprocess, sys
    from wsnfd import interop as IO, report as RP
    nb = json.loads(IO.make_notebook()); assert nb["nbformat"] == 4 and sum(c["cell_type"] == "code" for c in nb["cells"]) >= 5
    fx = os.path.join(os.path.dirname(__file__), "fixtures"); runs = pd.read_csv(os.path.join(fx, "runs.csv")); blocks = pd.read_csv(os.path.join(fx, "blocks.csv"))
    RP.write_all(runs, blocks, str(tmp_path))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code").replace("plt.show()", "plt.close()")
    r = subprocess.run([sys.executable, "-I", "-c", "import matplotlib; matplotlib.use('Agg')\n" + src], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]

def test_peak_memory_helper_works_here_and_never_crashes_without_the_unix_module(monkeypatch):
    import sys, math
    from wsnfd import run as R
    assert R.peak_rss_mb() > 1                                           # real value where `resource` exists
    monkeypatch.setitem(sys.modules, "resource", None)                   # `import resource` now raises ImportError, as on Windows
    v = R.peak_rss_mb(); assert isinstance(v, float) and (math.isnan(v) or v > 1)

def test_run_module_imports_without_the_unix_module(monkeypatch):
    import sys, importlib
    monkeypatch.setitem(sys.modules, "resource", None)
    import wsnfd.run as R; importlib.reload(R); assert callable(R.one_run)


def test_accuracy_is_reported_and_derived_for_older_results(tmp_path):
    from wsnfd import analyse as A
    r, b = _fixture(); r = r.drop(columns=[c for c in ["accuracy"] if c in r.columns])
    assert {"tp", "fp", "fn", "tn"} <= set(r.columns)
    d = tmp_path / "old"; d.mkdir(); r.to_csv(d / "runs.csv", index=False); b.to_csv(d / "blocks.csv", index=False)
    R, B = A.load_folders([str(d)])
    assert np.allclose(R.accuracy, (R.tp + R.tn) / (R.tp + R.fp + R.fn + R.tn))
    assert "accuracy_mean" in A.summary(R).columns

def test_periodic_model_first_retrain_waits_one_period():
    """M2's clock starts at the start of the test stream: no retraining on a one-batch window in the first hour."""
    c = cfg(n_estimators=5, period_days=1.0, window_days=3.0); T = 6000; te = int(c.train_frac * T)
    rng = np.random.default_rng(0); F = rng.normal(size=(T, 4, 3)); y = (F[:, :, 0] > 0.8).astype(np.int8)
    res = run_stream(F, y, y, np.ones((T, 4), bool), te, c, ["M2"], 0)
    log = res["models"]["M2"].update_log; P = int(c.period_days * c.epochs_per_day())
    assert log and log[0][0] - te >= P                              # first update at least one period after the stream starts
    assert all(b[0] - a[0] >= P for a, b in zip(log, log[1:]))       # and updates are at least P apart

def test_hyperparameters_are_tuned_once_and_logged(tmp_path):
    from wsnfd import run as R
    cfg = Config(grid_s=900, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    best = R.tune_once(X, cfg, info.get("neighbours"))
    assert (cfg.n_estimators, cfg.max_depth, cfg.min_samples_leaf) == (best["n_estimators"], best["max_depth"], best["min_samples_leaf"])
    env = R.environment_record(cfg, {"source": "sim"})        # what both the command line and the workbench write beside the results
    assert env["config"]["n_estimators"] == best["n_estimators"] and {"sklearn", "river", "scipy", "numpy", "pandas", "python", "platform"} <= set(env)
    assert {"machine", "processor", "logical_cpus", "memory_gb"} <= set(env["hardware"]) and env["hardware"]["logical_cpus"] >= 1
    ts = env["tuning_scores"]; assert len(ts) == 4 and {t["n_estimators"] for t in ts} == {10}        # trees are fixed; depth and leaf size are scored
    assert max(ts, key=lambda t: t["mean_f1"])["max_depth"] == best["max_depth"]
    assert "tuning_scores" not in R.environment_record(Config(), {"source": "sim"})                  # a run without tuning records none

def test_tune_only_writes_scores_and_runs_nothing(tmp_path):
    from wsnfd import run as R
    out = tmp_path / "t"
    R.main(["--source", "sim", "--grid-s", "900", "--sim-nodes", "30", "--sim-days", "36", "--sim-seed", "1", "--trees", "10", "--tune-only", "--out", str(out)])
    t = pd.read_csv(out / "tuning_scores.csv"); assert len(t) == 4 and t.chosen.sum() == 1 and not (out / "runs.csv").exists()

def test_only_the_drift_triggered_model_reports_alarms():
    from wsnfd import run as R
    cfg = Config(grid_s=900, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    rows, _ = R.one_run(X, cfg, "S2", 0, ["M1", "M2", "M3"])
    by = {r["model"]: r for r in rows}
    assert np.isnan(by["M1"]["n_alarms"]) and np.isnan(by["M2"]["n_alarms"]) and not np.isnan(by["M3"]["n_alarms"])     # no zero that could be read as "no false alarms"
    assert by["M3"]["predict_time_per_obs_s"] > 0

def test_cleaning_is_tight_on_battery_failure_and_implausible_indoor_values():
    """Readings above 40 C or outside 5-95% humidity are removed, and a mote is treated as degraded from its first reading below the cut-off."""
    df = pd.DataFrame({"ts": pd.date_range("2004-03-01", periods=6, freq="31s"), "moteid": 1,
                       "temperature": [20, 41, 20, 20, 20, 20.], "humidity": [40., 40, 3, 40, 40, 40], "light": 100.,
                       "voltage": [2.7, 2.7, 2.7, 2.7, 2.15, 2.7]})
    c = D.clean(df)
    assert np.isnan(c.loc[1, "temperature"]) and np.isnan(c.loc[2, "humidity"])
    assert c.loc[[0, 3], "temperature"].notna().all() and c.loc[4:, "temperature"].isna().all()        # 2.15 V < 2.2 V: degraded from row 4 on
    assert D.clean(df, dead_voltage=2.1).loc[4:, "temperature"].notna().all()                         # the cut-off is adjustable

def test_cleaning_comparison_and_settings_reach_the_selection():
    from wsnfd.config import Config
    df = D.simulate(n_nodes=14, days=34, grid_s=600, seed=0)
    t = D.cleaning_sweep(df, Config(grid_s=600), cutoffs=(2.1, 2.4))
    assert len(t) == 4 and {"battery_cutoff_V", "nodes_in_best_window", "rule_met", "bounds"} <= set(t.columns)
    a = t[(t.battery_cutoff_V == 2.1)].nodes_in_best_window.max(); b = t[(t.battery_cutoff_V == 2.4)].nodes_in_best_window.max()
    assert a >= b                                                          # a stricter cut-off can only remove readings
    cfg = Config(grid_s=600, temp_min=15.0, temp_max=30.0, dead_voltage=2.3); assert cfg.bounds()["temperature"] == (15.0, 30.0)

def test_same_folder_given_twice_is_loaded_once(tmp_path):
    import shutil, os
    import wsnfd.analyse as A
    d = tmp_path / "res" / "run1"; d.mkdir(parents=True)
    fix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    for f in ("runs.csv", "blocks.csv"): shutil.copy(os.path.join(fix, f), d / f)
    one, _ = A.load_folders([str(d)])
    two, _ = A.load_folders([str(d), str(d) + os.sep, str(tmp_path / "res" / "." / "run1")])
    assert len(two) == len(one)
    assert all(os.path.normpath(f) == f for f in A.find_result_folders(str(tmp_path / "res")))


# ---- measures on usable blocks, and the change-specific analysis (S0 as reference)
def test_usable_blocks_drop_thin_blocks():
    import numpy as np, wsnfd.metrics as M
    n = np.array([3700, 3700, 3600, 1100, 3700, 1300, 150, 0])
    assert M.usable_blocks(n).tolist() == [True, True, True, False, True, False, False, False]

def test_adaptation_ignores_unusable_blocks_and_uses_early_window():
    import numpy as np, wsnfd.metrics as M
    bf = np.array([0.6, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0])          # onset at block 3; last block is a thin one scoring 0
    full = M.adaptation(bf, 3)
    cut = M.adaptation(bf, 3, usable=np.array([1, 1, 1, 1, 1, 1, 1, 0], bool))
    assert abs(cut["pre_change_f1"] - 0.6) < 1e-12
    assert abs(cut["degradation"] - (0.6 - np.mean([0.5, 0.4, 0.3]))) < 1e-12          # mean of the first three blocks after the change
    assert abs(cut["post_change_f1"] - np.mean([0.5, 0.4, 0.3, 0.2])) < 1e-12
    assert cut["post_change_f1"] > full["post_change_f1"]                                # the thin zero block no longer drags the mean down

def _mini(tmp_path, with_counts):
    import numpy as np, pandas as pd, os
    rows, curves = [], []
    for sc in ("S0", "S1"):
        for seed in range(4):
            for model in ("M1", "M3"):
                bf = [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.0] if (sc == "S1" and model == "M1") else [0.5] * 7 + [0.0]
                rows.append(dict(scenario=sc, seed=seed, model=model, onset_block=3, pre_change_f1=9, degradation=9, recovery_blocks=9, post_change_f1=9))
                for i, v in enumerate(bf):
                    c = dict(scenario=sc, seed=seed, model=model, block=i, f1=v)
                    if with_counts: c["n_obs"] = 3000 if i < 7 else 100; c["n_faults"] = 300
                    curves.append(c)
    d = tmp_path / "res" / "mini"; d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "runs.csv", index=False); pd.DataFrame(curves).to_csv(d / "blocks.csv", index=False)
    return str(d)

def test_loading_recomputes_measures_from_blocks_old_and_new_results(tmp_path):
    import wsnfd.analyse as A
    for with_counts, rule in ((False, "final scored block dropped"), (True, "block counts")):
        d = _mini(tmp_path / str(with_counts), with_counts)
        runs, _ = A.load_folders([d])
        assert set(runs.block_rule) == {rule}
        a = runs[(runs.scenario == "S0") & (runs.model == "M3")].iloc[0]
        assert abs(a.post_change_f1 - 0.5) < 1e-12 and abs(a.degradation) < 1e-12        # the thin final block is not scored
        b = runs[(runs.scenario == "S1") & (runs.model == "M1")].iloc[0]
        assert abs(b.degradation - 0.1) < 1e-12 and abs(b.post_change_f1 - 0.4) < 1e-12

def test_change_effects_use_s0_of_the_same_seed_as_reference(tmp_path):
    import numpy as np, pandas as pd, wsnfd.analyse as A
    rows = []
    for seed in range(8):
        base = 0.2 + 0.01 * seed
        for sc in ("S0", "S1"):
            drop = 0.0 if sc == "S0" else 0.10                 # the change lowers the static model by 0.10, not the adaptive one
            rows.append(dict(scenario=sc, seed=seed, model="M1", post_change_f1=base - drop))
            rows.append(dict(scenario=sc, seed=seed, model="M3", post_change_f1=base + 0.08))   # adaptive: +0.08 over M1 in S0, +0.18 in S1
    runs = pd.DataFrame(rows)
    E = A.change_effects(runs, reference="M1", models=["M1", "M3"])
    h = E[(E.effect == "harm of the change") & (E.model == "M1")].iloc[0]; g = E[E.effect == "extra gain of adaptation"].iloc[0]
    assert abs(h.mean_diff + 0.10) < 1e-9 and h.significant_holm and h.beyond_margin
    assert abs(g.mean_diff - 0.10) < 1e-9 and g.model == "M3 vs M1" and g.significant_holm
    assert (E[(E.effect == "harm of the change") & (E.model == "M3")].mean_diff.abs() < 1e-9).all()

def test_worthwhile_beyond_drift_is_false_when_the_gain_is_already_there_in_s0():
    import pandas as pd, wsnfd.analyse as A
    rows = []
    for seed in range(8):
        for sc in ("S0", "S1"):
            rows.append(dict(scenario=sc, seed=seed, model="M1", post_change_f1=0.2 + 0.01 * seed, fit_time_s=0.0, predict_time_s=1.0, predict_time_per_obs_s=0, n_updates=0, n_train_rows=0, labels_consumed=0, model_bytes=1))
            rows.append(dict(scenario=sc, seed=seed, model="M3", post_change_f1=0.3 + 0.01 * seed + 0.001 * (seed % 3), fit_time_s=5.0, predict_time_s=1.0, predict_time_per_obs_s=0, n_updates=3, n_train_rows=10, labels_consumed=10, model_bytes=1))
    runs = pd.DataFrame(rows); T = A.tests(runs, "post_change_f1", 0.02, ["M1", "M3"]); W = A.worthwhile(T, runs, 0.02, "M3", "M1")
    assert W.worthwhile.all()                                   # adaptive beats static in both scenarios, as in the real data
    assert not W.worthwhile_beyond_drift.any()                  # but the injected change added nothing beyond what S0 already shows

def test_command_line_change_table_equals_function(tmp_path, capsys):
    import wsnfd.analyse as A
    d = _mini(tmp_path, True); out = str(tmp_path / "c.csv")
    A.main([d, "--what", "change", "--models", "M1,M3", "--reference", "M1", "--out", out])
    import pandas as pd
    runs, _ = A.load_folders([d]); E = A.change_effects(runs, "post_change_f1", 0.02, "M1", ["M1", "M3"])
    assert len(pd.read_csv(out)) == len(E) > 0

def test_new_runs_write_block_counts(tmp_path):
    import numpy as np, wsnfd.run as R, wsnfd.config as C, wsnfd.data as D
    cfg = C.Config(grid_s=900, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    rows, curves = R.one_run(X, cfg, "S2", 0, ["M1"], False, info.get("neighbours"))
    assert {"n_obs", "n_faults"} <= set(curves[0]) and sum(c["n_obs"] for c in curves) > 0


def test_add_block_counts_matches_new_runs_and_refuses_wrong_settings(tmp_path):
    import shutil, numpy as np, pandas as pd, pytest, wsnfd.run as R, wsnfd.analyse as A
    from wsnfd.config import Config
    cfg = Config(grid_s=900, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    rows, curves = R.one_run(X, cfg, "S2", 0, ["M1"], False, info.get("neighbours"))
    d = tmp_path / "res" / "old"; d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "runs.csv", index=False); new = pd.DataFrame(curves); new.to_csv(tmp_path / "new_blocks.csv", index=False)
    new.drop(columns=["n_obs", "n_faults"]).to_csv(d / "blocks.csv", index=False)
    with pytest.raises(ValueError): R.add_block_counts(str(d), X[::2], cfg, info.get("neighbours"))
    assert "n_obs" not in pd.read_csv(d / "blocks.csv").columns                 # nothing written on a mismatch
    R.add_block_counts(str(d), X, cfg, info.get("neighbours"))
    got = pd.read_csv(d / "blocks.csv"); assert np.array_equal(got.n_obs.to_numpy(), new.n_obs.to_numpy())
    assert (d / "blocks_before_counts.csv").exists()
    runs, _ = A.load_folders([str(d)]); assert set(runs.block_rule) == {"block counts"}


def test_charts_and_explorer_accept_the_same_column_twice():
    import pandas as pd, numpy as np, wsnfd.plots as P, wsnfd.analyse as A
    runs = pd.DataFrame({"scenario": ["S0", "S0", "S1", "S1"] * 3, "model": ["M1", "M2"] * 6, "seed": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], "post_change_f1": np.linspace(.1, .5, 12)})
    for fn in (P.fig_bar, P.fig_box):
        for x, hue in (("model", "model"), ("scenario", "scenario"), ("scenario", "model"), ("model", None)): assert P.to_png(fn(runs, "post_change_f1", x, hue))
    assert len(A.explore(runs, "post_change_f1", ["model", "model"])) == 2

def test_windows_memory_structure_and_fallback():
    import ctypes
    from wsnfd import run as R
    MS = R._memorystatusex()
    assert ctypes.sizeof(MS) == 64 and MS.ullTotalPhys.offset == 8          # the Windows MEMORYSTATUSEX layout
    class Fake:                                                              # stands in for kernel32 on a machine with 16 GB
        def GlobalMemoryStatusEx(self, p):
            assert p.contents.dwLength == 64; p.contents.ullTotalPhys = 17_000_000_000; return 1
    assert R.windows_memory_gb(Fake()) == 17.0

def test_metrics_agree_with_scikit_learn_and_river():
    """Independent check of the detection measures: the same predictions scored by wsnfd, by scikit-learn and by River's streaming metrics."""
    from sklearn import metrics as SK
    from river import metrics as RM
    rng = np.random.default_rng(7)
    for rate, skill in [(0.10, 0.6), (0.10, 0.2), (0.02, 0.5), (0.30, 0.8)]:
        y = (rng.random(4000) < rate).astype(int)
        p = np.where(rng.random(4000) < skill, y, (rng.random(4000) < rate).astype(int))
        s = M.scores(y, p)
        assert s["precision"] == pytest.approx(SK.precision_score(y, p, zero_division=0))
        assert s["recall"] == pytest.approx(SK.recall_score(y, p, zero_division=0))
        assert s["f1"] == pytest.approx(SK.f1_score(y, p, zero_division=0))
        assert s["mcc"] == pytest.approx(SK.matthews_corrcoef(y, p))
        assert s["accuracy"] == pytest.approx(SK.accuracy_score(y, p))
        f1, pr, rc, mcc = RM.F1(), RM.Precision(), RM.Recall(), RM.MCC()
        for a, b in zip(y.tolist(), p.tolist()):
            for m in (f1, pr, rc, mcc): m.update(bool(a), bool(b))
        assert s["f1"] == pytest.approx(f1.get()) and s["precision"] == pytest.approx(pr.get())
        assert s["recall"] == pytest.approx(rc.get()) and s["mcc"] == pytest.approx(mcc.get())

def test_bundle_carries_block_curves_and_each_runs_environment(tmp_path):
    import os, shutil, zipfile
    from wsnfd import report as RP
    fx = os.path.join(os.path.dirname(__file__), "fixtures"); f = tmp_path / "Run-A"; f.mkdir()
    for n in ("runs.csv", "blocks.csv"): shutil.copy(os.path.join(fx, n), f / n)
    (f / "environment.json").write_text('{"config": {"feature_set": "FS-X"}, "tuning_scores": []}')
    RP.main([str(f), "--out", str(tmp_path / "out")])
    names = zipfile.ZipFile(tmp_path / "out" / "bundle.zip").namelist()
    assert "tables/block_curves.csv" in names and "run_files/Run-A_environment.json" in names and "tables/raw_runs.csv" in names
    assert len(pd.read_csv(tmp_path / "out" / "tables" / "block_curves.csv")) == len(pd.read_csv(os.path.join(fx, "blocks.csv")))

def test_report_files_are_named_after_the_runs(tmp_path):
    from wsnfd import report as RP
    runs = pd.DataFrame({"run_label": ["FS-B-Main-Run", "FS-A Main/Run", "FS-B-Main-Run"]})
    assert RP.file_tag(runs) == "FS-A-Main-Run+FS-B-Main-Run"
    assert RP.file_tag(pd.DataFrame({"run_label": [f"Sens-run-number-{i}-with-a-long-name" for i in range(6)]})).endswith("+4-more")
    assert RP.file_tag(pd.DataFrame({"x": [1]})) == "results"

def _sim_stream(**kw):
    from wsnfd import run as R, features as Fe
    cfg = Config(grid_s=900, n_estimators=10, **kw); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=1))
    Xs, yb, yt, onset, train_end, valid = R.make_stream(X, cfg, "S2", 0)
    F, _ = Fe.build_features(Xs, train_end, cfg.feature_set, cfg.roll_window, info.get("neighbours"))
    ok = valid & np.isfinite(F).all(axis=2)
    return cfg, F, yb, yt, ok, train_end

def test_prefilled_buffer_gives_a_full_first_window_and_cold_start_reproduces_the_first_design():
    from wsnfd.prequential import run_stream
    cfg, F, yb, yt, ok, te = _sim_stream()
    new = run_stream(F, yb, yt, ok, te, cfg, ["M2", "M3"], 0)["models"]
    cfg0, F0, yb0, yt0, ok0, te0 = _sim_stream(prefill_buffer=False, adwin_input="observation")
    old = run_stream(F0, yb0, yt0, ok0, te0, cfg0, ["M2", "M3"], 0)["models"]
    assert new["M2"].update_log[0][1] > old["M2"].update_log[0][1]                  # the first periodic retraining sees the last W days, not one day
    assert len(old["M3"].alarms) > 5 and len(new["M3"].alarms) < len(old["M3"].alarms)       # one value per reading alarms far more often than one per batch
    assert old["M3"].update_log[0][1] == old["M2"].update_log[0][1]                          # cold start: both first retrain on the same single day of revealed data

def test_batch_detector_gets_one_value_per_batch():
    from wsnfd.models import DriftTriggered
    cfg = Config(n_estimators=5); seen = []
    m = DriftTriggered(cfg, 0)
    class Spy:
        drift_detected = False
        def update(self, v): seen.append(v)
    m.det = Spy(); y = np.array([0, 1, 1, 0, 1]); yh = np.array([0, 0, 1, 0, 0])
    m.observe(100, 100, np.zeros((5, 3)), y, yh)
    assert seen == [pytest.approx(0.4)]
    m2 = DriftTriggered(Config(n_estimators=5, adwin_input="observation"), 0); seen.clear(); m2.det = Spy()
    m2.observe(100, 100, np.zeros((5, 3)), y, yh)
    assert seen == [0, 1, 0, 0, 1]

def test_bundle_for_a_run_without_the_reference_model(tmp_path):
    """A label-delay run holds M2 and M3 only (M1 is unaffected by delay): the report bundle must still build, with an empty 'worthwhile' table."""
    import os, zipfile
    from wsnfd import report as RP, analyse as A
    fx = os.path.join(os.path.dirname(__file__), "fixtures")
    runs = pd.read_csv(os.path.join(fx, "runs.csv")); blocks = pd.read_csv(os.path.join(fx, "blocks.csv"))
    runs = runs[runs.model != "M1"].reset_index(drop=True); blocks = blocks[blocks.model != "M1"].reset_index(drop=True)
    assert set(runs.model) == {"M2", "M3"}
    tabs = RP.build_tables(runs, blocks)
    assert tabs["worthwhile"].empty and "worthwhile_beyond_drift" in tabs["worthwhile"].columns
    assert len(tabs["tests_post_change_f1"]) > 0                               # M2 against M3 is still tested
    files = RP.write_all(runs, blocks, str(tmp_path / "rep"))
    assert os.path.exists(tmp_path / "rep" / "bundle.zip") and any(f.endswith("worthwhile.csv") for f in files)
