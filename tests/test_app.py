import os, shutil, pytest, pandas as pd
from streamlit.testing.v1 import AppTest
HERE = os.path.dirname(os.path.abspath(__file__)); APP = os.path.join(HERE, "..", "app.py"); FIX = os.path.join(HERE, "fixtures")

def results_root(tmp_path, names=("expA",)):
    for n in names:
        d = tmp_path / "res" / n; d.mkdir(parents=True)
        for f in ("runs.csv", "blocks.csv"): shutil.copy(os.path.join(FIX, f), d / f)
    return str(tmp_path / "res")

def open_app(tmp_path, names=("expA",)):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_input(key="root_in").set_value(results_root(tmp_path, names)); at.run()
    assert at.multiselect(key="picked_w").value == []                                     # opening a results root loads nothing by itself
    at.multiselect(key="picked_w").set_value(list(at.multiselect(key="picked_w").options)).run()
    return at

def test_app_loads_and_prepares_simulated_data():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and any("Nodes kept" in m.label for m in at.metric)

def test_workspace_explorer_and_statistics(tmp_path):
    at = open_app(tmp_path)
    assert not at.exception and len(at.dataframe) >= 6            # summary, recall, recovery, explorer, tests, cost ...
    titles = [s.value for s in at.subheader]
    assert "Friedman and Holm-corrected Wilcoxon, with effect sizes" in titles and "Block-F1 over time" in titles

def test_two_experiments_side_by_side_and_sensitivity(tmp_path):
    at = open_app(tmp_path, ("expA", "expB"))
    at.multiselect(key="picked_w").set_value([p for p in at.multiselect(key="picked_w").options]).run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "Experiments"] == ["2"]      # two experiments loaded
    assert "Sensitivity" in [s.value for s in at.subheader]

def test_export_bundle_builds(tmp_path):
    at = open_app(tmp_path)
    [b for b in at.button if b.label == "Build export bundle"][0].click().run()
    assert not at.exception and len(at.session_state["bundle"]) > 1000 and len(at.session_state["xlsx"]) > 1000

def test_dataset_export_builds():
    at = AppTest.from_file(APP, default_timeout=200).run()
    [b for b in at.button if b.label == "Build dataset"][0].click().run()
    assert not at.exception and len(at.session_state["dataset"]) > 10000

def test_app_runs_a_small_experiment_and_loads_it(tmp_path):
    at = AppTest.from_file(APP, default_timeout=280).run()
    ms = {m.label: m for m in at.sidebar.multiselect}
    ms["Scenarios"].set_value(["S0"]); ms["Models"].set_value(["M1", "M3"])
    [n for n in at.sidebar.number_input if n.label == "Seeds per scenario"][0].set_value(1)
    at.sidebar.text_input[-1].set_value(str(tmp_path / "out" / "r1"))
    at.run(); [b for b in at.button if b.label == "Run experiments"][0].click().run()
    assert not at.exception
    assert set(pd.read_csv(tmp_path / "out" / "r1" / "runs.csv").model) == {"M1", "M3"}
    assert at.session_state["picked_w"] == [str(tmp_path / "out" / "r1")]     # run is preselected in the Workspace

def test_live_view_renders_a_frame_and_changes_with_position():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    assert any(b.label == "Play from here" for b in at.button)          # replay built, controls present
    s = at.select_slider(key="lv_speed"); assert s.value == 6
    sl = at.slider(key="lv_pos"); first = sl.value
    sl.set_value(sl.max).run()
    assert not at.exception and at.slider(key="lv_pos").value == sl.max != first

# --- live engine
def test_live_frame_status_and_f1():
    import numpy as np
    from wsnfd.config import Config
    from wsnfd import run as R, live as L
    cfg = Config(grid_s=600, n_estimators=10); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=0))
    rp = L.build_replay(X, info, cfg, "S2", 0, ["M1"], info.get("neighbours"))
    t = rp["onset"] + 5; s = L.frame_status(rp, "M1", t)
    assert s.shape == (rp["N"],) and set(np.unique(s)) <= {0, 1, 2, 3, 4}
    y = rp["y"][t].astype(bool); assert ((s == L.TP) | (s == L.FN)).sum() == (y & (rp["preds"]["M1"][t] >= 0)).sum()
    f = L.trailing_f1(rp, "M1", rp["T"] - 1, rp["T"]); assert np.isnan(f) or 0 <= f <= 1

def test_sandbox_tab_buttons_drive_the_network():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.button(key="sb_create").click().run(); assert not at.exception
    sbo = at.session_state["sb"]; n0 = len(sbo.nodes)
    at.number_input(key="sb_add_n").set_value(7).run(); at.button(key="sb_addb").click().run(); assert len(at.session_state["sb"].nodes) == n0 + 7
    for _ in range(14): at.button(key="sb_stepb").click().run()
    assert not at.exception and at.session_state["sb"].stats()["ticks"] == 14
    ids = list(at.session_state["sb"].nodes)[:3]; at.multiselect(key="sb_targets").set_value(ids).run()
    at.button(key="sb_f_bias").click().run(); assert all("bias" in at.session_state["sb"].nodes[i].faults for i in ids)
    at.button(key="sb_offb").click().run(); assert not any(at.session_state["sb"].nodes[i].on for i in ids)
    at.button(key="sb_delb").click().run(); assert not (set(ids) & set(at.session_state["sb"].nodes))
    at.button(key="sb_cd").click().run(); assert at.session_state["sb"].stats()["recorded_rows"] == 0 and not at.exception
    at.button(key="sb_delallb").click().run(); assert len(at.session_state["sb"].nodes) == 0 and not at.exception

def test_intel_source_accepts_a_folder_and_finds_locations_next_to_it(tmp_path):
    import sys; sys.path.insert(0, HERE); from test_filepick import write_intel
    write_intel(str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Intel Berkeley file").run()
    assert any("Choose the Intel" in w.value for w in at.sidebar.warning)             # nothing chosen yet: a hint, no crash
    at.text_input(key="intel_path").set_value(str(tmp_path) + os.sep).run()           # a FOLDER, as typed in the screenshot
    assert not at.exception
    assert any("Using " in s.value and "data.txt.gz" in s.value for s in at.sidebar.success)
    assert any("mote_locs.txt found next to the data file" in c.value for c in at.sidebar.caption)
    assert any("Nodes kept" in m.label for m in at.metric)                            # the real-format file went through the whole pipeline
    assert at.session_state["intel_path"] and not any("Could not prepare" in e.value for e in at.sidebar.error)
    at.text_input(key="intel_path").set_value(str(tmp_path / "nope.gz")).run()
    assert not at.exception and any("does not exist" in w.value for w in at.sidebar.warning)

def test_intel_file_without_a_location_file_still_loads(tmp_path):
    import sys; sys.path.insert(0, HERE); from test_filepick import write_intel
    f, _ = write_intel(str(tmp_path), locs=False)
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Intel Berkeley file").run()
    at.text_input(key="intel_path").set_value(f).run()
    assert not at.exception and not any("Could not prepare" in e.value for e in at.sidebar.error)
    assert any("Nodes kept" in m.label for m in at.metric)
    assert any("unlocated" in c.value for c in at.sidebar.caption)

def test_failed_selection_shows_the_report_and_relaxing_the_rule_recovers(tmp_path):
    import sys; sys.path.insert(0, HERE); from test_filepick import write_intel
    f, _ = write_intel(str(tmp_path), n_nodes=8)
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Intel Berkeley file").run(); at.text_input(key="intel_path").set_value(f).run()
    assert not at.exception and any("Selection rule not met" in e.value for e in at.error)
    assert "Nodes that would qualify under looser rules" in [s.value for s in at.subheader] and "Each node" in [s.value for s in at.subheader]
    assert any("What was read" in m.value for m in at.markdown)
    assert not any("Nodes kept" in m.label for m in at.metric)
    at.number_input(key="sel_nodes").set_value(3).run(); at.number_input(key="sel_days").set_value(21).run(); at.number_input(key="sel_cov").set_value(60).run()
    assert not at.exception and any("Nodes kept" in m.label for m in at.metric)
    assert any("Changed from the protocol" in w.value for w in at.sidebar.warning)
    assert any("--min-nodes 3" in c.value for c in at.code)                             # the equivalent command carries the relaxed rule

def test_app_passes_calibration_and_tuning_settings_to_the_run(tmp_path):
    import json
    at = AppTest.from_file(APP, default_timeout=290).run()
    ms = {m.label: m for m in at.sidebar.multiselect}
    ms["Scenarios"].set_value(["S0"]); ms["Models"].set_value(["M1"])
    [n for n in at.sidebar.number_input if n.label == "Seeds per scenario"][0].set_value(1)
    [n for n in at.sidebar.number_input if n.label.startswith("Fault magnitude")][0].set_value(1.25)
    [c for c in at.sidebar.checkbox if "--tune" in c.label][0].set_value(True)
    out = str(tmp_path / "out" / "r2"); at.sidebar.text_input[-1].set_value(out)
    at.run(); [b for b in at.button if b.label == "Run experiments"][0].click().run()
    assert not at.exception
    env = json.load(open(os.path.join(out, "environment.json")))
    assert env["config"]["mag_scale"] == 1.25
    assert "--mag-scale 1.25" in env["command"] and "--tune" in env["command"]

def test_interface_run_and_its_command_line_give_identical_results(tmp_path):
    """The command shown on the Run tab, run from the command line, reproduces the interface's detection results."""
    import json, shlex
    from wsnfd import run as R
    at = AppTest.from_file(APP, default_timeout=290).run()
    ms = {m.label: m for m in at.sidebar.multiselect}
    ms["Scenarios"].set_value(["S2"]); ms["Models"].set_value(["M1", "M2", "M3"])
    for n in at.sidebar.number_input:
        if n.label == "Seeds per scenario": n.set_value(1)
        if n.label == "Trees": n.set_value(10)
        if n.label == "Time grid (s)": n.set_value(600)
        if n.label.startswith("Fault magnitude"): n.set_value(1.25)
        if n.label.startswith("Label delay"): n.set_value(0.5)
        if n.label.startswith("Retraining period"): n.set_value(2.0)
    ui = str(tmp_path / "ui"); at.sidebar.text_input[-1].set_value(ui)
    at.run(); [b for b in at.button if b.label == "Run experiments"][0].click().run()
    assert not at.exception
    cmd = shlex.split(json.load(open(os.path.join(ui, "environment.json")))["command"])
    cli = str(tmp_path / "cli"); cmd[cmd.index("--out") + 1] = cli
    R.main(cmd[3:])
    a, b = pd.read_csv(os.path.join(ui, "runs.csv")), pd.read_csv(os.path.join(cli, "runs.csv"))
    cols = ["scenario", "seed", "model", "precision", "recall", "f1", "post_change_f1", "n_updates"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))

def test_analyse_command_line_matches_the_tables_of_the_interface(tmp_path, capsys):
    from wsnfd import analyse as A
    root = results_root(tmp_path, ("expA",)); folder = os.path.join(root, "expA")
    runs, _ = A.load_folders([folder]); out = str(tmp_path / "t.csv")
    A.main([folder, "--what", "explore", "--measure", "f1", "--by", "scenario,model", "--stats", "mean,n", "--out", out])
    pd.testing.assert_frame_equal(pd.read_csv(out), A.explore(runs, "f1", ["scenario", "model"], ["mean", "n"]), check_dtype=False)
    A.main([folder, "--what", "tests", "--out", out]); assert len(pd.read_csv(out)) == len(A.tests(runs))
    A.main([folder, "--what", "recall", "--out", out]); assert "recall_spike" in pd.read_csv(out).columns


def test_run_folder_defaults_to_next_free_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for n in ("run1", "run2"):
        d = tmp_path / "results" / n; d.mkdir(parents=True); shutil.copy(os.path.join(FIX, "runs.csv"), d / "runs.csv")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.session_state["outdir_w"] == "results/run3"                                  # the first name not yet used
    at.sidebar.text_input[-1].set_value("results/run1"); at.run()
    run_btn = [b for b in at.button if b.label == "Run experiments"][0]
    assert run_btn.disabled and any("already holds results" in w.value for w in at.sidebar.warning)
    ow = [c for c in at.sidebar.checkbox if c.label == "Overwrite this folder"][0]; ow.set_value(True).run()
    assert not [b for b in at.button if b.label == "Run experiments"][0].disabled

def test_command_line_refuses_an_existing_results_folder(tmp_path):
    import wsnfd.run as R
    d = tmp_path / "res"; d.mkdir(); shutil.copy(os.path.join(FIX, "runs.csv"), d / "runs.csv")
    with pytest.raises(SystemExit) as e: R.main(["--source", "sim", "--out", str(d), "--seeds", "1", "--scenarios", "S0", "--models", "M1"])
    assert "already holds results" in str(e.value)


def test_live_view_compares_five_models_on_two_rows():
    at = AppTest.from_file(APP, default_timeout=900).run()
    ms = {m.label: m for m in at.sidebar.multiselect}
    ms["Models"].set_value(["M1", "M2", "M3", "M4", "M4b"]); at.run()
    at.multiselect(key="lv_ms").set_value(["M1", "M2", "M3", "M4", "M4b"]).run()
    assert not at.exception and at.multiselect(key="lv_ms").value == ["M1", "M2", "M3", "M4", "M4b"]
    sl = at.slider(key="lv_pos"); sl.set_value(sl.max).run()
    assert not at.exception


def test_export_bundle_is_hidden_when_the_loaded_results_change(tmp_path):
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.text_input(key="root_in").set_value(results_root(tmp_path, ("expA", "expB"))); at.run()
    opts = list(at.multiselect(key="picked_w").options); a, b = [o for o in opts if o.endswith("expA")][0], [o for o in opts if o.endswith("expB")][0]
    at.multiselect(key="picked_w").set_value([a]).run()
    [x for x in at.button if x.label == "Build export bundle"][0].click().run()
    assert not at.exception and any("Bundle built from: expA" in c.value for c in at.caption)
    at.multiselect(key="picked_w").set_value([b]).run()
    assert any("so it is hidden" in w.value for w in at.warning) and not any("Bundle built from" in c.value for c in at.caption)

def test_adwin_input_selector_changes_the_command_and_the_run(tmp_path):
    """Default (batch) leaves the command as before; 'Reading' adds --adwin-input observation, runs with it and records it."""
    at = AppTest.from_file(APP, default_timeout=280).run()
    sel = [s for s in at.sidebar.selectbox if s.label == "ADWIN input (M3)"][0]
    assert sel.value == "Batch"
    assert not any("--adwin-input" in c.value for c in at.code)                        # the default command is unchanged
    ms = {m.label: m for m in at.sidebar.multiselect}
    ms["Scenarios"].set_value(["S0"]); ms["Models"].set_value(["M3"])
    [n for n in at.sidebar.number_input if n.label == "Seeds per scenario"][0].set_value(1)
    sel.set_value("Reading"); at.sidebar.text_input[-1].set_value(str(tmp_path / "out" / "r1")); at.run()
    assert any("--adwin-input observation" in c.value for c in at.code)
    [b for b in at.button if b.label == "Run experiments"][0].click().run()
    assert not at.exception
    import json
    assert json.load(open(tmp_path / "out" / "r1" / "environment.json"))["config"]["adwin_input"] == "observation"
