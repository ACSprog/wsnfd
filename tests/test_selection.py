import os, sys, threading, http.server, socketserver, functools, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(__file__))
from test_filepick import write_intel
from wsnfd.config import Config
from wsnfd import data as D, run as R, fetch as FE, filepick as FP

def test_clean_report_counts_out_of_range_and_the_battery_rule():
    ts = pd.date_range("2004-03-01", periods=6, freq="h")
    df = pd.DataFrame({"ts": list(ts) * 2, "moteid": [1] * 6 + [2] * 6,
                       "temperature": [20, 20, 90, 20, 20, 20] + [20] * 6, "humidity": [40.0] * 12, "light": [100.0] * 12,
                       "voltage": [2.7, 2.7, 2.7, 2.05, 2.7, 2.7] + [2.7] * 6})          # mote 1: bad temperature at row 2, voltage 2.05 at row 3
    oor, tbl = D.clean_report(df); t = tbl.set_index("moteid")
    assert oor["temperature"] == 1 and oor["voltage"] == 0
    assert t.loc[1, "rows_removed_by_battery_rule"] == 3 and t.loc[1, "battery_rule_from"] == ts[3]      # rows 3,4,5 are treated as dead
    assert t.loc[2, "rows_removed_by_battery_rule"] == 0 and pd.isna(t.loc[2, "battery_rule_from"])
    assert D.clean(df).loc[[3, 4, 5], "temperature"].isna().all() and D.clean(df).loc[[0, 1], "temperature"].notna().all()   # report agrees with clean()

@pytest.fixture(scope="module")
def failing(tmp_path_factory):
    d = tmp_path_factory.mktemp("fail"); f, _ = write_intel(str(d), n_nodes=8, days=36)
    cfg = Config(grid_s=300); X, info = R.prepare_baseline(cfg, "intel", f, None, str(d / "mote_locs.txt")); return f, d, X, info

def test_failure_report_is_consistent_with_the_rule(failing):
    f, d, X, info = failing
    assert X is None and info["selection_rule_passed"] is False and info["rows_read"] > 40000
    dg = info["diagnostics"]; rl = dg["relax"]
    assert dg["nodes_meeting_rule"] < 10 and dg["rule"] == dict(min_nodes=10, min_days=28, min_coverage=0.8, grid_s=300)
    assert rl.loc["28 days", "80%"] <= dg["nodes_meeting_rule"]                        # one shared window can't beat per-node best windows
    assert dg["best_common_nodes"] == rl.loc["28 days", "80%"] < 10                    # the headline number is the shared-window count the rule actually uses
    assert (rl.diff(axis=1).dropna(axis=1) >= 0).all().all() and (rl.diff(axis=0).dropna(axis=0) >= 0).all().all()   # looser rule never loses nodes
    n = dg["nodes"]; assert len(n) <= 8 and n["overall_coverage"].between(0, 1).all() and n["meets_rule_in_some_window"].sum() == dg["nodes_meeting_rule"]
    assert 30 <= dg["span_days"] <= 40 and dg["raw"]["motes"] == 8 and len(dg["cleaning"]) == 8

def test_relaxed_rule_lets_the_same_file_through(failing):
    f, d, X0, info0 = failing
    cfg = Config(grid_s=300, min_nodes=3, min_days=21, min_coverage=0.6); X, info = R.prepare_baseline(cfg, "intel", f, None, str(d / "mote_locs.txt"))
    assert X is not None and info["n_nodes"] >= 3 and info["xy"] and "diagnostics" not in info

def test_short_data_is_reported_as_too_short(tmp_path):
    f, _ = write_intel(str(tmp_path), n_nodes=30, days=20); X, info = R.prepare_baseline(Config(grid_s=300), "intel", f, None, None)
    dg = info["diagnostics"]; assert X is None and dg["span_days"] < 22 and (dg["relax"].loc["28 days"] == 0).all() and (dg["relax"].loc["14 days"] > 0).any()

def test_command_line_failure_prints_what_would_qualify(failing, capsys):
    f, d, *_ = failing
    with pytest.raises(SystemExit) as e: R.main(["--source", "intel", "--path", f, "--grid-s", "300", "--out", str(d / "o")])
    out = capsys.readouterr().out; assert "looser rules" in out and "28 days" in out and "80%" in out and "--min-days" in str(e.value)
    X, info = R.prepare_baseline(Config(grid_s=300, min_nodes=3, min_days=21, min_coverage=0.6), "intel", f, None, None); assert X is not None     # the flags it names exist and work

# ---------------------------------------------------------------- downloading mote_locs.txt without internet: a local server stands in for the Intel site
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "www"; root.mkdir()
    h = functools.partial(Quiet, directory=str(root)); srv = socketserver.TCPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start(); yield root, f"http://127.0.0.1:{srv.server_address[1]}"; srv.shutdown(); srv.server_close()

def test_fetch_locs_saves_a_valid_file_and_the_app_can_use_it(server, tmp_path):
    root, base = server; (root / "mote_locs.txt").write_text("".join(f"{i} {i * 1.5:.2f} {40 - i:.2f}\n" for i in range(1, 55)))
    out, msg = FE.fetch_locs(str(tmp_path / "data"), base + "/mote_locs.txt")
    assert out and os.path.basename(out) == "mote_locs.txt" and "54 motes" in msg and os.listdir(tmp_path / "data") == ["mote_locs.txt"]
    c = D.load_locations(out); assert len(c) == 54 and c[2] == (3.0, 38.0)
    assert FP.resolve_locs("", str(tmp_path / "data" / "data.txt.gz"))[0] == out          # found automatically next to the data file

def test_fetch_locs_refuses_error_pages_unreachable_servers_and_leaves_nothing_behind(server, tmp_path):
    root, base = server; dest = str(tmp_path / "d")
    (root / "page.txt").write_text("<html><body>403 Forbidden</body></html>"); p, msg = FE.fetch_locs(dest, base + "/page.txt")
    assert p is None and "not a valid" in msg or "not a location" in msg
    (root / "short.txt").write_text("1 0 0\n2 1 1\n"); p, msg = FE.fetch_locs(dest, base + "/short.txt"); assert p is None and "expected one per mote" in msg
    (root / "bin.txt").write_bytes(b"\xff\xfe\x00\x01"); assert FE.fetch_locs(dest, base + "/bin.txt")[0] is None
    p, msg = FE.fetch_locs(dest, base + "/missing.txt"); assert p is None and "Ctrl+S" in msg
    p, msg = FE.fetch_locs(dest, "http://127.0.0.1:9/mote_locs.txt", timeout=2); assert p is None and "browser" in msg
    assert not os.path.exists(dest) or os.listdir(dest) == []


def test_suggestions_name_rules_that_really_reach_the_required_nodes(tmp_path):
    f, _ = write_intel(str(tmp_path), n_nodes=30, days=36, seed=2)
    _, i0 = R.prepare_baseline(Config(grid_s=300, min_nodes=99, min_days=28, min_coverage=0.9), "intel", f, None, None)   # impossible rule: just to read the grid
    need = int(i0["diagnostics"]["relax"].loc["28 days", "90%"]) + 1                  # one more node than the strict rule can find: fails by construction
    cfg = Config(grid_s=300, min_nodes=need, min_days=28, min_coverage=0.9); X, info = R.prepare_baseline(cfg, "intel", f, None, None); assert X is None
    dg = info["diagnostics"]; assert dg["suggestions"], "a looser rule should reach one more node"
    for days, cov, n in dg["suggestions"]:
        assert n >= need and dg["relax"].loc[f"{days} days", f"{int(round(cov * 100))}%"] == n
        X2, i2 = R.prepare_baseline(Config(grid_s=300, min_nodes=need, min_days=days, min_coverage=cov), "intel", f, None, None)   # following the advice works
        assert X2 is not None and i2["n_nodes"] >= need
    days, cov, n = dg["suggestions"][0]; stricter = [c for c in (0.9, 0.8, 0.7, 0.6, 0.5) if c > cov]      # 'strictest coverage' really is the strictest
    assert all(dg["relax"].loc[f"{days} days", f"{int(round(c * 100))}%"] < need for c in stricter)

def test_corrupted_mote_ids_are_ignored_reported_and_do_not_change_the_selection(tmp_path):
    clean_dir, dirty_dir = tmp_path / "c", tmp_path / "d"; clean_dir.mkdir(); dirty_dir.mkdir()
    fc, _ = write_intel(str(clean_dir), n_nodes=12, days=36); fd, _ = write_intel(str(dirty_dir), n_nodes=12, days=36, bad_ids=True)
    Xc, ic = R.prepare_baseline(Config(grid_s=300), "intel", fc, None, None); Xd, idd = R.prepare_baseline(Config(grid_s=300), "intel", fd, None, None)
    assert idd["ignored_rows_bad_moteid"] == 48 and ic["ignored_rows_bad_moteid"] == 0
    assert Xc is not None and Xd is not None and idd["nodes"] == ic["nodes"] and np.array_equal(Xc, Xd, equal_nan=True)     # identical result
    Xf, f2 = R.prepare_baseline(Config(grid_s=300, min_nodes=40), "intel", fd, None, None)
    raw = f2["diagnostics"]["raw"]; assert raw["ignored_rows"] == 48 and raw["ignored_ids"] == [0, 55, 56, 61, 99, 1234] and raw["motes"] == 12

def test_a_file_that_is_not_the_intel_data_gets_a_clear_message(tmp_path):
    f = tmp_path / "other.txt"; f.write_text("2004-03-01 00:00:00.0 1 77 20 40 100 2.7\n2004-03-01 00:00:31.0 2 88 20 40 100 2.7\n")
    with pytest.raises(ValueError, match="does not look like the Intel data file"): R.prepare_baseline(Config(grid_s=300), "intel", str(f), None, None)


def test_rule_tries_exactly_min_days_as_its_last_window():
    """Regression: with 37 days of data the old loop tried 37 and 30 days but never 28, so the rule silently demanded 30+ days."""
    epd = 24; T = 37 * epd; X = np.full((T, 12, 4), np.nan)
    X[5 * epd:33 * epd, :, :] = 20.0                                                    # twelve nodes, present for exactly 28 days
    sel = D.select_nodes_period(X, 3600, min_nodes=10, min_days=28, min_cov=0.99)
    assert sel is not None and len(sel[0]) == 12 and sel[2] - sel[1] == 28 * epd and sel[1] == 5 * epd
    assert D.select_nodes_period(X, 3600, min_nodes=10, min_days=29, min_cov=0.99) is None      # one day longer than the data really supports
    full = np.full((T, 12, 4), 20.0); s2 = D.select_nodes_period(full, 3600, 10, 28, 0.99); assert s2[2] - s2[1] == T   # still prefers the longest window
    assert D.select_nodes_period(X[: 20 * epd], 3600, 10, 28, 0.5) is None                      # record shorter than min_days

def test_report_and_rule_agree_on_every_window_length(tmp_path):
    f, _ = write_intel(str(tmp_path), n_nodes=30, days=36, seed=4)
    _, info = R.prepare_baseline(Config(grid_s=300, min_nodes=99), "intel", f, None, None); rl = info["diagnostics"]["relax"]
    for days in (28, 21, 14, 7):
        for cov in (0.9, 0.8, 0.7, 0.6, 0.5):
            n = int(rl.loc[f"{days} days", f"{int(round(cov * 100))}%"])
            if n < 1: continue
            Xp, ip = R.prepare_baseline(Config(grid_s=300, min_nodes=n, min_days=days, min_coverage=cov), "intel", f, None, None)
            assert Xp is not None and ip["n_nodes"] >= n, (days, cov, n)                       # what the report promises, the rule delivers
            Xq, iq = R.prepare_baseline(Config(grid_s=300, min_nodes=n + 1, min_days=days, min_coverage=cov), "intel", f, None, None)
            assert Xq is None or iq["n_nodes"] >= n + 1, (days, cov, n)                        # and it is the most the rule can find
