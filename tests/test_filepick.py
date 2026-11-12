import gzip, os, pandas as pd, numpy as np
from wsnfd import filepick as FP, data as D

def write_intel(folder, gz=True, bad_lines=True, n_nodes=30, days=36, grid_s=300, seed=0, locs=True, bad_ids=False):
    """A file in the exact Intel layout: 'date time epoch moteid temperature humidity light voltage', plus mote_locs.txt."""
    df, coords = D.simulate(n_nodes=n_nodes, days=days, grid_s=grid_s, seed=seed, return_coords=True)
    lines = []; ep = {}
    for r in df.itertuples():
        ep[r.moteid] = ep.get(r.moteid, 0) + 1
        lines.append(f"{r.ts:%Y-%m-%d} {r.ts:%H:%M:%S.%f} {ep[r.moteid]} {r.moteid} {r.temperature:.4f} {r.humidity:.4f} {r.light:.2f} {r.voltage:.5f}")
    if bad_lines: lines[5:5] = ["2004-03-01 12:00:00.000 1 3 19.5", "not a data row at all", "2004-03-01 12:00:01.000 1 3 19.5 40 100 2.7 extra extra"]
    if bad_ids: lines[10:10] = [f"2004-03-01 12:{i:02d}:00.000 {i} {mid} {20 + i % 7}.5 40.5 100.0 2.7" for i, mid in enumerate([0, 55, 56, 61, 99, 99, 99, 1234] * 6)]   # 48 rows, unique timestamps
    f = os.path.join(folder, "data.txt.gz" if gz else "data.txt"); txt = "\n".join(lines) + "\n"
    (gzip.open(f, "wt") if gz else open(f, "w")).write(txt)
    if locs: open(os.path.join(folder, "mote_locs.txt"), "w").write("".join(f"{k} {x:.2f} {y:.2f}\n" for k, (x, y) in coords.items()))
    return f, df

def test_resolve_intel_accepts_file_folder_quotes_and_rejects_the_rest(tmp_path):
    f = tmp_path / "data.txt.gz"; f.write_bytes(b"x")
    assert FP.resolve_intel(str(f))[0] == str(f)
    assert FP.resolve_intel(f'  "{f}"  ')[0] == str(f)                               # Windows 'Copy as path'
    p, msg = FP.resolve_intel(str(tmp_path) + os.sep); assert p == str(f) and "Found" in msg   # folder, as pasted in the screenshot
    sub = tmp_path / "x" / "y"; sub.mkdir(parents=True); (sub / "data.txt").write_text("1"); (tmp_path / "data.txt.gz").unlink()
    assert FP.resolve_intel(str(tmp_path))[0] == str(sub / "data.txt")              # found two levels down
    empty = tmp_path / "empty"; empty.mkdir(); assert FP.resolve_intel(str(empty))[0] is None
    assert FP.resolve_intel(str(tmp_path / "nope.gz"))[0] is None and FP.resolve_intel("")[0] is None

def test_resolve_locs_prefers_given_path_then_the_sibling_file(tmp_path):
    d = tmp_path / "a"; d.mkdir(); data = d / "data.txt.gz"; data.write_bytes(b"x"); loc = d / "mote_locs.txt"; loc.write_text("1 0 0\n")
    assert FP.resolve_locs("", str(data))[0] == str(loc)
    other = tmp_path / "o.txt"; other.write_text("1 0 0"); assert FP.resolve_locs(str(other), str(data))[0] == str(other)
    assert FP.resolve_locs(str(tmp_path / "missing.txt"), str(data))[0] is None
    loc.unlink(); p, msg = FP.resolve_locs("", str(data)); assert p is None and "unlocated" in msg

def test_file_window_plumbing_returns_path_cancel_or_error_without_raising():
    ok = "import sys; sys.stdout.write(r'C:\\Users\\Amos Sikazwe\\data.txt.gz')"
    assert FP.pick_file("t", _script=ok) == (r"C:\Users\Amos Sikazwe\data.txt.gz", "")
    p, msg = FP.pick_file("t", _script="import sys; sys.stdout.write('')"); assert p is None and "No file" in msg          # user pressed Cancel
    p, msg = FP.pick_file("t", _script="raise SystemExit('no display')"); assert p is None and "could not be opened" in msg and "upload" in msg
    p, msg = FP.pick_file("t", _script="import time; time.sleep(5)", timeout=0.3); assert p is None and "could not be opened" in msg   # hung dialog
    p, msg = FP.pick_file("t", _script="import sys; sys.stdout.write(sys.argv[1] + '|' + sys.argv[2])", patterns="*.gz;*.txt")
    assert p == "t|*.gz;*.txt"                                                      # arguments reach the dialog script

def test_default_dialog_degrades_gracefully_when_tkinter_is_missing_or_headless():
    p, msg = FP.pick_file("t", timeout=60)
    assert p is None and isinstance(msg, str) and msg                               # here: no tkinter/display; on Windows it opens the window

def test_save_upload_writes_bytes_and_file_stamp_tracks_changes(tmp_path):
    class Up:
        name = r"..\evil/data.txt.gz"
        def getbuffer(self): return memoryview(b"abc123")
    out = FP.save_upload(Up(), str(tmp_path / "uploads")); assert open(out, "rb").read() == b"abc123" and os.path.dirname(out) == str(tmp_path / "uploads")
    s1 = FP.file_stamp(out); open(out, "ab").write(b"more"); assert FP.file_stamp(out) != s1 and FP.file_stamp(str(tmp_path / "none")) == (0, 0) and FP.file_stamp(None) == (0, 0) and FP.file_stamp("") == (0, 0)

def test_fast_loader_matches_the_tolerant_loader_and_skips_bad_rows(tmp_path):
    f, df = write_intel(str(tmp_path), n_nodes=12, days=3)
    fast = D.load_intel(f)
    slow = pd.read_csv(f, sep=r"\s+", names=D.INTEL_COLS, engine="python", on_bad_lines="skip")
    assert len(fast) <= len(slow) and len(fast) >= len(df) * 0.99                   # malformed rows dropped, good rows kept
    assert fast.moteid.between(1, 12).all() and fast.temperature.notna().mean() > 0.99
    a = fast.drop_duplicates(["ts", "moteid"]).set_index(["ts", "moteid"]).temperature
    assert np.isclose(a.iloc[0], df.sort_values(["ts", "moteid"]).temperature.iloc[0], atol=1e-3)
