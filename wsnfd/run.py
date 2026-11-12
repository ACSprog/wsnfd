"""Experiment driver: python -m wsnfd.run --source sim|intel [--path data.txt.gz] ..."""
import argparse, json, os, platform, sys, time
import numpy as np, pandas as pd
from .config import Config, FAULT_TYPES
from . import data as D, change as C, faults as Fa, features as Fe, metrics as M
from .prequential import run_stream
from .models import tune_rf


def peak_rss_mb():
    """Peak memory of this process in MB on Linux, macOS and Windows; NaN if the platform offers no way to read it."""
    try:
        import resource                                   # Linux and macOS only
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024     # macOS reports bytes, Linux kilobytes
    except ImportError:
        pass
    try:                                                  # Windows
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb): return c.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return float("nan")


def _memorystatusex():
    """Layout of the Windows MEMORYSTATUSEX structure: two 32-bit fields, then seven 64-bit fields (64 bytes)."""
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64)] + [(n, ctypes.c_uint64) for n in (
            "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    return MS

def windows_memory_gb(kernel32=None):
    """Installed memory in GB on Windows: GlobalMemoryStatusEx first, PowerShell as a fallback; None if neither works."""
    try:
        import ctypes
        MS = _memorystatusex(); m = MS(); m.dwLength = ctypes.sizeof(MS)
        k32 = kernel32 or ctypes.windll.kernel32
        if k32.GlobalMemoryStatusEx(ctypes.pointer(m)) and m.ullTotalPhys: return round(m.ullTotalPhys / 1e9, 1)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return round(int(out) / 1e9, 1)
    except Exception:
        return None

def hardware():
    """Processor, logical CPUs and installed memory of the machine that runs the experiments (logged with every result set)."""
    h = dict(machine=platform.machine(), processor=platform.processor() or "not reported", logical_cpus=os.cpu_count(), memory_gb=None)
    try:
        h["memory_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)       # Linux and macOS
    except (AttributeError, ValueError, OSError):
        h["memory_gb"] = windows_memory_gb()                                                          # Windows
    return h

TUNING_LOG = {}                                      # forest tuning scores per configuration object, written with the results

def environment_record(cfg, info):
    """What is written to environment.json by the command line and by the workbench: parameters, selection outcome, library versions, hardware."""
    import sklearn, river, scipy
    return dict(info=info, config=cfg.__dict__, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, river=river.__version__, scipy=scipy.__version__, platform=platform.platform(), hardware=hardware(), **({"tuning_scores": TUNING_LOG[id(cfg)]} if id(cfg) in TUNING_LOG else {}))

def tune_once(Xbase, cfg, neighbours=None, seed=0):
    """Choose the forest hyperparameters once, on the training segment of scenario S0 (forward-chaining CV), and write them into cfg.
    Every run then uses them unchanged, so M1, M2 and M3 differ only in their update policy."""
    X, yb, yt, onset, train_end, valid = make_stream(Xbase, cfg, "S0", seed)
    F, _ = Fe.build_features(X, train_end, cfg.feature_set, cfg.roll_window, neighbours)
    ok = valid & np.isfinite(F).all(axis=2); tr = ok[:train_end]
    log = []; best = tune_rf(F[:train_end][tr], yb[:train_end][tr], cfg, seed, log=log); TUNING_LOG[id(cfg)] = log
    cfg.n_estimators, cfg.max_depth, cfg.min_samples_leaf = best["n_estimators"], best["max_depth"], best["min_samples_leaf"]
    return best

def prepare_baseline(cfg, source="sim", path=None, sim_kwargs=None, locs=None):
    if source == "intel":
        df = D.load_intel(path); coords = D.load_locations(locs) if locs else None
    else:
        df, coords = D.simulate(grid_s=cfg.grid_s, return_coords=True, **(sim_kwargs or {}))
    ignored = pd.Series([], dtype=int)
    if source == "intel":                                  # drop corrupted mote ids (the dataset documents ids 1 to 54) and report them
        ok = df["moteid"].between(*D.INTEL_MOTE_IDS); ignored = df.loc[~ok, "moteid"]; df = df[ok]
        if df.empty: raise ValueError(f"No readings with mote ids {D.INTEL_MOTE_IDS[0]} to {D.INTEL_MOTE_IDS[1]} were found, so this does not look like the Intel data file.")
    raw = D.raw_summary(df); raw.update(ignored_rows=int(len(ignored)), ignored_ids=sorted(int(i) for i in ignored.unique())[:30])
    X, nodes, t0 = D.align(D.clean(df, bounds=cfg.bounds(), dead_voltage=cfg.dead_voltage), cfg.grid_s)
    X = D.fill_short_gaps(X, cfg.max_gap_bins)
    sel = D.select_nodes_period(X, cfg.grid_s, cfg.min_nodes, cfg.min_days, cfg.min_coverage)
    info = dict(source=source, selection_rule_passed=sel is not None, rows_read=raw["rows"], motes_in_file=raw["motes"], ignored_rows_bad_moteid=raw["ignored_rows"])
    if sel is None:
        oor, ctbl = D.clean_report(df, bounds=cfg.bounds(), dead_voltage=cfg.dead_voltage)
        info["diagnostics"] = D.diagnose_selection(X, nodes, t0, cfg, raw, ctbl, oor)
        return None, info
    ok_nodes, a, b = sel
    X = X[a:b][:, ok_nodes]
    info.update(t0=str(t0), start_bin=int(a), grid_s=cfg.grid_s)
    chosen = [int(n) for n in nodes[ok_nodes]]
    if coords and all(n in coords for n in chosen):
        info["xy"] = [[float(coords[n][0]), float(coords[n][1])] for n in chosen]
        info["neighbours"] = D.knn_neighbours(coords, chosen, cfg.knn)
    info.update(n_nodes=len(ok_nodes), n_epochs=int(b - a), nodes=[int(n) for n in nodes[ok_nodes]])
    return X, info

def make_stream(Xbase, cfg, scenario, seed):
    """Apply the controlled change, then inject faults (training and test segments). Order as in Figure 4.1.
    Returns X, binary labels, type labels, onset index, train_end, valid mask."""
    T = Xbase.shape[0]; train_end = int(cfg.train_frac * T)
    valid = np.isfinite(Xbase).all(axis=2)
    X, onset = C.apply_change(Xbase, scenario, train_end, cfg)
    rng = np.random.default_rng(seed); sigma = Fa.node_sigma(Xbase, train_end)
    yb = np.zeros(valid.shape, np.int8); yt = np.zeros(valid.shape, np.int8)
    for seg in [(0, train_end), (train_end, T)]:
        b, t = Fa.inject(X, valid, sigma, seg, rng, cfg); yb |= b; yt |= t
    return X, yb, yt, onset, train_end, valid

def one_run(Xbase, cfg, scenario, seed, model_names, tune=False, neighbours=None):
    T = Xbase.shape[0]
    X, yb, yt, onset, train_end, valid = make_stream(Xbase, cfg, scenario, seed)
    F, names = Fe.build_features(X, train_end, cfg.feature_set, cfg.roll_window, neighbours)   # ... features last
    ok = valid & np.isfinite(F).all(axis=2)
    res = run_stream(F, yb, yt, ok, train_end, cfg, model_names, seed, tune)
    meta, B = res["meta"], int(cfg.block_days * cfg.epochs_per_day())
    nblocks = int(np.ceil((T - train_end) / B)); onset_block = (onset - train_end) // B
    rows, curves = [], []
    for n, p in res["pred"].items():
        y = meta[:, 2]; s = M.scores(y, p); m = res["models"][n]
        bf = M.block_f1(meta[:, 0], y, p, train_end, B, nblocks); nobs, npos = M.block_counts(meta[:, 0], y, train_end, B, nblocks)
        row = dict(scenario=scenario, seed=seed, model=n, onset_block=int(onset_block), feature_set=cfg.feature_set, **s, **M.adaptation(bf, onset_block, usable=M.usable_blocks(nobs)),
                   **M.per_type_recall(meta[:, 3], y, p, FAULT_TYPES), **(M.alarm_stats(m.alarms, onset) if m.detector else dict(n_alarms=np.nan, false_alarms=np.nan, detection_delay=np.nan)),
                   fit_time_s=m.fit_time, init_fit_time_s=getattr(m, "fit_time_init", np.nan), predict_time_s=m.predict_time, predict_time_per_obs_s=m.predict_time / max(1, len(y)),
                   n_updates=m.n_updates, n_train_rows=m.n_train_rows, labels_consumed=m.labels_consumed, model_bytes=m.model_bytes(),
                   label_delay_days=cfg.label_delay_days,
                   peak_rss_mb=peak_rss_mb())  # process-level (shared by models in a run)
        rows.append(row); curves += [dict(scenario=scenario, seed=seed, model=n, block=i, f1=v, n_obs=int(nobs[i]), n_faults=int(npos[i])) for i, v in enumerate(bf)]
    return rows, curves

def dataclasses_replace_feature(cfg, fs):
    import dataclasses; return dataclasses.replace(cfg, feature_set=fs)

def add_block_counts(folder, Xbase, cfg, neighbours=None):
    """Write the number of observations scored in each block into an older results folder (blocks.csv gains n_obs).
    The counts depend only on the prepared data, the grid and the feature set, so they are rebuilt without rerunning any model.
    Every run is checked: the counts must add up to that run's confusion counts (tp+fp+fn+tn) and the blocks must match; otherwise nothing is written."""
    import dataclasses
    runs = pd.read_csv(f"{folder}/runs.csv"); blocks = pd.read_csv(f"{folder}/blocks.csv")
    if "n_obs" in blocks.columns: return "This folder already has block counts."
    B = int(cfg.block_days * cfg.epochs_per_day()); cache = {}
    for fs in runs.feature_set.unique():
        c = dataclasses.replace(cfg, feature_set=fs)
        X, yb, yt, onset, train_end, valid = make_stream(Xbase, c, "S0", 0)
        F, _ = Fe.build_features(X, train_end, fs, c.roll_window, neighbours)
        ok = valid & np.isfinite(F).all(axis=2); T = Xbase.shape[0]
        nb = int(np.ceil((T - train_end) / B))
        idx = (np.nonzero(ok)[0] - train_end) // B; idx = idx[(idx >= 0) & (idx < nb)]
        cache[fs] = (np.bincount(idx, minlength=nb), int((onset - train_end) // B))
    for _, r in runs.iterrows():
        n, ob = cache[r.feature_set]
        if ob != int(r.onset_block): raise ValueError(f"The settings do not match this folder: onset block {ob} here, {int(r.onset_block)} in the results. Use the same time grid, selection rule and cleaning as the run.")
        d = blocks[(blocks.scenario == r.scenario) & (blocks.seed == r.seed) & (blocks.model == r.model)]
        if len(d) != len(n): raise ValueError(f"The settings do not match this folder: {len(n)} blocks here, {len(d)} in the results. Use the same time grid, selection rule and cleaning as the run.")
        if int(n.sum()) != int(r.tp + r.fp + r.fn + r.tn): raise ValueError(f"The settings do not match this folder: {int(n.sum())} observations here, {int(r.tp + r.fp + r.fn + r.tn)} in the results (scenario {r.scenario}, seed {int(r.seed)}, model {r.model}). Use the same data, grid, selection rule, cleaning and feature set as the run.")
    if len(cache) != 1: raise ValueError("Results with several feature sets in one folder are not supported here.")
    n = next(iter(cache.values()))[0]
    blocks = blocks.merge(pd.DataFrame({"block": np.arange(len(n)), "n_obs": n}), on="block", how="left")
    os.replace(f"{folder}/blocks.csv", f"{folder}/blocks_before_counts.csv"); blocks.to_csv(f"{folder}/blocks.csv", index=False)
    return f"Added block counts to {len(runs)} runs. The original curves are kept as blocks_before_counts.csv."

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim"); ap.add_argument("--path"); ap.add_argument("--out", default="results")
    ap.add_argument("--scenarios", default="S0,S1,S2,S3,S4"); ap.add_argument("--models", default="M1,M2,M3")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--feature-set", default="FS-A")
    ap.add_argument("--grid-s", type=int, default=31); ap.add_argument("--delay-days", type=float, default=0.0)
    ap.add_argument("--fault-rate", type=float, default=0.10); ap.add_argument("--tune", action="store_true"); ap.add_argument("--tune-only", action="store_true", help="choose the forest settings, print and save every combination's score to OUT/tuning_scores.csv, and stop (no experiment is run)"); ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--locs", help="Intel mote_locs.txt (enables nearest-neighbour features)"); ap.add_argument("--sim-nodes", type=int, default=30); ap.add_argument("--sim-days", type=int, default=36)
    ap.add_argument("--min-nodes", type=int, default=10); ap.add_argument("--min-days", type=float, default=28); ap.add_argument("--min-coverage", type=float, default=0.80); ap.add_argument("--dead-voltage", type=float, default=2.2, help="battery cut-off in volts (cleaning)")
    ap.add_argument("--temp-range", default="10,40", help="plausible temperature range, e.g. 10,40"); ap.add_argument("--hum-range", default="5,95", help="plausible humidity range, e.g. 5,95")
    ap.add_argument("--sweep-cleaning", action="store_true", help="Intel only: compare battery cut-offs and plausibility bounds on this file and stop")
    ap.add_argument("--sim-seed", type=int, default=0, help="simulator seed (fallback source)"); ap.add_argument("--check-data", action="store_true", help="prepare the data, print the selection report and stop")
    ap.add_argument("--mag-scale", type=float, default=1.0, help="pilot calibration multiplier on fault magnitudes")
    ap.add_argument("--period-days", type=float, default=1.0, help="M2 retraining period P"); ap.add_argument("--window-days", type=float, default=3.0, help="training window W")
    ap.add_argument("--adwin-delta", type=float, default=0.002); ap.add_argument("--cooldown", type=int, default=24, help="M3 cooldown in batches")
    ap.add_argument("--cold-start", action="store_true", help="first design: M2 and M3 start with an empty retraining buffer (default: buffer prefilled with the last W days of the training segment)")
    ap.add_argument("--adwin-input", choices=["batch", "observation"], default="batch", help="what M3's ADWIN receives: one value per batch (default) or one per reading (first design)")
    ap.add_argument("--overwrite", action="store_true", help="allow --out to replace a folder that already holds results (otherwise the run stops, so earlier results are never lost)")
    ap.add_argument("--add-block-counts", metavar="FOLDER", help="add per-block observation counts to an older results folder (use the same data, grid, selection and cleaning settings as the run); no model is rerun")
    a = ap.parse_args(argv)
    if not (a.overwrite or a.check_data or a.sweep_cleaning or a.add_block_counts or a.tune_only) and os.path.exists(os.path.join(a.out, "runs.csv")):
        sys.exit(f"{a.out} already holds results. Choose another --out folder (nothing was run), or add --overwrite to replace them.")
    cfg = Config(mag_scale=a.mag_scale, period_days=a.period_days, window_days=a.window_days, adwin_delta=a.adwin_delta, cooldown_batches=a.cooldown, prefill_buffer=not a.cold_start, adwin_input=a.adwin_input, grid_s=a.grid_s, feature_set=a.feature_set, label_delay_days=a.delay_days, fault_rate=a.fault_rate, n_estimators=a.trees,
                 min_nodes=a.min_nodes, min_days=a.min_days, min_coverage=a.min_coverage, dead_voltage=a.dead_voltage,
                 temp_min=float(a.temp_range.split(',')[0]), temp_max=float(a.temp_range.split(',')[1]), hum_min=float(a.hum_range.split(',')[0]), hum_max=float(a.hum_range.split(',')[1]))
    if a.sweep_cleaning:
        df = D.load_intel(a.path) if a.source == "intel" else D.simulate(grid_s=cfg.grid_s, n_nodes=a.sim_nodes, days=a.sim_days, seed=a.sim_seed)
        if a.source == "intel": df = df[df["moteid"].between(*D.INTEL_MOTE_IDS)]
        print(D.cleaning_sweep(df, cfg).to_string(index=False)); return
    X, info = prepare_baseline(cfg, a.source, a.path, dict(n_nodes=a.sim_nodes, days=a.sim_days, seed=a.sim_seed), a.locs)
    print("selection:", {k: v for k, v in info.items() if k != "diagnostics"})
    if X is None:
        dg = info.get("diagnostics", {})
        if dg: print(f"Selection rule failed ({cfg.min_nodes} nodes, {cfg.min_days:g} days, {cfg.min_coverage:.0%} coverage). Nodes meeting it: {dg['nodes_meeting_rule']}.\nNodes that would qualify under looser rules:\n{dg['relax'].to_string()}")
        sys.exit("Selection rule failed. Relax it with --min-days / --min-coverage / --min-nodes (and say so in Chapter 3), or use --source sim as the fallback of Section 3.3.")
    if a.check_data:
        print(json.dumps({k: v for k, v in info.items() if k not in ("neighbours", "nodes", "diagnostics", "xy")}, indent=2, default=str)); return
    if a.add_block_counts:
        cfg = dataclasses_replace_feature(cfg, pd.read_csv(f"{a.add_block_counts}/runs.csv").feature_set.iloc[0])
        try: print(add_block_counts(a.add_block_counts, X, cfg, info.get("neighbours")))
        except ValueError as e: sys.exit(str(e))
        return
    os.makedirs(a.out, exist_ok=True)
    if a.tune_only:
        best = tune_once(X, cfg, info.get("neighbours")); rows = TUNING_LOG[id(cfg)]; t = pd.DataFrame(rows); t["chosen"] = [(r["n_estimators"], r["max_depth"], r["min_samples_leaf"]) == (best["n_estimators"], best["max_depth"], best["min_samples_leaf"]) for r in rows]
        t.insert(0, "feature_set", cfg.feature_set); t.to_csv(f"{a.out}/tuning_scores.csv", index=False); print(t.to_string(index=False)); return
    if a.tune:
        best = tune_once(X, cfg, info.get("neighbours")); print("hyperparameters chosen once on the training segment and used for every run:", best)
    allrows, allcurves = [], []
    for sc in a.scenarios.split(","):
        for seed in range(a.seeds):
            t = time.time(); r, c = one_run(X, cfg, sc, seed, a.models.split(","), False, info.get("neighbours")); allrows += r; allcurves += c
            print(f"{sc} seed {seed}: {time.time()-t:.1f}s", flush=True)
    pd.DataFrame(allrows).to_csv(f"{a.out}/runs.csv", index=False); pd.DataFrame(allcurves).to_csv(f"{a.out}/blocks.csv", index=False)
    json.dump(environment_record(cfg, info), open(f"{a.out}/environment.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()
