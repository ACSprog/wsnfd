"""wsnfd workbench: data, experiments, results workspace, explorer, statistics and export.  Run:  streamlit run app.py
Everything here is also available from the command line (python -m wsnfd.run / .report / .export) and as plain Python (wsnfd.analyse)."""
import os, re, json, time, tempfile
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsnfd.config import Config
from wsnfd import run as R, analyse as A, plots as P, report as RP, export as EX, live as LV, interop as IO, sandbox as SBX, filepick as FP, fetch as FE

st.set_page_config(page_title="wsnfd workbench", layout="wide")
st.title("Adaptive fault detection: experiment and analysis workbench")
st.caption("Configure and run experiments, load results side by side, explore any measure, test, and export. All outputs are reproducible from the command line.")

SS = st.session_state
def dl_csv(df, name, key):
    st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=key)
def show(df, name, key, digits=4):
    st.dataframe(df.round(digits) if len(df) else df, hide_index=True, width="stretch")
    if len(df): dl_csv(df, name, key)
def show_fig(fig, name, key):
    png = P.to_png(fig, 200) if not isinstance(fig, bytes) else fig
    st.image(png); st.download_button(f"Download {name}.png", png, f"{name}.png", "image/png", key=key)

# ------------------------------------------------------------------ choosing the Intel files
def cb_browse(kind):
    if kind == "intel":
        p, msg = FP.pick_file("Choose the Intel data file (data.txt.gz or data.txt)", "*.gz;*.txt", os.path.dirname(FP.clean_path(SS.get("intel_path", ""))) or "")
        if p:
            SS["intel_path"] = p
            if not SS.get("locs_path"):
                f, _ = FP.resolve_locs("", p); SS["locs_path"] = f or ""
    else:
        p, msg = FP.pick_file("Choose mote_locs.txt", "*.txt", os.path.dirname(FP.clean_path(SS.get("intel_path", ""))) or "")
        if p: SS["locs_path"] = p
    SS["pick_msg"] = "" if p else msg

def cb_fetch_locs():
    d = os.path.dirname(FP.resolve_intel(SS.get("intel_path", ""))[0] or "") or os.path.join(os.getcwd(), "data")
    p, msg = FE.fetch_locs(d)
    if p: SS["locs_path"] = p
    SS["pick_msg"] = "" if p else msg
    SS["fetch_ok"] = msg if p else ""

def cb_upload(kind):
    f = SS.get("up_intel" if kind == "intel" else "up_locs")
    if f is not None:
        SS["intel_path" if kind == "intel" else "locs_path"] = FP.save_upload(f, os.path.join(os.getcwd(), "uploads"))

# ------------------------------------------------------------------ sidebar: experiment settings
def next_free_run(root="results"):
    """results/run1, run2, ...: the first name not yet used."""
    used = [int(m.group(1)) for d in (os.listdir(root) if os.path.isdir(root) else []) for m in [re.fullmatch(r"run(\d+)", d)] if m]
    return f"{root}/run{max(used, default=0) + 1}"
if SS.pop("bump_outdir", False) or "outdir_w" not in SS: SS["outdir_w"] = next_free_run()

with st.sidebar:
    st.header("Experiment settings")
    st.subheader("Data")
    source = st.radio("Source", ["Simulated (fallback)", "Intel Berkeley file"])
    path = locs = None; sim = {}; sim_seed = 0
    if source == "Intel Berkeley file":
        st.text_input("Intel data file (data.txt.gz or data.txt)", key="intel_path", help="Paste a path or a folder, click Browse, or upload the file below.")
        st.button("Browse for the data file...", on_click=cb_browse, args=("intel",), key="br_intel")
        st.file_uploader("...or upload the data file", type=["gz", "txt"], key="up_intel", on_change=cb_upload, args=("intel",), help="Copied into an 'uploads' folder next to the app.")
        path, pmsg = FP.resolve_intel(SS.get("intel_path", ""))
        if path: st.success(f"Using {path} ({os.path.getsize(path)/1e6:.1f} MB)" + (f". {pmsg}" if pmsg else ""))
        else: st.warning(pmsg)
        st.text_input("mote_locs.txt (optional, gives node positions)", key="locs_path")
        lb1, lb2 = st.columns(2)
        lb1.button("Browse...", on_click=cb_browse, args=("locs",), key="br_locs")
        lb2.button("Download it", on_click=cb_fetch_locs, key="dl_locs", help="Fetches mote_locs.txt from the Intel Lab site (needs internet) and saves it next to the data file.")
        st.file_uploader("...or upload mote_locs.txt", type=["txt"], key="up_locs", on_change=cb_upload, args=("locs",))
        locs, lmsg = FP.resolve_locs(SS.get("locs_path", ""), path)
        (st.caption if locs or not SS.get("locs_path") else st.warning)((f"Using {locs}. " if locs else "") + lmsg)
        if SS.get("pick_msg"): st.warning(SS["pick_msg"])
        if SS.get("fetch_ok"): st.success(SS["fetch_ok"])
    else:
        sim = dict(n_nodes=int(st.number_input("Nodes", 12, 60, 30, help="The realistic simulator loses nodes to death, late start and packet loss, so simulate more than the 10 the rule needs.")),
                   days=int(st.number_input("Days", 28, 45, 36)))
        sim_seed = int(st.number_input("Simulator seed", 0, 999, 0))
    grid_s = int(st.number_input("Time grid (s)", 1, 3600, 300, 1, help="31 s is the Intel sampling interval; coarser grids run much faster. Any value is accepted, as on the command line."))
    with st.expander("Selection rule (Chapter 3)"):
        sel_nodes = int(st.number_input("Minimum nodes", 2, 60, 10, key="sel_nodes")); sel_days = float(st.number_input("Minimum days", 1, 60, 28, key="sel_days"))
        sel_cov = float(st.number_input("Minimum coverage (%)", 10, 100, 80, key="sel_cov")) / 100
        sel_vcut = float(st.number_input("Battery cut-off (V)", 2.0, 3.0, 2.2, 0.05, key="sel_vcut", help="Cleaning: a mote is treated as degraded from its first reading below this voltage. On the full Intel file, 2.2 V is the strictest cut-off (of 2.1 to 2.4 V) at which the selection rule is still met."))
        t1, t2 = st.columns(2); sel_tmin = float(t1.number_input("Temperature min (C)", -20.0, 60.0, 10.0, 1.0, key="sel_tmin")); sel_tmax = float(t2.number_input("Temperature max (C)", -20.0, 130.0, 40.0, 1.0, key="sel_tmax"))
        h1, h2 = st.columns(2); sel_hmin = float(h1.number_input("Humidity min (%)", 0.0, 50.0, 5.0, 1.0, key="sel_hmin")); sel_hmax = float(h2.number_input("Humidity max (%)", 50.0, 100.0, 95.0, 1.0, key="sel_hmax"))
        if (sel_vcut, sel_tmin, sel_tmax, sel_hmin, sel_hmax) != (2.2, 10.0, 40.0, 5.0, 95.0): st.warning("Cleaning changed from the protocol (2.2 V; 10-40 C; 5-95 %). State the change and the reason in Chapter 3.")
        if (sel_nodes, sel_days, round(sel_cov, 2)) != (10, 28.0, 0.8): st.warning("Changed from the protocol (10 nodes, 28 days, 80%). State the change and the reason in Chapter 3.")
    st.subheader("Experiment")
    scenarios = st.multiselect("Scenarios", ["S0", "S1", "S2", "S3", "S4"], ["S0", "S2"])
    models = st.multiselect("Models", ["M1", "M2", "M3", "M4", "M4b"], ["M1", "M2", "M3"])
    seeds = int(st.number_input("Seeds per scenario", 1, 100, 3, 1))
    fs = st.radio("Feature set", ["FS-A", "FS-B"], horizontal=True)
    fault_rate = float(st.number_input("Fault rate", 0.01, 0.5, 0.10, 0.01, format="%.2f"))
    delay = float(st.number_input("Label delay (days)", 0.0, 30.0, 0.0, 0.5))
    mag_scale = float(st.number_input("Fault magnitude multiplier (pilot calibration)", 0.1, 5.0, 1.0, 0.05, format="%.2f", help="Scales every fault magnitude. Chosen in the pilot so that M1 on S0 is neither trivial nor saturated, then kept fixed."))
    st.subheader("Model parameters")
    tune = st.checkbox("Choose the forest settings once on the training segment (--tune)", False, help="Main runs: tick this. The chosen depth and leaf size are then used for every run and recorded. The number of trees is NOT tuned: it stays at the Trees value (use 100 for every run).")
    trees = int(st.number_input("Trees", 1, 1000, 30, 10))
    period = float(st.number_input("Retraining period P (days)", 0.1, 30.0, 1.0, 0.5))
    window = float(st.number_input("Window W (days)", 0.5, 30.0, 3.0, 0.5))
    adwin = float(st.number_input("ADWIN delta", 0.00001, 0.5, 0.002, 0.0005, format="%.5f"))
    cool = int(st.number_input("Cooldown (batches)", 0, 1000, 24, 1))
    adwin_in = "observation" if st.selectbox("ADWIN input (M3)", ["Batch", "Reading"], help="What M3's detector receives: one value per batch, the mean error of the batch (final design, the default), or one value per reading (first design). Use Reading only for the runs that test the detector input.").startswith("Reading") else "batch"
    outdir = st.text_input("Save this run to", key="outdir_w")
    out_exists = os.path.exists(os.path.join(outdir, "runs.csv"))
    overwrite = st.checkbox("Overwrite this folder", False, help="Off by default so earlier results are never replaced by accident.", key="ow_w") if out_exists else False
    if out_exists and not overwrite: st.warning(f"{outdir} already holds results. Choose another name, or tick Overwrite.")

def make_cfg():
    return Config(min_nodes=sel_nodes, min_days=sel_days, min_coverage=sel_cov, dead_voltage=sel_vcut, temp_min=sel_tmin, temp_max=sel_tmax, hum_min=sel_hmin, hum_max=sel_hmax, grid_s=grid_s, feature_set=fs, label_delay_days=delay, fault_rate=fault_rate, mag_scale=mag_scale, n_estimators=trees,
                  period_days=period, window_days=window, adwin_delta=adwin, cooldown_batches=cool, adwin_input=adwin_in)

def src_args():
    """Command-line arguments that reproduce the data settings chosen in the sidebar."""
    a = "--source " + ("intel" if source.startswith("Intel") else "sim") + (f' --path "{path}"' if path else "") + (f' --locs "{locs}"' if locs else "")
    if (sel_nodes, sel_days, round(sel_cov, 2)) != (10, 28.0, 0.8): a += f" --min-nodes {sel_nodes} --min-days {sel_days:g} --min-coverage {sel_cov:g}"
    if sel_vcut != 2.2: a += f" --dead-voltage {sel_vcut:g}"
    if (sel_tmin, sel_tmax) != (10.0, 40.0): a += f" --temp-range {sel_tmin:g},{sel_tmax:g}"
    if (sel_hmin, sel_hmax) != (5.0, 95.0): a += f" --hum-range {sel_hmin:g},{sel_hmax:g}"
    if sim: a += f" --sim-nodes {sim['n_nodes']} --sim-days {sim['days']} --sim-seed {sim_seed}"
    return a + f" --grid-s {grid_s}"

def an_cmd(what, extra=""):
    """Equivalent 'python -m wsnfd.analyse' command for the folders and filters currently loaded in the Workspace."""
    return f"python -m wsnfd.analyse {' '.join(SS.get('picked') or ['results/run1'])} --what {what}{SS.get('flt_args', '')}{extra}"

def equiv(cmd, note=""):
    with st.expander("Equivalent command line"):
        st.code(cmd, language="bash")
        if note: st.caption(note)

@st.cache_data(show_spinner="Preparing data...")
def prepare(source, path, locs, grid_s, sim, sim_seed, stamp=None, rule=(10, 28.0, 0.8, 2.2, 10.0, 40.0, 5.0, 95.0)):
    cfg = Config(grid_s=grid_s, min_nodes=rule[0], min_days=rule[1], min_coverage=rule[2], dead_voltage=rule[3], temp_min=rule[4], temp_max=rule[5], hum_min=rule[6], hum_max=rule[7])
    if source == "Intel Berkeley file": return R.prepare_baseline(cfg, "intel", path, None, locs)
    kw = dict(sim); kw["seed"] = sim_seed; return R.prepare_baseline(cfg, "sim", None, kw)

ready = source == "Simulated (fallback)" or bool(path)
Xb, info = (None, {})
if ready:
    try: Xb, info = prepare(source, path, locs, grid_s, sim, sim_seed, (FP.file_stamp(path), FP.file_stamp(locs)) if path else None, (sel_nodes, sel_days, round(sel_cov, 2), sel_vcut, sel_tmin, sel_tmax, sel_hmin, sel_hmax))
    except Exception as e: st.sidebar.error(f"Could not prepare the data: {e}")

tabs = st.tabs(["1. Data and dataset export", "2. Run", "3. Workspace", "4. Explorer", "5. Statistics", "6. Export", "7. Live view", "8. Live sandbox"])
t_data, t_run, t_ws, t_exp, t_stat, t_out, t_live, t_sb = tabs

def selection_failure(info):
    dg = info.get("diagnostics")
    if not dg: st.warning("The selection rule failed. Use the simulated fallback."); return
    r, raw = dg["rule"], dg["raw"]
    st.error(f"Selection rule not met: the best single {r['min_days']:g}-day window has only {dg['best_common_nodes']} node(s) with at least {r['min_coverage']:.0%} coverage; {r['min_nodes']} are needed.")
    if dg["nodes_meeting_rule"] > dg["best_common_nodes"]:
        st.caption(f"{dg['nodes_meeting_rule']} nodes reach {r['min_coverage']:.0%} in some {r['min_days']:g}-day window, but not in the same one: nodes stop reporting at different dates, and the rule needs them together.")
    st.write(f"**What was read:** {raw.get('rows', 0):,} readings from {raw.get('motes', 0)} motes, {raw.get('first', '?')} to {raw.get('last', '?')} "
             f"({dg['span_days']:.1f} days on the {r['grid_s']} s grid). The real Intel file has about 2.3 million readings between 28 Feb and 5 Apr 2004; "
             "if these numbers are far off, the file was not read correctly."
             + (f" Ignored {raw['ignored_rows']:,} readings with mote ids outside 1 to 54 (the dataset documents ids 1 to 54): {raw['ignored_ids']}." if raw.get("ignored_rows") else ""))
    oor = dg.get("out_of_range", {}); cl = dg.get("cleaning")
    if oor: st.write("**Removed as impossible values:** " + ", ".join(f"{k} {v:,}" for k, v in oor.items()) + ". "
                     + (f"**Battery rule** (voltage below {sel_vcut:g} V, then the node is treated as degraded): {int((cl['rows_removed_by_battery_rule'] > 0).sum())} motes affected, {int(cl['rows_removed_by_battery_rule'].sum()):,} readings removed." if cl is not None and len(cl) else ""))
    st.subheader("Nodes that would qualify under looser rules")
    st.caption("Rows are window lengths, columns are required coverage; each cell is the most nodes that meet that rule in any window. Green cells reach the number of nodes required.")
    st.dataframe(dg["relax"].style.map(lambda v: "background-color: rgba(70,170,70,0.35)" if v >= r["min_nodes"] else ""), width="stretch")
    sg = dg.get("suggestions", [])
    if sg: st.write(f"**Rules from this grid that reach {r['min_nodes']} nodes** (the strictest coverage for each window length): " + "; ".join(f"{d} days at {c:.0%} ({n} nodes)" for d, c, n in sg) + ".")
    else: st.write(f"**No rule in this grid reaches {r['min_nodes']} nodes.** Check the 'Each node' table: the data may not have that many usable nodes.")
    st.info("To relax the rule, open 'Selection rule (Chapter 3)' in the sidebar and change it. A looser rule changes your method, so record the new rule and the reason in Chapter 3.")
    st.subheader("Each node"); st.caption("meets_rule_in_some_window: the node reaches the coverage in at least one window of that length, not necessarily the same window as other nodes.")
    show(dg["nodes"], "selection_nodes", "dl_sel_nodes", 3)
    if cl is not None and len(cl):
        with st.expander("Per-mote cleaning report"): show(cl, "cleaning_report", "dl_clean_rep", 3)

# ------------------------------------------------------------------ 1. data
@st.cache_data(show_spinner="Comparing cleaning rules on the whole file (about a minute)...")
def sweep_table(source, path, grid_s, sim, sim_seed, stamp, rule):
    from wsnfd import data as D
    cfg = Config(grid_s=grid_s, min_nodes=rule[0], min_days=rule[1], min_coverage=rule[2])
    if source == "Intel Berkeley file":
        df = D.load_intel(path); df = df[df["moteid"].between(*D.INTEL_MOTE_IDS)]
    else: df = D.simulate(grid_s=grid_s, seed=sim_seed, **sim)
    return D.cleaning_sweep(df, cfg)

with t_data:
    if ready:
        with st.expander("Compare cleaning rules (battery cut-off and plausibility bounds)"):
            st.caption("Runs the same raw data through several battery cut-offs (2.1 to 2.4 V) with tight and loose plausibility bounds, and shows for each the share of temperature readings removed and how many nodes still meet the selection rule. Use it to choose the cleaning, then set it in the sidebar. The chosen values are part of your method (Chapter 3).")
            if st.button("Run comparison", key="run_sweep"): SS["sweep"] = sweep_table(source, path, grid_s, sim, sim_seed, (FP.file_stamp(path),) if path else None, (sel_nodes, sel_days, round(sel_cov, 2)))
            if "sweep" in SS: show(SS["sweep"], "cleaning_comparison", "dl_sweep")
            equiv("python -m wsnfd.run " + src_args() + " --sweep-cleaning")
    if not ready: st.info("Choose the Intel data file in the sidebar (Browse, upload, or paste its path), or switch to the simulated source.")
    elif Xb is None: selection_failure(info)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes kept", info["n_nodes"]); c2.metric("Epochs", f'{info["n_epochs"]:,}'); c3.metric("Neighbour features", "by distance" if info.get("neighbours") else "all other nodes")
        ch = st.selectbox("Channel to plot", ["temperature", "humidity", "light", "voltage"]); ci = ["temperature", "humidity", "light", "voltage"].index(ch)
        n_show = st.slider("Nodes to plot", 1, min(8, Xb.shape[1]), 3)
        fig, ax = plt.subplots(figsize=(10, 3))
        for n in range(n_show): ax.plot(Xb[:, n, ci], lw=.6, label=f"node {info['nodes'][n]}")
        ax.set_xlabel("epoch"); ax.set_ylabel(ch); ax.legend(ncol=n_show, fontsize=8); st.pyplot(fig); plt.close(fig)
        st.caption("Cleaned and aligned baseline, before change or fault injection. Gaps are missing readings.")
        equiv("python -m wsnfd.run " + src_args() + " --check-data", "Prints the selection outcome. If the rule fails, the command prints the report of looser rules shown here.")
        with st.expander("Selection details"): st.json({k: v for k, v in info.items() if k not in ("neighbours", "nodes")})
        st.subheader("Export the labelled dataset")
        st.caption("Long format, one row per node-epoch: readings as the models see them, labels, segment and (optionally) the features. Opens in Excel, SPSS, R, Python.")
        e1, e2, e3, e4 = st.columns(4)
        ds_sc = e1.selectbox("Scenario", ["baseline", "S0", "S1", "S2", "S3", "S4"], 3, help="baseline = cleaned data before any change or faults")
        ds_seed = int(e2.number_input("Seed", 0, 99, 0)); ds_feat = e3.checkbox("Include features", False); ds_test = e4.checkbox("Test stream only", False)
        est = int(np.isfinite(Xb).all(axis=2).sum() * (0.72 if ds_test else 1.0))
        ds_fmt = st.radio("File format", ["csv", "mat", "parquet"], horizontal=True, help="csv: Excel, SPSS, Stata, Origin, Tableau, Power BI, R. mat: MATLAB. parquet: compact and fast for Python, R, Power BI.")
        st.write(f"About {est:,} rows.")
        if ds_fmt == "csv" and est > IO.EXCEL_MAX_ROWS: st.warning(f"Excel holds at most {IO.EXCEL_MAX_ROWS:,} rows. Choose a coarser grid, 'test stream only', or the mat/parquet format.")
        if est > 1_500_000: st.warning("This is large for a browser download. Use a coarser grid, 'test stream only', or the command line (python -m wsnfd.export).")
        equiv(f"python -m wsnfd.export {src_args()} --scenario {ds_sc if ds_sc == 'baseline' else ds_sc} --seed {ds_seed} --feature-set {fs} --fault-rate {fault_rate} --mag-scale {mag_scale}"
              + (" --features" if ds_feat else "") + (" --test-only" if ds_test else "") + f" --out dataset.{ds_fmt}")
        if st.button("Build dataset"):
            b, ext, mime = IO.dataset_bytes(EX.build_dataset(Xb, info, make_cfg(), ds_sc, ds_seed, ds_feat, ds_test, info.get("neighbours")), ds_fmt)
            SS["dataset"], SS["dataset_mime"] = b, mime; SS["dataset_name"] = f"dataset_{ds_sc}_seed{ds_seed}{'_features' if ds_feat else ''}.{ext}"
        if "dataset" in SS: st.download_button(f"Download {SS['dataset_name']} ({len(SS['dataset'])/1e6:.1f} MB)", SS["dataset"], SS["dataset_name"], SS.get("dataset_mime", "text/csv"))

# ------------------------------------------------------------------ 2. run
with t_run:
    st.write(f"{len(scenarios)} scenarios × {seeds} seeds × {len(models)} models, feature set {fs}, label delay {delay} d.")
    if "M4" in models or "M4b" in models: st.warning("M4 and M4b learn one observation at a time and are slow (about a minute or more per run on the pilot).")
    cmd = ("python -m wsnfd.run " + src_args() + f" --scenarios {','.join(scenarios)} --models {','.join(models)} --seeds {seeds} --feature-set {fs}"
           + f" --fault-rate {fault_rate} --mag-scale {mag_scale} --delay-days {delay} --trees {trees} --period-days {period} --window-days {window} --adwin-delta {adwin} --cooldown {cool}"
           + (" --adwin-input observation" if adwin_in == "observation" else "") + (" --tune" if tune else "") + f" --out {outdir}" + (" --overwrite" if overwrite else ""))
    with st.expander("Equivalent command line (manual route)"):
        st.code(cmd, language="bash")
        st.caption("The command shown carries every setting chosen here, so the same experiment can be repeated from the command line.")
    if st.button("Run experiments", type="primary", disabled=not (scenarios and models and Xb is not None) or (out_exists and not overwrite)):
        cfg = make_cfg(); os.makedirs(outdir, exist_ok=True); rows, curves = [], []
        if tune:
            with st.spinner("Choosing the forest settings on the training segment..."):
                best = R.tune_once(Xb, cfg, info.get("neighbours")); st.info(f"Forest settings chosen once and used for every run: {best}")
        bar = st.progress(0.0); status = st.empty(); total = len(scenarios) * seeds; k = 0
        for sc in scenarios:
            for sd in range(seeds):
                status.write(f"Running {sc}, seed {sd} ...")
                r, c = R.one_run(Xb, cfg, sc, sd, models, False, info.get("neighbours")); rows += r; curves += c; k += 1; bar.progress(k / total)
                pd.DataFrame(rows).to_csv(f"{outdir}/runs.csv", index=False); pd.DataFrame(curves).to_csv(f"{outdir}/blocks.csv", index=False)
        env = R.environment_record(cfg, {k: v for k, v in info.items() if k != "neighbours"}); env.update(source=source, command=cmd)
        json.dump(env, open(f"{outdir}/environment.json", "w"), indent=2, default=str)
        SS["root_in"] = os.path.dirname(outdir) or "."; SS["last_root"] = SS["root_in"]
        SS["picked_w"] = sorted(set(map(os.path.normpath, SS.get("picked_w", []))) | {os.path.normpath(outdir)})   # preselect the new run in the Workspace
        SS["bump_outdir"] = True                                      # the next run defaults to the next free folder
        status.success(f"Done. Saved to {outdir} and added to the Workspace.")

# ------------------------------------------------------------------ 3. workspace
runs = blocks = None
with t_ws:
    c1, c2 = st.columns([2, 1])
    root = c1.text_input("Results root (all folders below it with runs.csv and blocks.csv are listed)", "results", key="root_in")
    found = A.find_result_folders(root) if os.path.isdir(root) else []
    if SS.get("last_root") != root:                       # new results root: nothing is loaded until the folders are picked
        SS["picked_w"] = []; SS["last_root"] = root
    picked = c1.multiselect("Result folders to load (load several to compare experiments side by side)", sorted(set(map(os.path.normpath, found)) | set(map(os.path.normpath, SS["picked_w"]))), key="picked_w")
    SS["picked"] = picked
    up = c2.file_uploader("...or upload runs.csv + blocks.csv", accept_multiple_files=True, type="csv")
    parts_r, parts_b = [], []
    if picked:
        r_, b_ = A.load_folders(picked); parts_r.append(r_); parts_b.append(b_)
    if up and len(up) == 2:
        d = {f.name: pd.read_csv(f) for f in up}
        if "runs.csv" in d and "blocks.csv" in d:
            d["runs.csv"]["run_label"] = "uploaded"; d["blocks.csv"]["run_label"] = "uploaded"; parts_r.append(A.refresh_measures(d["runs.csv"], d["blocks.csv"])); parts_b.append(d["blocks.csv"])
    if not parts_r: st.info("No results loaded. Pick one or more result folders above, or run experiments (tab 2), or upload the two files.")
    else:
        allruns, allblocks = pd.concat(parts_r, ignore_index=True), pd.concat(parts_b, ignore_index=True)
        f1_, f2_, f3_ = st.columns(3)
        fsc = f1_.multiselect("Scenarios", sorted(allruns.scenario.unique()), sorted(allruns.scenario.unique()))
        fmo = f2_.multiselect("Models", sorted(allruns.model.unique()), sorted(allruns.model.unique()))
        fru = f3_.multiselect("Experiments", sorted(allruns.run_label.unique()), sorted(allruns.run_label.unique()))
        runs = allruns[allruns.scenario.isin(fsc) & allruns.model.isin(fmo) & allruns.run_label.isin(fru)].reset_index(drop=True)
        blocks = allblocks[allblocks.scenario.isin(fsc) & allblocks.model.isin(fmo) & allblocks.run_label.isin(fru)].reset_index(drop=True)
        SS["fruns"], SS["fblocks"] = runs, blocks
        SS["flt_args"] = "".join(f" --filter-{opt} {','.join(sel)}" for opt, sel, allv in (("scenarios", fsc, allruns.scenario.unique()), ("models", fmo, allruns.model.unique()), ("experiments", fru, allruns.run_label.unique())) if set(sel) != set(allv))
        equiv(an_cmd("summary"), "Other tables of this tab: --what recovery, recall, blocks. Add --out table.csv to save a table.")
        st.caption(f"{len(runs)} runs loaded from {len(fru)} experiment(s). Filters here apply to every tab that follows.")
        m1, m2, m3, m4 = st.columns(4); m1.metric("Experiments", len(fru)); m2.metric("Scenarios", len(fsc)); m3.metric("Models", len(fmo)); m4.metric("Seeds (max)", int(runs.groupby(["run_label", "scenario", "model"]).size().max()) if len(runs) else 0)
        if "block_rule" in runs.columns and len(runs): st.caption("Pre/post-change F1, degradation and recovery use only blocks with enough observations (" + ", ".join(sorted(runs.block_rule.unique())) + "). Blocks where sensor batteries have died are not scored.")
        old_ = [f for f in picked if "n_obs" not in pd.read_csv(os.path.join(f, "blocks.csv"), nrows=1).columns]
        if old_ and Xb is not None:
            with st.expander("Older results without block counts (" + ", ".join(os.path.basename(f) for f in old_) + ")"):
                st.write("These folders were written before blocks carried observation counts, so only the final scored block is dropped. Add the counts to exclude every thin block. This does not rerun any model. Use the same data, time grid, selection rule and cleaning as the run (the sidebar settings); a mismatch is detected and nothing is written.")
                fo = st.selectbox("Folder", old_, key="abc_folder")
                equiv(f"python -m wsnfd.run --source intel --path <data file> --locs <mote_locs.txt> --grid-s {grid_s} --add-block-counts {fo}", "Add the same --min-nodes/--min-days/--min-coverage/--dead-voltage/--temp-range/--hum-range flags if you changed them.")
                if st.button("Add block counts", key="abc_go"):
                    try: st.success(R.add_block_counts(fo, Xb, make_cfg(), info.get("neighbours"))); st.info("Press Rerun (or reload the page) to load the updated curves.")
                    except ValueError as e: st.error(str(e))
        st.subheader("Summary over seeds (mean ± 95% CI)")
        S = A.summary(runs); keycols = [c for c in S.columns if c in A.ctx(runs) or c in ("scenario", "model")]
        shown = S[keycols].copy(); shown["n"] = S.n_seeds
        for m in ["precision", "recall", "f1", "fpr", "post_change_f1", "degradation"]:
            shown[m] = [f"{a:.3f} ± {b:.3f}" if pd.notna(b) else f"{a:.3f}" for a, b in zip(S[m + "_mean"], S[m + "_ci95"])]
        st.dataframe(shown, hide_index=True, width="stretch"); dl_csv(S, "summary_full", "dl_sum")
        st.subheader("Block-F1 over time")
        b1, b2 = st.columns(2)
        bsc = b1.selectbox("Scenario", sorted(runs.scenario.unique())); blab = b2.selectbox("Experiment", sorted(runs.run_label.unique()))
        band = st.checkbox("Show ±1 SD band", True); r_ = runs[(runs.scenario == bsc) & (runs.run_label == blab)]
        onset = int(r_.onset_block.iloc[0]) if "onset_block" in r_.columns and len(r_) else None
        if len(r_): show_fig(P.fig_blocks(blocks, bsc, onset, blab, band), f"blocks_{blab}_{bsc}", "dl_blk")
        st.subheader("Recall by fault type"); show(runs.groupby(A.ctx(runs) + ["scenario", "model"])[[c for c in runs.columns if c.startswith("recall_")]].mean().reset_index(), "recall_by_fault_type", "dl_rec", 3)
        st.subheader("Recovery after the change"); show(A.recovery_table(runs), "recovery", "dl_recov", 2)

def need_data():
    if "fruns" not in SS or SS["fruns"] is None or SS["fruns"].empty:
        st.info("Load results in the Workspace tab first."); return None
    return SS["fruns"], SS["fblocks"]

# ------------------------------------------------------------------ 4. explorer
with t_exp:
    dd = need_data()
    if dd:
        runs, blocks = dd; num = A.numeric_columns(runs); cat = [c for c in ["scenario", "model", "run_label", "feature_set", "label_delay_days"] if c in runs.columns]
        e1, e2, e3 = st.columns(3)
        meas = e1.selectbox("Measure", num, num.index("post_change_f1") if "post_change_f1" in num else 0)
        by = e2.multiselect("Group by", cat, [c for c in ("scenario", "model") if c in cat])
        sts = e3.multiselect("Statistics", A.STAT_FUNCS, ["mean", "sd", "ci95", "n"])
        with st.expander("Filters"):
            where = {c: st.multiselect(c, sorted(runs[c].unique()), sorted(runs[c].unique()), key=f"w_{c}") for c in cat}
        if sts:
            tab = A.explore(runs, meas, by, sts, where); show(tab, f"explore_{meas}", "dl_expl")
            equiv(an_cmd("explore", f" --measure {meas} --by {','.join(by)} --stats {','.join(sts)}" + "".join(f" --where {c}={','.join(v)}" for c, v in where.items() if v and set(v) != set(runs[c].unique()))))
        st.subheader("Chart")
        k1, k2, k3 = st.columns(3)
        kind = k1.selectbox("Type", ["Bar (mean, 95% CI)", "Box plot over seeds"]); xcol = k2.selectbox("x", cat, cat.index("scenario") if "scenario" in cat else 0)
        hue = k3.selectbox("Colour by", ["(none)"] + cat, (cat.index("model") + 1) if "model" in cat else 0)
        d_ = runs
        for c, v in where.items(): d_ = d_[d_[c].isin(v)]
        if len(d_):
            fn = P.fig_bar if kind.startswith("Bar") else P.fig_box
            show_fig(fn(d_, meas, xcol, None if hue == "(none)" else hue), f"{meas}_{'bar' if kind.startswith('Bar') else 'box'}", "dl_chart")
            equiv(an_cmd("explore", f" --measure {meas} --by {','.join(by)} --chart {'bar' if kind.startswith('Bar') else 'box'} --x {xcol} --hue {'none' if hue == '(none)' else hue} --chart-out chart.png" + "".join(f" --where {c}={','.join(v)}" for c, v in where.items() if v and set(v) != set(runs[c].unique()))))
        st.subheader("Sensitivity")
        facs = [c for c in A.ctx(runs)]
        if facs:
            fac = st.selectbox("Factor", facs); st.caption("Mean of the measure above by model and scenario across the levels of the factor.")
            show(A.sensitivity(runs, fac, meas), f"sensitivity_{fac}", "dl_sens")
            equiv(an_cmd("sensitivity", f" --factor {fac} --measure {meas}"))
        else: st.caption("Load two or more experiments (for example FS-A and FS-B, or different label delays) to compare them here.")

# ------------------------------------------------------------------ 5. statistics
with t_stat:
    dd = need_data()
    if dd:
        runs, blocks = dd
        s1, s2, s3 = st.columns(3)
        meas = s1.selectbox("Measure to test", [c for c in A.numeric_columns(runs)], (A.numeric_columns(runs).index("post_change_f1") if "post_change_f1" in A.numeric_columns(runs) else 0), key="st_meas")
        mods = s2.multiselect("Models in the test", sorted(runs.model.unique()), [m for m in ["M1", "M2", "M3"] if m in set(runs.model)])
        margin = s3.number_input("Practical margin (absolute)", 0.0, 0.5, 0.02, 0.01)
        nseeds = int(runs.groupby(A.ctx(runs) + ["scenario", "model"]).seed.nunique().min()) if len(runs) else 0
        if nseeds < 7: st.warning(f"Only {nseeds} seeds per cell. With fewer than 7, a Wilcoxon test with Holm correction over three comparisons cannot reach p < 0.05 whatever the effect. Use the 10 seeds planned in Chapter 4.")
        if len(mods) >= 2:
            T = A.tests(runs, meas, margin, mods)
            equiv(an_cmd("tests", f" --measure {meas} --models {','.join(mods)} --margin {margin}"))
            st.subheader("Friedman and Holm-corrected Wilcoxon, with effect sizes")
            if len(T): show(T, "tests", "dl_tests")
            else: st.write("Not enough seeds yet (need at least 3 per scenario).")
            st.caption("median_diff = median paired difference (a minus b) with bootstrap 95% interval; cohen_dz = mean paired difference / its SD; beyond_margin = |median_diff| > margin.")
            if len(T) and {"M1", "M3"} <= set(mods):
                st.subheader("Is adaptation worthwhile? (Section 3.5)")
                a1, a2 = st.columns(2); ad = a1.selectbox("Adaptive model", [m for m in mods if m != "M1"], 0); rf = a2.selectbox("Reference", mods, mods.index("M1"))
                W = A.worthwhile(T, runs, margin, ad, rf) if meas == "post_change_f1" else pd.DataFrame()
                if len(W): equiv(an_cmd("worthwhile", f" --models {','.join(mods)} --margin {margin} --adaptive {ad} --reference {rf}")); show(W, "worthwhile", "dl_worth"); st.caption("All three conditions must hold: difference beyond the margin, significant after Holm correction, cost reported. Seeds share one stream, so inference is limited to this data source.")
                else: st.caption("The criterion is defined on post_change_f1; select that measure.")
        if len(mods) >= 2 and "S0" in set(runs.scenario):
            ref_ = "M1" if "M1" in mods else mods[0]
            st.subheader("What did the injected change add? (S0 as reference, same seeds)")
            CE = A.change_effects(runs, meas, margin, ref_, mods)
            equiv(an_cmd("change", f" --measure {meas} --models {','.join(mods)} --margin {margin} --reference {ref_}"))
            if len(CE): show(CE, "change_effects", "dl_change")
            else: st.write("Not enough seeds yet (need at least 3 per scenario).")
            st.caption(f"harm of the change = scenario minus S0 for the same seed, per model. extra gain of adaptation = (adaptive minus {ref_} in the scenario) minus (the same in S0). The real stream drifts by itself, so only what exceeds S0 is due to the injected change.")
        st.subheader("Computational cost (means over seeds)"); show(A.cost_table(runs), "cost", "dl_cost", 3); equiv(an_cmd("cost"), "Gain per CPU second: --what gain.")
        g = A.gain_per_cost(runs)
        if len(g): st.subheader("Gain per extra CPU second (relative to M1)"); show(g, "gain_per_cost", "dl_gpc")
        show_fig(P.fig_cost(runs), "cost", "dl_costfig")

# ------------------------------------------------------------------ 6. export
def _bundle_key(runs, margin):
    """What a bundle was built from: the experiments loaded, the number of runs and the margin; used to stop a stale bundle being downloaded."""
    return (tuple(sorted(runs.run_label.unique())) if "run_label" in runs.columns else (), len(runs), margin)
with t_out:
    dd = need_data()
    if dd:
        runs, blocks = dd
        margin = st.number_input("Practical margin used in the exported tests", 0.0, 0.5, 0.02, 0.01, key="exp_margin")
        st.write("One click builds every table (CSV and an Excel workbook), every figure (PNG) and a zip of all of it, from the data currently loaded and filtered.")
        if st.button("Build export bundle", type="primary"):
            with tempfile.TemporaryDirectory() as td:
                env_files = [os.path.join(f, "environment.json") for f in SS.get("picked", []) if os.path.exists(os.path.join(f, "environment.json"))]    # each loaded run's settings and tuning scores go into the zip
                RP.write_all(runs, blocks, td, margin, extra_files=env_files)
                SS["bundle_src"] = _bundle_key(runs, margin)
                SS["bundle"] = open(f"{td}/bundle.zip", "rb").read(); SS["xlsx"] = open(f"{td}/results.xlsx", "rb").read(); SS["mat"] = open(f"{td}/results.mat", "rb").read(); SS["nb"] = open(f"{td}/analysis.ipynb", "rb").read()
        if "bundle" in SS and SS.get("bundle_src") != _bundle_key(runs, margin):
            st.warning("The bundle below was built from different results (" + ", ".join(SS.get("bundle_src", ((),))[0]) + "), so it is hidden. Press Build export bundle to build it from what is loaded now.")
        elif "bundle" in SS:
            st.caption("Bundle built from: " + ", ".join(SS["bundle_src"][0]) + f" ({SS['bundle_src'][1]} runs).")
            x1, x2, x3, x4, x5, x6 = st.columns(6)
            x1.download_button("Full bundle (zip)", SS["bundle"], f"{RP.file_tag(runs)}_report.zip", "application/zip")
            x2.download_button("Excel workbook", SS["xlsx"], f"{RP.file_tag(runs)}_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            x6.download_button("Jupyter notebook", SS["nb"], f"{RP.file_tag(runs)}_analysis.ipynb", "application/x-ipynb+json")
            x4.download_button("MATLAB file (.mat)", SS["mat"], f"{RP.file_tag(runs)}_results.mat", "application/octet-stream")
            x3.download_button("Raw runs (CSV)", runs.to_csv(index=False).encode(), f"{RP.file_tag(runs)}_runs.csv", "text/csv")
            x5.download_button("Block curves (CSV)", blocks.to_csv(index=False).encode(), f"{RP.file_tag(runs)}_blocks.csv", "text/csv")
        st.divider(); st.subheader("Manual route (no interface)")
        st.code("python -m wsnfd.report " + " ".join(SS.get("picked", ["results/run1"])) + f" --out report_out --margin {margin}", language="bash")
        st.caption("The report command produces the same tables, workbook, figures and zip. Filters set in the Workspace tab apply only to the interface; the command uses every row of the folders named.")

# ------------------------------------------------------------------ 7. live view
@st.cache_resource(show_spinner="Computing the replay (trains the models once)...")
def replay(_Xb, _info, cfg_repr, cfg, scenario, seed, models, key):
    return LV.build_replay(_Xb, _info, cfg, scenario, seed, list(models), _info.get("neighbours"))

def draw_frame(rp, ms, t, win_ep):
    n = len(ms); ncols = min(n + 1, 4); nrows = int(np.ceil((n + 1) / ncols))        # up to 3 models in one row; more wrap onto a second row
    fig = plt.figure(figsize=(4.6 * ncols + (1.4 if nrows == 1 else 0), 4.6 * nrows))
    gs = fig.add_gridspec(nrows, ncols, width_ratios=[1] * (ncols - 1) + [1.15 if nrows == 1 else 1])
    xy = rp["xy"]
    for i, m in enumerate(ms):
        ax = fig.add_subplot(gs[i // ncols, i % ncols]); s = LV.frame_status(rp, m, t)
        for code in (LV.NO_DATA, LV.TN, LV.TP, LV.FN, LV.FP):
            k = s == code
            if k.any(): ax.scatter(xy[k, 0], xy[k, 1], s=170 if code in (LV.TP, LV.FN, LV.FP) else 90, c=LV.STATUS_COLORS[code], edgecolors="k", linewidths=0.6, label=LV.STATUS_NAMES[code], zorder=3)
        for j, nd in enumerate(rp["nodes"]): ax.annotate(str(nd), xy[j], fontsize=6, ha="center", va="center", zorder=4, color="white" if s[j] in (LV.TP, LV.FN) else "black")
        f1 = LV.trailing_f1(rp, m, t, win_ep); ax.set_title(f"{m}   F1 (last {win_ep} epochs) = " + ("n/a" if np.isnan(f1) else f"{f1:.2f}"), fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.08)
        if i == 0: ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False)
    ax = fig.add_subplot(gs[n // ncols, n % ncols]); x0 = rp["train_end"]
    for m in ms:
        y = LV.f1_series(rp, m, t, win_ep); ax.plot(np.arange(x0, t + 1) / rp["epochs_per_day"], y, lw=1.2, label=m)
        for e in rp["events"].get(m, []):
            if x0 <= e <= t: ax.axvline(e / rp["epochs_per_day"], ls=":", lw=0.6, alpha=0.5)
    if t >= rp["onset"]: ax.axvline(rp["onset"] / rp["epochs_per_day"], c="r", ls="--", lw=1.2, label="change")
    ax.set_xlim(x0 / rp["epochs_per_day"], rp["T"] / rp["epochs_per_day"]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("day"); ax.set_ylabel("trailing F1"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3); ax.set_title("Trailing F1 (dotted = model update)", fontsize=9)
    fig.tight_layout(); return fig

with t_live:
    st.write("Replays the test stream epoch by epoch. Each node is coloured by what the model decided, and the curve shows how detection quality evolves, including the moment of the change and each retraining. "
             "This is a replay of the experiment stream, not a connection to hardware.")
    if Xb is None: st.info("Prepare data in the sidebar first.")
    else:
        l1, l2, l3 = st.columns(3)
        lsc = l1.selectbox("Scenario", ["S0", "S1", "S2", "S3", "S4"], index=2, key="lv_sc"); lseed = int(l2.number_input("Seed", 0, 99, 0, key="lv_seed"))
        lms = l3.multiselect("Models to compare (up to 5)", models or ["M1", "M3"], (models or ["M1", "M3"])[:2], max_selections=5, key="lv_ms")
        if lms:
            cfg = make_cfg(); key = (source, path, grid_s, tuple(sorted(sim.items())), sim_seed)
            try: rp = replay(Xb, info, repr(cfg), cfg, lsc, lseed, tuple(lms), key)
            except Exception as e: rp = None; st.error(f"Could not build the replay: {e}")
            if rp:
                epd = rp["epochs_per_day"]; first, last = rp["train_end"], rp["T"] - 1
                c1, c2, c3 = st.columns([3, 1, 1])
                pos = c1.slider("Position (day of stream)", float(first / epd), float(last / epd), float(max(first, rp["onset"] - epd) / epd), 1.0 / epd, key="lv_pos")
                step_d = c2.select_slider("Speed (hours per frame)", [1, 3, 6, 12, 24], 6, key="lv_speed"); win_d = c3.select_slider("F1 window (days)", [0.5, 1.0, 2.0], 1.0, key="lv_win")
                win_ep = max(1, int(win_d * epd)); t = int(round(pos * epd)); t = min(max(t, first), last)
                b1, b2 = st.columns([1, 5]); play = b1.button("Play from here", type="primary", key="lv_play")
                slot = st.empty(); info_slot = st.empty()
                def paint(tt):
                    fig = draw_frame(rp, lms, tt, win_ep); slot.pyplot(fig, clear_figure=True); plt.close(fig)
                    info_slot.caption(f"Day {tt / epd:.2f} of the stream" + ("  |  after the change" if tt >= rp["onset"] else "  |  before the change")
                                      + f"  |  true faults this epoch: {int(rp['y'][tt].sum())} of {rp['N']} nodes")
                if play:
                    step = max(1, int(step_d / 24 * epd)); tt = t
                    while tt <= last: paint(tt); tt += step; time.sleep(0.15)
                else: paint(t)
                st.caption("Green: normal and not flagged. Blue: fault correctly detected. Red: fault missed. Orange: false alarm. Grey: no reading. Press Play, or drag the position slider to jump.")
                fr = pd.DataFrame({"epoch": np.arange(first, last + 1)})
                for m in lms: fr[f"{m}_f1"] = LV.f1_series(rp, m, last, win_ep)
                st.download_button("Download the F1 curves (CSV)", fr.to_csv(index=False).encode(), "live_f1_curves.csv", "text/csv", key="dl_live")

# ------------------------------------------------------------------ 8. live sandbox
FT_LABEL = {"spike": "S", "noise": "N", "bias": "B", "drift": "D", "stuck": "K"}

def _sb(): return SS.get("sb")
def cb_run(flag): SS["sb_running"] = flag
def cb_step():
    if _sb(): _sb().step()
def cb_all(): SS["sb_targets"] = list(_sb().nodes)
def cb_none(): SS["sb_targets"] = []
def cb_random():
    ids = list(_sb().nodes); k = min(int(SS.get("sb_k", 3)), len(ids)); SS["sb_targets"] = [int(i) for i in _sb().rng.choice(ids, k, replace=False)] if k else []
def cb_add(): _sb().add_nodes(int(SS.get("sb_add_n", 1)))
def cb_delete(): _sb().delete_nodes(SS.get("sb_targets", [])); SS["sb_targets"] = []
def cb_delete_all(): _sb().delete_all(); SS["sb_targets"] = []
def cb_on(flag): _sb().set_on(SS.get("sb_targets", []), flag)
def cb_on_all(): _sb().set_on(list(_sb().nodes), True)
def cb_clear_faults(): _sb().clear_faults(SS.get("sb_targets", []))
def cb_clear_all_faults(): _sb().clear_faults(None)
def cb_clear_data(): _sb().clear_data(); SS.pop("sb_csv", None)
def cb_jump(): _sb().jump_feed()
def cb_inject(ft):
    mode = SS.get("sb_dur_mode", "Typical for the type"); dur = "default" if mode.startswith("Typical") else None if mode.startswith("Until") else int(SS.get("sb_dur", 30))
    mag = float(SS.get("sb_mag", 0.0)) or None; sign = {"random": 0, "+": 1, "-": -1}[SS.get("sb_sign", "random")]
    chs = tuple(SS.get("sb_chs", ["temperature"])) or ("temperature",)
    _sb().inject(SS.get("sb_targets", []), ft, chs, mag, dur, sign)

def draw_sandbox(sb, inspect_id):
    ids = list(sb.nodes); st_ = sb.status(ids); xy = np.array([sb.nodes[i].xy for i in ids]) if ids else np.zeros((0, 2))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    tg = set(SS.get("sb_targets", []))
    for code in (LV.WARM, LV.NO_DATA, LV.OFF, LV.TN, LV.TP, LV.FN, LV.FP):
        k = st_ == code
        if k.any(): ax.scatter(xy[k, 0], xy[k, 1], s=190 if code in (LV.TP, LV.FN, LV.FP) else 100, c=LV.STATUS_COLORS[code], edgecolors="k", linewidths=0.6, label=LV.STATUS_NAMES[code], zorder=3)
    for j, i in enumerate(ids):
        n = sb.nodes[i]
        if i in tg: ax.scatter([xy[j, 0]], [xy[j, 1]], s=330, facecolors="none", edgecolors="#7b3294", linewidths=2, zorder=2)
        ax.annotate(str(i) + ("*" if n.clone else ""), xy[j], fontsize=6, ha="center", va="center", zorder=4, color="white" if st_[j] in (LV.TP, LV.FN, LV.OFF) else "black")
        if n.faults: ax.annotate("".join(FT_LABEL[f] for f in n.faults), xy[j], xytext=(0, 11), textcoords="offset points", fontsize=7, ha="center", color="#d7191c", weight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.1)
    if ids: ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, frameon=False)
    ax.set_title(f"{len(ids)} nodes ({sum(n.on for n in sb.nodes.values())} on)   * = clone   purple ring = selected   letters = injected fault", fontsize=8)
    fig.tight_layout(); return fig

def draw_trace(sb, i):
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
    h = [(k, d[i]) for k, d in enumerate(list(sb.hist)[-240:]) if i in d]
    if h:
        t = np.array([sb.tick - len(list(sb.hist)[-240:]) + k for k, _ in h]); v = np.array([x for _, x in h], float)
        for a, ch, nm in ((axs[0], 0, "temperature"), (axs[1], 1, "humidity")):
            a.plot(t, v[:, ch], lw=1, c="#1f77b4"); y = v[:, 2] > 0; p = v[:, 3] == 1
            a.fill_between(t, np.nanmin(v[:, ch]) if np.isfinite(v[:, ch]).any() else 0, np.nanmax(v[:, ch]) if np.isfinite(v[:, ch]).any() else 1, where=y, color="#d7191c", alpha=0.15, label="injected fault")
            a.scatter(t[p], v[p, ch], marker="x", c="k", s=18, zorder=3, label="flagged")
            a.set_ylabel(nm, fontsize=8); a.grid(alpha=0.3)
        axs[0].legend(fontsize=6, loc="upper left")
    axs[1].set_xlabel("tick since the sandbox started"); axs[0].set_title(f"Node {i}: readings, injected faults and detector flags", fontsize=8)
    fig.tight_layout(); return fig

def sandbox_panel():
    sb = _sb()
    if SS.get("sb_running"):
        for _ in range(int(SS.get("sb_tpf", 1))): sb.step()
    s_ = sb.stats(); m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recorded ticks", f"{s_['ticks']:,}"); m2.metric("Recorded rows", f"{s_['recorded_rows']:,}"); m3.metric("Alarms raised", f"{s_['alarms']:,}")
    m4.metric("F1 since clear", "n/a" if np.isnan(s_["f1"]) else f"{s_['f1']:.2f}"); m5.metric("Missed / false alarms", f"{s_['fn']} / {s_['fp']}")
    if not sb.hist: st.caption("No data recorded yet. Press Run or Step.")
    elif len(sb.buf) < sb.cfg.roll_window: st.caption(f"The detector is warming up ({len(sb.buf)} of {sb.cfg.roll_window} readings needed per node).")
    p1, p2 = st.columns(2)
    fig = draw_sandbox(sb, None); p1.pyplot(fig, clear_figure=True); plt.close(fig)
    ins = SS.get("sb_inspect")
    if ins in sb.nodes:
        fig = draw_trace(sb, ins); p2.pyplot(fig, clear_figure=True); plt.close(fig)
    else: p2.info("Choose a node to inspect.")
    rec = pd.DataFrame([(k, v[1], v[0]) for k, v in sb.by_type.items()], columns=["fault type", "rows with fault", "flagged"]).assign(recall=lambda d: (d["flagged"] / d["rows with fault"]).where(d["rows with fault"] > 0))
    st.dataframe(rec.round(2), hide_index=True, width="stretch")

with t_sb:
    st.write("A hands-on network. Add or delete nodes, switch them off, inject any fault type, clear the recorded data and watch the trained detector react. "
             "Every reading comes from the prepared data and keeps its source node and source timestamp, so it can be checked against the original file.")
    if Xb is None: st.info("Prepare data in the sidebar first.")
    else:
        g1, g2, g3, g4 = st.columns([1, 1, 1, 1])
        n0 = g1.slider("Initial nodes", 4, 30, 12, key="sb_n0"); g2.slider("Ticks per frame", 1, 20, 1, key="sb_tpf"); refresh = g3.select_slider("Refresh every (s)", [0.25, 0.5, 1.0, 2.0], 0.5, key="sb_refresh")
        if g4.button("Create or reset the network", type="primary", key="sb_create"):
            with st.spinner("Training the detector on the prepared data..."):
                SS["sb"] = SBX.Sandbox(Xb, info, make_cfg(), seed=sim_seed, n_nodes=n0); SS["sb_running"] = False; SS["sb_targets"] = []; SS.pop("sb_csv", None)
        sb = _sb()
        if sb is None: st.info("Press 'Create or reset the network' to start.")
        else:
            ids = list(sb.nodes); SS["sb_targets"] = [i for i in SS.get("sb_targets", []) if i in ids]
            r1, r2, r3, r4 = st.columns([1, 1, 1, 3])
            r1.button("Run", on_click=cb_run, args=(True,), type="primary", key="sb_runb", disabled=bool(SS.get("sb_running")))
            r2.button("Pause", on_click=cb_run, args=(False,), key="sb_pauseb", disabled=not SS.get("sb_running"))
            r3.button("Step once", on_click=cb_step, key="sb_stepb")
            r4.caption("Running" if SS.get("sb_running") else "Paused")
            with st.expander("Select nodes", expanded=True):
                st.multiselect("Selected nodes (the actions below apply to these)", ids, key="sb_targets")
                q1, q2, q3, q4 = st.columns(4)
                q1.button("Select all", on_click=cb_all, key="sb_all"); q2.button("Select none", on_click=cb_none, key="sb_none")
                q3.number_input("Random k", 1, 500, 3, key="sb_k"); q4.button("Select k at random", on_click=cb_random, key="sb_rand")
            with st.expander("Network: add, delete, switch off and on", expanded=True):
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.number_input("Nodes to add", 1, 200, 5, key="sb_add_n"); a1.button("Add nodes", on_click=cb_add, key="sb_addb")
                a2.button("Delete selected", on_click=cb_delete, key="sb_delb"); a3.button("Delete ALL nodes", on_click=cb_delete_all, key="sb_delallb")
                a4.button("Switch selected OFF", on_click=cb_on, args=(False,), key="sb_offb"); a5.button("Switch selected ON", on_click=cb_on, args=(True,), key="sb_onb")
                st.button("Switch every node ON", on_click=cb_on_all, key="sb_onallb")
                st.caption("New nodes use the prepared data's own nodes first. Once those are used up, extra nodes are clones (marked *): a real series shifted by whole days with a little added sensor noise.")
            with st.expander("Faults: introduce an error on the selected nodes", expanded=True):
                f1_, f2_, f3_, f4_ = st.columns(4)
                f1_.multiselect("Channels", ["temperature", "humidity"], ["temperature"], key="sb_chs")
                f2_.number_input("Magnitude (x node sigma, 0 = typical)", 0.0, 20.0, 0.0, 0.5, key="sb_mag")
                f3_.radio("Duration", ["Typical for the type", "Fixed number of ticks", "Until cleared"], key="sb_dur_mode")
                f3_.number_input("Ticks (if fixed)", 1, 2000, 30, key="sb_dur"); f4_.radio("Direction", ["random", "+", "-"], horizontal=True, key="sb_sign")
                b1, b2, b3, b4, b5 = st.columns(5)
                for col, (ft, lab) in zip((b1, b2, b3, b4, b5), (("spike", "Spike"), ("noise", "Noise"), ("bias", "Bias"), ("drift", "Drift"), ("stuck", "Stuck-at"))):
                    col.button(lab, on_click=cb_inject, args=(ft,), key=f"sb_f_{ft}", width="stretch")
                c1_, c2_ = st.columns(2); c1_.button("Clear faults on selected", on_click=cb_clear_faults, key="sb_cf"); c2_.button("Clear all faults", on_click=cb_clear_all_faults, key="sb_caf")
            with st.expander("Recorded data", expanded=True):
                d1, d2, d3, d4 = st.columns(4)
                d1.button("Clear recorded data", on_click=cb_clear_data, key="sb_cd", help="Empties the log and the detector's history; nodes and faults stay. The detector warms up again.")
                d2.button("Jump feed to a random time", on_click=cb_jump, key="sb_jmp")
                if d3.button("Prepare CSV of the recording", key="sb_prep"): SS["sb_csv"] = sb.log_df().to_csv(index=False).encode()
                if "sb_csv" in SS: d4.download_button(f"Download recording ({len(SS['sb_csv'])/1e6:.1f} MB)", SS["sb_csv"], "live_recording.csv", "text/csv", key="sb_dl")
                if len(sb.log): st.dataframe(sb.log_df().tail(30), hide_index=True, width="stretch")
            with st.expander("Where do these readings come from?"):
                st.write(("**Real data:** the Intel Berkeley file you loaded. Each row's source_node and source_time can be found in the original data.txt." if sb.real else
                          "**Simulated data:** the fallback generator in this package, not field measurements. Load the Intel file in the sidebar to demonstrate with real readings.")
                         + f" The feed is now at source epoch {sb.f} of {sb.T}. After the end of the data it restarts at the first test epoch.")
                st.dataframe(pd.DataFrame([dict(node=n.id, on=n.on, clone=n.clone, source_node=sb.base_ids[n.src], x=round(n.xy[0], 2), y=round(n.xy[1], 2), faults=",".join(n.faults)) for n in sb.nodes.values()]), hide_index=True, width="stretch")
                if sb.events: st.caption("Recent events: " + "; ".join(f"t{t}: {e}" for t, e in sb.events[-8:]))
            if SS.get("sb_inspect") not in ids: SS.pop("sb_inspect", None)
            if ids: st.selectbox("Inspect node", ids, key="sb_inspect")
            st.fragment(sandbox_panel, run_every=(refresh if SS.get("sb_running") else None))()
