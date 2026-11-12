"""Results analysis for Chapter 6: summaries, statistical comparison, cost, sensitivity and a general explorer.
Works on one results folder or on several loaded side by side (a run_label column keeps them apart)."""
import os, glob
import numpy as np, pandas as pd
from scipy import stats as sps
from .stats import compare, holm, boot_ci
from . import metrics as M

MEASURES = ["precision", "recall", "f1", "fpr", "mcc", "accuracy", "pre_change_f1", "degradation", "post_change_f1"]
COST = ["fit_time_s", "predict_time_s", "predict_time_per_obs_s", "n_updates", "n_train_rows", "labels_consumed", "model_bytes", "peak_rss_mb"]
STAT_FUNCS = ["mean", "sd", "ci95", "median", "min", "max", "n"]

def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    return float(sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

# ---------- loading ----------
def find_result_folders(root):
    """Sub-folders of root (and root itself) that contain runs.csv and blocks.csv."""
    out = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "runs.csv"), recursive=True)]
    return sorted(os.path.normpath(f) for f in out if os.path.exists(os.path.join(f, "blocks.csv")))

def load_folders(paths):
    """Concatenate runs and blocks from several folders; adds run_label (the folder name)."""
    R, B, seen = [], [], set()
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))                  # "results/run1" and "results\\run1" are the same folder: load it once
        if key in seen: continue
        seen.add(key)
        label = os.path.basename(os.path.normpath(p)) or p
        r = pd.read_csv(os.path.join(p, "runs.csv")); b = pd.read_csv(os.path.join(p, "blocks.csv"))
        if "accuracy" not in r.columns and {"tp", "fp", "fn", "tn"} <= set(r.columns):      # results written before accuracy was added
            r["accuracy"] = (r.tp + r.tn) / (r.tp + r.fp + r.fn + r.tn).replace(0, float("nan"))
        r["run_label"] = label; b["run_label"] = label; R.append(r); B.append(b)
    R, B = pd.concat(R, ignore_index=True), pd.concat(B, ignore_index=True)
    return refresh_measures(R, B), B

def refresh_measures(runs, blocks):
    """Recompute pre_change_f1, degradation, recovery_blocks and post_change_f1 from the saved block curves, on usable blocks only.
    New results carry the number of observations per block and drop thin blocks; older results carry none, so only the final block is dropped.
    One definition for every route (interface, command line, report), so results written before this rule are brought in line when loaded."""
    runs = runs.copy()
    if blocks is None or not len(blocks) or "onset_block" not in runs.columns: return runs
    keys = [c for c in ("run_label", "scenario", "seed", "model") if c in runs.columns and c in blocks.columns]
    have_n = "n_obs" in blocks.columns; out = {}
    for k, d in blocks.groupby(keys):
        d = d.sort_values("block"); bf = d.f1.to_numpy(float)
        if have_n: us = M.usable_blocks(d.n_obs.to_numpy(float))
        else:
            us = np.isfinite(bf); last = np.nonzero(us)[0]                      # no counts: drop empty trailing blocks and the final scored block
            if len(last): us[last[-1]] = False
        out[k if isinstance(k, tuple) else (k,)] = (bf, us)
    cols = ["pre_change_f1", "degradation", "recovery_blocks", "post_change_f1"]; new = {c: [] for c in cols}; rule = []
    for _, r in runs.iterrows():
        k = tuple(r[c] for c in keys)
        if k in out:
            bf, us = out[k]; a = M.adaptation(bf, int(r.onset_block), usable=us); rule.append("block counts" if have_n else "final scored block dropped")
        else: a = {c: r[c] for c in cols}; rule.append("as saved")
        for c in cols: new[c].append(a[c])
    for c in cols: runs[c] = new[c]
    runs["block_rule"] = rule
    return runs

def ctx(df, extra=("run_label", "feature_set", "label_delay_days")):
    """Columns that distinguish separate experiments inside the data (only those that actually vary)."""
    return [c for c in extra if c in df.columns and df[c].nunique(dropna=False) > 1]

# ---------- tables ----------
def summary(runs, measures=MEASURES, by=("scenario", "model")):
    """Mean, SD and 95% CI over seeds for each measure."""
    measures = [m for m in measures if m in runs.columns]
    keys = ctx(runs) + list(by); g = runs.groupby(keys)
    out = g[measures].mean().add_suffix("_mean").join(g[measures].std(ddof=1).add_suffix("_sd")).join(g[measures].agg(ci95).add_suffix("_ci95"))
    out["n_seeds"] = g.size(); return out.reset_index()

def recovery_table(runs):
    """Median recovery time in blocks and the share of runs that did not recover."""
    g = runs.groupby(ctx(runs) + ["scenario", "model"])["recovery_blocks"]
    return pd.DataFrame({"median_recovery_blocks": g.median(), "not_recovered_share": g.apply(lambda s: s.isna().mean())}).reset_index()

def cost_table(runs):
    return runs.groupby(ctx(runs) + ["scenario", "model"])[[c for c in COST if c in runs.columns]].mean().reset_index()

def gain_per_cost(runs, baseline="M1"):
    """(post-change F1 minus baseline's) / extra CPU seconds, per seed then averaged. NaN when there is no extra cost."""
    keys = ctx(runs) + ["scenario"]; rows = []
    for k, d in runs.groupby(keys + ["seed"]):
        b = d[d.model == baseline]
        if b.empty: continue
        b = b.iloc[0]
        for _, m in d[d.model != baseline].iterrows():
            extra = (m.fit_time_s + m.predict_time_s) - (b.fit_time_s + b.predict_time_s); gain = m.post_change_f1 - b.post_change_f1
            rows.append(dict(zip(keys, k[:-1]), seed=k[-1], model=m.model, gain=gain, extra_cpu_s=extra,
                             gain_per_cpu_s=gain / extra if extra > 0 else np.nan))
    df = pd.DataFrame(rows)
    return df.groupby(keys + ["model"])[["gain", "extra_cpu_s", "gain_per_cpu_s"]].mean().reset_index() if len(df) else df

def tests(runs, measure="post_change_f1", margin=0.02, models=None):
    """Friedman + Holm-corrected Wilcoxon per experiment and scenario on per-seed values (paired by seed)."""
    keys = ctx(runs) + ["scenario"]; rows = []
    models = models or sorted(runs.model.unique())
    for k, d in runs.groupby(keys):
        piv = d[d.model.isin(models)].pivot_table(index="seed", columns="model", values=measure).dropna()
        if piv.shape[0] < 3 or piv.shape[1] < 2: continue
        res = compare({m: piv[m].to_numpy() for m in piv.columns}, margin)
        k = k if isinstance(k, tuple) else (k,)
        for p in res["pairs"]: rows.append(dict(zip(keys, k), measure=measure, friedman_p=res["friedman_p"], **p))
    return pd.DataFrame(rows)

def change_effects(runs, measure="post_change_f1", margin=0.02, reference="M1", models=None, baseline="S0"):
    """What the injected change did, with the no-change scenario of the same seed as the reference (the real stream drifts by itself).
    harm: scenario minus S0, per model (negative = the change lowered the score).
    extra gain: (adaptive minus reference in the scenario) minus (adaptive minus reference in S0), per adaptive model.
    Wilcoxon on per-seed differences, Holm-corrected over the scenarios of each row type, with a bootstrap interval for the median."""
    keys = ctx(runs); rows = []
    models = models or sorted(runs.model.unique())
    for k, d in (runs.groupby(keys) if keys else [((), runs)]):
        k = k if isinstance(k, tuple) else (k,)
        piv = d.pivot_table(index="seed", columns=["scenario", "model"], values=measure)
        if baseline not in piv.columns.get_level_values(0): continue
        fam = {}
        for sc in sorted(set(piv.columns.get_level_values(0)) - {baseline}):
            for m in models:
                if (sc, m) in piv.columns and (baseline, m) in piv.columns:
                    fam.setdefault(("harm of the change", m), []).append((sc, (piv[(sc, m)] - piv[(baseline, m)]).dropna()))
                if m != reference and all(c in piv.columns for c in [(sc, m), (sc, reference), (baseline, m), (baseline, reference)]):
                    dd = ((piv[(sc, m)] - piv[(sc, reference)]) - (piv[(baseline, m)] - piv[(baseline, reference)])).dropna()
                    fam.setdefault(("extra gain of adaptation", f"{m} vs {reference}"), []).append((sc, dd))
        for (kind, who), items in fam.items():
            items = [(sc, x) for sc, x in items if len(x) >= 3]
            if not items: continue
            ps = []
            for sc, x in items:
                try: ps.append(float(sps.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0)
                except ValueError: ps.append(1.0)
            adj = holm(ps)
            for (sc, x), p_, pa in zip(items, ps, adj):
                med, lo, hi = boot_ci(x.to_numpy())
                rows.append(dict(zip(keys, k), scenario=sc, effect=kind, model=who, mean_diff=float(x.mean()), median_diff=med, ci_lo=lo, ci_hi=hi,
                                 p=p_, p_holm=float(pa), n_pairs=len(x), beyond_margin=bool(abs(med) > margin), significant_holm=bool(pa < 0.05)))
    return pd.DataFrame(rows)

def worthwhile(tests_df, runs, margin=0.02, adaptive="M3", reference="M1"):
    """Section 3.5 criterion for the adaptive model against the reference, per experiment and scenario."""
    keys = [c for c in ctx(runs)] + ["scenario"]; rows = []; gpc = gain_per_cost(runs, reference)
    if tests_df.empty: return pd.DataFrame()
    for k, d in tests_df.groupby(keys):
        t = d[((d.a == reference) & (d.b == adaptive)) | ((d.a == adaptive) & (d.b == reference))]
        if t.empty: continue
        t = t.iloc[0]; diff = -t.median_diff if t.a == reference else t.median_diff
        k = k if isinstance(k, tuple) else (k,)
        sel = np.ones(len(gpc), bool) if len(gpc) else []
        if len(gpc):
            for c, v in zip(keys, k): sel &= (gpc[c] == v).to_numpy()
            sel &= (gpc.model == adaptive).to_numpy()
        cost = bool(np.any(sel)) if len(gpc) else False
        rows.append(dict(zip(keys, k), adaptive_minus_reference=diff, beyond_margin=bool(diff > margin),
                         significant_holm=bool(t.p_holm < 0.05), cost_reported=cost,
                         worthwhile=bool(diff > margin and t.p_holm < 0.05 and cost)))
    W = pd.DataFrame(rows)
    if W.empty:        # the reference model is not in these runs (for example a label-delay run of M2 and M3 only): no comparison against it is possible
        return pd.DataFrame(columns=keys + ["adaptive_minus_reference", "beyond_margin", "significant_holm", "cost_reported", "worthwhile",
                                            "extra_gain_over_S0", "extra_significant_holm", "extra_beyond_margin", "worthwhile_beyond_drift"])
    # beyond natural drift: the adaptive model's advantage must also exceed its advantage in S0 (same seeds), beyond the margin and Holm-significant
    E = change_effects(runs, "post_change_f1", margin, reference, [adaptive, reference]) if len(W) else pd.DataFrame()
    E = E[(E.effect == "extra gain of adaptation") & (E.model == f"{adaptive} vs {reference}")] if len(E) else E
    ek = [c for c in keys]
    if len(E): W = W.merge(E[ek + ["median_diff", "significant_holm", "beyond_margin"]].rename(columns={"median_diff": "extra_gain_over_S0", "significant_holm": "extra_significant_holm", "beyond_margin": "extra_beyond_margin"}), on=ek, how="left")
    else: W["extra_gain_over_S0"] = np.nan; W["extra_significant_holm"] = np.nan; W["extra_beyond_margin"] = np.nan
    W["worthwhile_beyond_drift"] = (W.worthwhile & W.extra_beyond_margin.fillna(False).astype(bool) & W.extra_significant_holm.fillna(False).astype(bool) & (W.extra_gain_over_S0.fillna(0) > 0))
    return W

def block_curves(blocks):
    """Mean and SD of block-F1 per experiment, scenario, model and block (for plotting)."""
    keys = ctx(blocks, ("run_label",)) + ["scenario", "model", "block"]
    return blocks.groupby(keys)["f1"].agg(["mean", "std"]).reset_index()

# ---------- general explorer ----------
def numeric_columns(runs):
    return [c for c in runs.columns if pd.api.types.is_numeric_dtype(runs[c]) and c not in ("seed",)]

def explore(runs, measure, by, stats=("mean", "sd", "ci95", "n"), where=None):
    """Any measure by any grouping, with chosen statistics. `where` is a dict column -> allowed values."""
    d = runs
    for c, vals in (where or {}).items():
        if vals: d = d[d[c].isin(vals)]
    fn = {"mean": "mean", "sd": lambda x: x.std(ddof=1), "ci95": ci95, "median": "median", "min": "min", "max": "max",
          "n": lambda x: x.notna().sum()}
    by = list(dict.fromkeys(by))                                       # a column named twice is grouped once
    if not by: d = d.assign(_all="all"); by = ["_all"]
    out = d.groupby(by)[measure].agg([fn[s] for s in stats]); out.columns = list(stats)
    return out.reset_index()

def sensitivity(runs, factor, measure="post_change_f1"):
    """Mean measure by model across the levels of a factor (label delay, feature set, run folder) and scenario."""
    keys = [c for c in ctx(runs) if c != factor]
    return runs.pivot_table(index=keys + ["scenario", "model"], columns=factor, values=measure, aggfunc="mean").reset_index()


def recall_by_type(runs):
    """Mean recall of each fault type by experiment, scenario and model."""
    return runs.groupby(ctx(runs) + ["scenario", "model"])[[c for c in runs.columns if c.startswith("recall_")]].mean().reset_index()

def main(argv=None):
    """Command-line counterpart of the Workspace, Explorer and Statistics tabs.
    python -m wsnfd.analyse results/fsA results/fsB --what summary|recovery|cost|gain|recall|blocks|tests|worthwhile|change|explore|sensitivity [options] [--out table.csv]"""
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("folders", nargs="+")
    ap.add_argument("--what", required=True, choices=["summary", "recovery", "cost", "gain", "recall", "blocks", "tests", "worthwhile", "change", "explore", "sensitivity"])
    ap.add_argument("--filter-scenarios"); ap.add_argument("--filter-models"); ap.add_argument("--filter-experiments", help="comma lists, as the filters of the Workspace tab")
    ap.add_argument("--measure", default="post_change_f1"); ap.add_argument("--by", default="scenario,model"); ap.add_argument("--stats", default="mean,sd,ci95,n")
    ap.add_argument("--where", action="append", default=[], help="column=v1,v2 (repeatable), as the Filters of the Explorer tab")
    ap.add_argument("--models", help="models in the tests (Statistics tab); default M1,M2,M3"); ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--adaptive", default="M3"); ap.add_argument("--reference", default="M1"); ap.add_argument("--factor", default="run_label")
    ap.add_argument("--out", help="write the table as CSV")
    ap.add_argument("--chart", choices=["bar", "box"], help="with --what explore: also draw the chart of the Explorer tab (bar = mean with 95% CI, box = over seeds)")
    ap.add_argument("--x", default="scenario"); ap.add_argument("--hue", default="model"); ap.add_argument("--chart-out", default="chart.png")
    a = ap.parse_args(argv); runs, blocks = load_folders(a.folders)
    for opt, col in ((a.filter_scenarios, "scenario"), (a.filter_models, "model"), (a.filter_experiments, "run_label")):
        if opt: runs = runs[runs[col].isin(opt.split(","))]; blocks = blocks[blocks[col].isin(opt.split(","))]
    runs = runs.reset_index(drop=True); blocks = blocks.reset_index(drop=True)
    w = a.what; mods = a.models.split(",") if a.models else [m for m in ["M1", "M2", "M3"] if m in set(runs.model)]
    if w == "summary": out = summary(runs)
    elif w == "recovery": out = recovery_table(runs)
    elif w == "cost": out = cost_table(runs)
    elif w == "gain": out = gain_per_cost(runs)
    elif w == "recall": out = recall_by_type(runs)
    elif w == "blocks": out = block_curves(blocks)
    elif w == "tests": out = tests(runs, a.measure, a.margin, mods)
    elif w == "worthwhile": out = worthwhile(tests(runs, "post_change_f1", a.margin, mods), runs, a.margin, a.adaptive, a.reference)
    elif w == "change": out = change_effects(runs, a.measure, a.margin, a.reference, mods)
    elif w == "explore": out = explore(runs, a.measure, [c for c in a.by.split(",") if c], a.stats.split(","), {k: v.split(",") for k, v in (x.split("=", 1) for x in a.where)})
    else: out = sensitivity(runs, a.factor, a.measure)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 50): print(out.to_string(index=False))
    if a.out: out.to_csv(a.out, index=False); print("written", a.out)
    if a.chart:
        from . import plots as P
        d = runs
        for k, v in (x.split("=", 1) for x in a.where):
            if v: d = d[d[k].isin(v.split(","))]
        fig = (P.fig_bar if a.chart == "bar" else P.fig_box)(d, a.measure, a.x, None if a.hue in ("", "none") else a.hue)
        open(a.chart_out, "wb").write(P.to_png(fig)); print("written", a.chart_out)

if __name__ == "__main__":
    main()
