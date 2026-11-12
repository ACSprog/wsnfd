"""Figures shared by the interface and the report command. Every function returns a matplotlib Figure."""
import io
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import analyse as A

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

PALETTE = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]

def to_png(fig, dpi=200):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig); return buf.getvalue()

def fig_blocks(blocks, scenario, onset_block=None, label=None, band=True):
    d = A.block_curves(blocks[blocks.scenario == scenario] if label is None else blocks[(blocks.scenario == scenario) & (blocks.run_label == label)])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (m, g) in enumerate(d.groupby("model")):
        ax.plot(g.block, g["mean"], label=m, color=PALETTE[i % 6], lw=1.6)
        if band: ax.fill_between(g.block, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), alpha=.12, color=PALETTE[i % 6])
    if onset_block is not None: ax.axvline(onset_block, color="k", ls="--", lw=1, label="change onset")
    ax.set_xlabel("block (days after the start of the test stream)"); ax.set_ylabel("F1"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{scenario}" + (f" ({label})" if label else "")); ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False); ax.grid(alpha=.25)      # legend below the plot so it never covers the curves
    fig.tight_layout()
    return fig

def fig_bar(runs, measure, x, hue=None, title=None):
    """Mean with 95% CI error bars."""
    same = hue == x                                           # colouring by the x variable itself: one colour per bar, no legend needed
    if same: hue = None
    d = A.explore(runs, measure, [x] + ([hue] if hue else []), ("mean", "ci95"))
    xs = list(dict.fromkeys(d[x])); hs = list(dict.fromkeys(d[hue])) if hue else [None]
    w = 0.8 / len(hs); fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, h in enumerate(hs):
        g = d if h is None else d[d[hue] == h]; g = g.set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + j * w, g["mean"], w, yerr=g["ci95"].fillna(0), capsize=3, label=None if h is None else str(h),
               color=[PALETTE[i % 6] for i in range(len(xs))] if same else PALETTE[j % 6])
    ax.set_xticks(np.arange(len(xs)) + w * (len(hs) - 1) / 2); ax.set_xticklabels([str(v) for v in xs]); ax.set_ylabel(measure)
    ax.set_title(title or f"{measure} by {x}" + (f" and {hue}" if hue else "") + " (mean, 95% CI)"); ax.grid(axis="y", alpha=.25)
    if hue: ax.legend(fontsize=8)
    return fig

def fig_box(runs, measure, x, hue=None, title=None):
    same = hue == x
    if same: hue = None
    xs = list(dict.fromkeys(runs[x])); hs = list(dict.fromkeys(runs[hue])) if hue else [None]
    w = 0.8 / len(hs); fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, h in enumerate(hs):
        data = [runs[(runs[x] == v) & ((runs[hue] == h) if hue else True)][measure].dropna().to_numpy() for v in xs]
        bp = ax.boxplot(data, positions=np.arange(len(xs)) + j * w, widths=w * 0.9, patch_artist=True, manage_ticks=False)
        for i, b in enumerate(bp["boxes"]): b.set_facecolor(PALETTE[(i if same else j) % 6]); b.set_alpha(.55)
        if hue: ax.plot([], [], color=PALETTE[j % 6], lw=6, alpha=.55, label=str(h))
    ax.set_xticks(np.arange(len(xs)) + w * (len(hs) - 1) / 2); ax.set_xticklabels([str(v) for v in xs]); ax.set_ylabel(measure)
    ax.set_title(title or f"{measure} by {x}" + (f" and {hue}" if hue else "") + " (box plot over seeds)"); ax.grid(axis="y", alpha=.25)
    if hue: ax.legend(fontsize=8)
    return fig

def fig_recall(runs):
    cols = [c for c in runs.columns if c.startswith("recall_")]
    d = runs.groupby("model")[cols].mean(); fig, ax = plt.subplots(figsize=(8, 3.4)); w = 0.8 / len(d)
    for j, (m, r) in enumerate(d.iterrows()): ax.bar(np.arange(len(cols)) + j * w, r.values, w, label=m, color=PALETTE[j % 6])
    ax.set_xticks(np.arange(len(cols)) + w * (len(d) - 1) / 2); ax.set_xticklabels([c.replace("recall_", "") for c in cols])
    ax.set_ylabel("recall"); ax.set_ylim(0, 1.02); ax.set_title("Recall by fault type (mean over scenarios and seeds)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    return fig

def fig_cost(runs):
    d = runs.groupby("model")[["fit_time_s", "predict_time_s"]].mean(); fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(d.index, d.fit_time_s, label="fitting / updating", color=PALETTE[0]); ax.bar(d.index, d.predict_time_s, bottom=d.fit_time_s, label="prediction", color=PALETTE[1])
    ax.set_ylabel("CPU seconds per run"); ax.set_title("Computational cost (mean)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    return fig
