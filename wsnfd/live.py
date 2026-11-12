"""Live replay of an experiment: per-node, per-epoch ground truth and model decisions, for the interface's Live view."""
import numpy as np
from . import run as R, features as Fe
from .prequential import run_stream

# node status codes for a chosen model at one epoch
NO_DATA, TN, TP, FN, FP, WARM, OFF = 0, 1, 2, 3, 4, 5, 6
STATUS_NAMES = {NO_DATA: "no data", TN: "normal, not flagged", TP: "fault detected", FN: "fault missed", FP: "false alarm", WARM: "warming up", OFF: "switched off"}
STATUS_COLORS = {NO_DATA: "#d9d9d9", TN: "#66a61e", TP: "#1f77b4", FN: "#d7191c", FP: "#fdae61", WARM: "#f2f2f2", OFF: "#404040"}

def layout_xy(info, n):
    """Node coordinates: real or simulated floor plan if known, otherwise a ring."""
    xy = info.get("xy")
    if xy and len(xy) == n: return np.array(xy, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False); return np.column_stack([np.cos(a), np.sin(a)])

def build_replay(Xbase, info, cfg, scenario, seed, models, neighbours=None):
    """Run the models exactly as in one_run, keeping predictions per epoch and node for the test stream."""
    X, yb, yt, onset, train_end, valid = R.make_stream(Xbase, cfg, scenario, seed)
    F, names = Fe.build_features(X, train_end, cfg.feature_set, cfg.roll_window, neighbours)
    ok = valid & np.isfinite(F).all(axis=2)
    res = run_stream(F, yb, yt, ok, train_end, cfg, models, seed, False)
    meta = res["meta"]; T, N = yb.shape
    preds = {}
    for m, p in res["pred"].items():
        g = np.full((T, N), -1, np.int8); g[meta[:, 0], meta[:, 1]] = p; preds[m] = g
    events = {m: [int(u[0]) for u in getattr(mod, "update_log", [])] for m, mod in res["models"].items()}
    alarms = {m: [int(a) for a in getattr(mod, "alarms", [])] for m, mod in res["models"].items()}
    # cumulative confusion counts per epoch for fast trailing-window F1
    cum = {}
    for m, g in preds.items():
        yv = yb.astype(bool); pv = g == 1; has = g >= 0
        tp = (yv & pv & has).sum(1); fp = (~yv & pv & has).sum(1); fn = (yv & ~pv & has).sum(1)
        cum[m] = tuple(np.concatenate([[0], np.cumsum(a)]) for a in (tp, fp, fn))
    return dict(X=X, valid=valid, y=yb, ytype=yt, preds=preds, onset=int(onset), train_end=int(train_end), T=int(T), N=int(N),
                events=events, alarms=alarms, cum=cum, xy=layout_xy(info, N), nodes=list(info.get("nodes", range(N))),
                t0=info.get("t0"), start_bin=int(info.get("start_bin", 0)), grid_s=cfg.grid_s, scenario=scenario, seed=seed,
                epochs_per_day=cfg.epochs_per_day())

def frame_status(rp, model, t):
    """Status code of every node at epoch t for one model."""
    g = rp["preds"][model][t]; y = rp["y"][t].astype(bool); p = g == 1
    s = np.full(rp["N"], TN, np.int8); s[y & p] = TP; s[y & ~p] = FN; s[~y & p] = FP; s[g < 0] = NO_DATA
    return s

def trailing_f1(rp, model, t, window):
    """F1 over epochs (t-window, t]; NaN if there were no faults and no alarms."""
    tp, fp, fn = rp["cum"][model]; a = max(rp["train_end"], t - window + 1); b = t + 1
    TP_, FP_, FN_ = tp[b] - tp[a], fp[b] - fp[a], fn[b] - fn[a]
    d = 2 * TP_ + FP_ + FN_
    return float(2 * TP_ / d) if d else float("nan")

def f1_series(rp, model, upto, window):
    return np.array([trailing_f1(rp, model, t, window) for t in range(rp["train_end"], upto + 1)])
