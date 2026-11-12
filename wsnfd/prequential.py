"""Prequential (test-then-train) engine with label delay (Section 4.5)."""
from collections import deque
import numpy as np
from .models import make_model, tune_rf

def run_stream(F, y, ytype, ok, train_end, cfg, model_names, seed, tune=False):
    T, N, nf = F.shape
    Xtr = F[:train_end][ok[:train_end]]; ytr = y[:train_end][ok[:train_end]]
    if tune:
        best = tune_rf(Xtr, ytr, cfg, seed)
        cfg.n_estimators, cfg.max_depth, cfg.min_samples_leaf = best["n_estimators"], best["max_depth"], best["min_samples_leaf"]
    models = {n: make_model(n, cfg, seed) for n in model_names}
    shared = None                                   # M1-M3 start from one identical forest (same segment, same seed)
    for n, m in models.items():
        if n in ("M1", "M2", "M3"):
            if shared is None: m.init(Xtr, ytr); shared = m
            else: m.rf = shared.rf; m.fit_time_init = shared.fit_time_init
        else: m.init(Xtr, ytr)
    if cfg.prefill_buffer:                          # at the start of the test stream M2 and M3 already hold the last W days of the (labelled) training segment
        Wn = int(cfg.window_days * cfg.epochs_per_day())
        for m in models.values():
            if m.name in ("M2", "M3"):
                for s in range(max(0, train_end - Wn), train_end, cfg.batch_epochs):
                    e = min(s + cfg.batch_epochs, train_end); mk = ok[s:e]
                    if mk.any(): m._push(e, e - s, F[s:e][mk], y[s:e][mk])
    B = cfg.batch_epochs; d = int(round(cfg.label_delay_days * cfg.epochs_per_day() / B))
    pending = deque(); out = {n: [] for n in models}; meta = []
    for bi, s in enumerate(range(train_end, T, B)):
        e = min(s + B, T); mask = ok[s:e]
        tt, nn = np.nonzero(mask)
        if len(tt) == 0: continue
        Xb = F[s:e][mask]; yb = y[s:e][mask]
        meta.append(np.column_stack([tt + s, nn, yb, ytype[s:e][mask]]))
        yh = {}
        for n, m in models.items():            # 1) test with the current model
            yh[n] = m.predict(Xb); out[n].append(yh[n])
        pending.append((bi, s, e, Xb, yb, yh))
        while pending and pending[0][0] <= bi - d:   # 2) then reveal labels (after the configured delay) and update
            _, s0, e0, X0, y0, yh0 = pending.popleft()
            for n, m in models.items(): m.observe(e0, e0 - s0, X0, y0, yh0[n])
    meta = np.vstack(meta)
    res = {"meta": meta, "pred": {n: np.concatenate(v) for n, v in out.items()}, "models": models}
    return res
