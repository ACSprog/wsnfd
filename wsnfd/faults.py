"""Node-level fault injection with binary and type labels (Section 4.3.1, Table 4.2)."""
import numpy as np
from .config import FAULT_TYPES

def node_sigma(X, train_end):
    """Per-node, per-channel training-segment standard deviation (temperature, humidity)."""
    s = np.nanstd(X[:train_end, :, :2], axis=0)
    return np.where(np.isfinite(s) & (s > 0), s, np.nanmean(s[s > 0]))

def _durations(cfg):
    k = 31.0 / cfg.grid_s                              # Table 4.2 durations are given in 31-s epochs
    f = lambda lo, hi: (max(1, int(round(lo * k))), max(1, int(round(hi * k))))
    return {"spike": (1, 3), "noise": f(20, 100), "bias": f(100, 1000), "drift": f(200, 1500), "stuck": f(50, 500)}

def inject(X, valid, sigma, seg, rng, cfg):
    """Inject faults into X[seg] in place. Returns (y_bin, y_type) arrays of shape [T,N] for the segment."""
    a, b = seg; T, N = X.shape[:2]
    y_type = np.zeros((T, N), dtype=np.int8)
    occupied = np.zeros((T, N), bool)
    target = cfg.fault_rate * valid[a:b].sum()
    dur = _durations(cfg); m = cfg.mag_scale; count = 0; tries = 0
    while count < target and tries < 200000:
        tries += 1
        ftype = FAULT_TYPES[rng.integers(0, 5)]
        lo, hi = dur[ftype]; d = int(rng.integers(lo, hi + 1))
        if b - a <= d: continue
        n = int(rng.integers(0, N)); ch = int(rng.integers(0, 2)); s = int(rng.integers(a, b - d))
        e = s + d
        if occupied[s:e, n].any() or not valid[s:e, n].all():
            continue
        sg = sigma[n, ch]; x = X[s:e, n, ch]
        if ftype == "spike":
            x += rng.choice([-1, 1]) * rng.uniform(3, 8) * m * sg
        elif ftype == "noise":
            x += rng.normal(0, rng.uniform(1, 4) * m * sg, d)
        elif ftype == "bias":
            x += rng.choice([-1, 1]) * rng.uniform(1, 3) * m * sg
        elif ftype == "drift":
            x += np.linspace(0, rng.choice([-1, 1]) * rng.uniform(1, 4) * m * sg, d)
        else:  # stuck-at: hold the value at onset
            x[:] = x[0]
        X[s:e, n, ch] = x
        y_type[s:e, n] = FAULT_TYPES.index(ftype) + 1; occupied[s:e, n] = True
        count += d
    return (y_type > 0).astype(np.int8), y_type
