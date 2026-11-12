"""Causal feature construction (Section 3.4). Every feature at time t uses observations up to t only."""
import numpy as np, pandas as pd

def _runlength(x, tol):
    s = pd.Series(x); same = (s.diff().abs() < tol).fillna(False)
    return same.groupby((~same).cumsum()).cumsum().to_numpy()

def fit_residual(X, train_end):
    """Linear temperature-humidity relation estimated on the training segment only."""
    tr = X[:train_end, :, :2].reshape(-1, 2); tr = tr[~np.isnan(tr).any(axis=1)]
    b, a = np.polyfit(tr[:, 0], tr[:, 1], 1)
    return a, b

def build_features(X, train_end, feature_set="FS-A", window=10, neighbours=None, resid=None, tol=None):
    """Return (F [T,N,nf], names). FS-B: instantaneous readings only. FS-A: adds temporal and spatial families."""
    T, N, C = X.shape
    names = ["temperature", "humidity", "light", "voltage"]
    F = [X[:, :, c] for c in range(4)]
    if feature_set == "FS-B":
        return np.stack(F, axis=2), names
    a, b = resid if resid is not None else fit_residual(X, train_end)
    if tol is None: tol = 1e-3 * np.nanmean(np.nanstd(X[:train_end, :, :2], axis=0))   # fixed value can be passed for live use
    for c, nm in enumerate(["temperature", "humidity"]):
        d1 = np.full((T, N), np.nan); rm = np.full((T, N), np.nan); rs = np.full((T, N), np.nan); rl = np.zeros((T, N))
        for n in range(N):
            s = pd.Series(X[:, n, c])
            d1[:, n] = s.diff().to_numpy()
            r = s.rolling(window, min_periods=2)
            rm[:, n] = r.mean().to_numpy(); rs[:, n] = r.std().to_numpy()
            rl[:, n] = _runlength(X[:, n, c], tol)
        F += [d1, rm, rs, rl]; names += [f"{nm}_diff1", f"{nm}_rmean", f"{nm}_rstd", f"{nm}_runlen"]
        dev = np.full((T, N), np.nan)
        for n in range(N):
            nb = neighbours[n] if neighbours is not None else [k for k in range(N) if k != n]
            with np.errstate(all="ignore"):
                med = np.nanmedian(X[:, nb, c], axis=1)
            dev[:, n] = X[:, n, c] - med
        F.append(dev); names.append(f"{nm}_nbdev")
    F.append(X[:, :, 1] - (a + b * X[:, :, 0])); names.append("th_residual")
    return np.stack(F, axis=2), names
