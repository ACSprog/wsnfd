"""Controlled network-wide change (Section 4.3.2). Applies to all nodes; observations stay labelled normal."""
import numpy as np

SCENARIOS = {  # id: (kind, severity in pooled standard deviations)
    "S0": (None, 0.0), "S1": ("abrupt", 1.0), "S2": ("abrupt", 3.0), "S3": ("gradual", 1.0), "S4": ("gradual", 3.0),
}

def apply_change(X, scenario, train_end, cfg):
    """Shift temperature and humidity of every node. Humidity moves along the training-segment
    temperature-humidity correlation so that the cross-channel relation is preserved.
    Returns (X', onset_index)."""
    X = X.copy(); T = X.shape[0]
    onset = train_end + (T - train_end) // 2
    kind, sev = SCENARIOS[scenario]
    if kind is None:
        return X, onset
    tr = X[:train_end, :, :2].reshape(-1, 2); tr = tr[~np.isnan(tr).any(axis=1)]
    sd = tr.std(axis=0); sign = np.sign(np.corrcoef(tr[:, 0], tr[:, 1])[0, 1]) or 1.0
    amp = np.array([sev * sd[0], sign * sev * sd[1]])
    ramp = np.zeros(T)
    if kind == "abrupt":
        ramp[onset:] = 1.0
    else:
        L = int(cfg.transition_days * cfg.epochs_per_day())
        ramp[onset:onset + L] = np.linspace(0, 1, min(L, T - onset)); ramp[onset + L:] = 1.0
    X[:, :, 0] += ramp[:, None] * amp[0]
    X[:, :, 1] += ramp[:, None] * amp[1]
    return X, onset
