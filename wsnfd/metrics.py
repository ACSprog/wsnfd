"""Detection, adaptation and cost measures (Sections 3.5 and 4.5)."""
import numpy as np

def confusion(y, p):
    y = y.astype(bool); p = p.astype(bool)
    return int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())

def scores(y, p):
    tp, fp, fn, tn = confusion(y, p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    n = tp + fp + fn + tn; acc = (tp + tn) / n if n else 0.0     # secondary measure: misleading when faults are a minority (Section 3.5)
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr, mcc=mcc, accuracy=acc, tp=tp, fp=fp, fn=fn, tn=tn)

def block_f1(t, y, p, train_end, block_epochs, n_blocks):
    idx = (t - train_end) // block_epochs; out = np.full(n_blocks, np.nan)
    for b in range(n_blocks):
        m = idx == b
        if m.sum(): out[b] = scores(y[m], p[m])["f1"]
    return out

def block_counts(t, y, train_end, block_epochs, n_blocks):
    """Observations scored and true faults in each block (a thin block, e.g. when sensor batteries have died, is not a reliable F1)."""
    idx = (t - train_end) // block_epochs
    n = np.bincount(idx[(idx >= 0) & (idx < n_blocks)], minlength=n_blocks).astype(float)
    pos = np.bincount(idx[(idx >= 0) & (idx < n_blocks) & (np.asarray(y) == 1)], minlength=n_blocks).astype(float)
    return n[:n_blocks], pos[:n_blocks]

MIN_BLOCK_SHARE = 0.5      # a block is used only if it holds at least this share of the median block's observations
EARLY_BLOCKS = 3           # degradation is measured over this many blocks after the change, the same length as the pre-change level

def usable_blocks(n_obs, share=MIN_BLOCK_SHARE):
    """Boolean mask of blocks with enough observations. With no counts (older results) only the final, usually partial, block is dropped."""
    if n_obs is None:
        m = np.ones(0, bool); return m
    n = np.asarray(n_obs, float); med = np.nanmedian(n[n > 0]) if (n > 0).any() else 0.0
    return (n >= share * med) & (n > 0)

def adaptation(bf, onset_block, tol=0.95, usable=None, early=EARLY_BLOCKS):
    """Pre-change level, degradation, recovery time (blocks; NaN = not recovered) and post-change F1, on usable blocks only.
    Degradation = pre-change level minus the mean F1 of the first `early` usable blocks after the change."""
    bf = np.asarray(bf, float).copy()
    if usable is not None and len(usable) == len(bf): bf[~np.asarray(usable)] = np.nan
    pre = np.nanmean(bf[max(0, onset_block - 3):onset_block]) if onset_block > 0 and np.isfinite(bf[max(0, onset_block - 3):onset_block]).any() else np.nan
    post = bf[onset_block:]; ok = post[np.isfinite(post)]
    deg = pre - np.nanmean(ok[:early]) if ok.size else np.nan
    rec = np.nan; thr = tol * pre
    for k in range(len(post) - 1):
        if post[k] >= thr and post[k + 1] >= thr: rec = k; break
    return dict(pre_change_f1=pre, degradation=deg, recovery_blocks=rec, post_change_f1=np.nanmean(ok) if ok.size else np.nan)

def per_type_recall(ytype, y, p, names):
    return {f"recall_{nm}": float(p[ytype == i + 1].mean()) if (ytype == i + 1).any() else np.nan for i, nm in enumerate(names)}

def alarm_stats(alarms, onset):
    a = np.array(alarms)
    post = a[a >= onset]
    return dict(n_alarms=len(a), false_alarms=int((a < onset).sum()), detection_delay=float(post[0] - onset) if post.size else np.nan)
