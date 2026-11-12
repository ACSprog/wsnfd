"""Friedman test, pairwise Wilcoxon signed-rank with Holm correction, bootstrap effect sizes (Section 3.5)."""
import itertools, numpy as np
from scipy import stats

def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    meds = [np.median(rng.choice(d, len(d))) for _ in range(n)]
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

def compare(table, margin=0.02):
    """table: dict model -> array of per-seed values (paired by seed)."""
    names = list(table); A = np.column_stack([table[n] for n in names]); out = {"friedman_p": np.nan, "pairs": []}
    if A.shape[0] >= 3 and A.shape[1] >= 3 and np.ptp(A, axis=1).any():
        out["friedman_p"] = float(stats.friedmanchisquare(*A.T).pvalue)
    pairs = list(itertools.combinations(range(len(names)), 2)); ps = []
    for i, j in pairs:
        d = A[:, i] - A[:, j]
        try: ps.append(float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        except ValueError: ps.append(1.0)
    adj = holm(ps) if ps else []
    for (i, j), p, pa in zip(pairs, ps, adj):
        med, lo, hi = boot_ci(A[:, i] - A[:, j]); dd = A[:, i] - A[:, j]
        dz = float(dd.mean() / dd.std(ddof=1)) if len(dd) > 1 and dd.std(ddof=1) > 0 else float('nan')
        out["pairs"].append(dict(a=names[i], b=names[j], p=p, p_holm=float(pa), median_diff=med, ci_lo=lo, ci_hi=hi, cohen_dz=dz, n_pairs=len(dd), beyond_margin=abs(med) > margin))
    return out
