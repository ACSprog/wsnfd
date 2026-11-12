"""Loading, simulation, cleaning, alignment and node/period selection."""
import numpy as np, pandas as pd
from .config import CHANNELS

INTEL_MOTE_IDS = (1, 54)       # the dataset description states that mote ids range from 1 to 54; other ids come from corrupted rows
# Plausibility bounds for an indoor laboratory (Section 3.4). Temperature and humidity are tighter than the physical limits because, in the Intel data,
# readings above 35 degrees C or outside 5-95% occur only while a mote's battery is failing.
BOUNDS = {"temperature": (10.0, 40.0), "humidity": (5.0, 95.0), "light": (0.0, 2500.0), "voltage": (2.0, 3.2)}
# Battery cut-off: a mote is treated as degraded from its first reading below it. 2.2 V is the strictest of 2.1-2.4 V at which the selection rule is still met on the full Intel file.
DEAD_VOLTAGE = 2.2
INTEL_COLS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

def load_intel(path):
    """Read the Intel Berkeley file (data.txt or data.txt.gz). Malformed rows are skipped."""
    try:                                                   # fast C parser: seconds for the 2.3-million-row file
        df = pd.read_csv(path, sep=r"\s+", names=INTEL_COLS, on_bad_lines="skip", dtype=str)
    except Exception:                                      # unusual file: fall back to the tolerant but slow parser
        df = pd.read_csv(path, sep=r"\s+", names=INTEL_COLS, engine="python", on_bad_lines="skip", dtype=str)
    for c in ["epoch", "moteid"] + CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["ts", "moteid"]).drop_duplicates(subset=["ts", "moteid"])
    df["moteid"] = df["moteid"].astype(int)
    return df[["ts", "moteid"] + CHANNELS].sort_values(["ts", "moteid"]).reset_index(drop=True)

def _bursts(T, p_good_loss, rng, mean_good=500, mean_bad=40, p_bad_loss=0.9):
    """Gilbert-Elliott packet loss: alternating good and bad link states of geometric length. True = delivered."""
    deliver = np.empty(T, bool); i = 0; good = rng.random() < 0.8
    while i < T:
        L = int(rng.geometric(1.0 / (mean_good if good else mean_bad)))
        loss = p_good_loss if good else p_bad_loss
        deliver[i:i + L] = rng.random(min(L, T - i)) > loss; i += L; good = not good
    return deliver

def make_layout(n_nodes, rng, width=40.0, height=30.0):
    """Node coordinates (metres) on a rectangular floor plan; nodes near two walls are treated as window nodes."""
    xy = np.column_stack([rng.uniform(1, width - 1, n_nodes), rng.uniform(1, height - 1, n_nodes)])
    window = (xy[:, 1] < 0.2 * height) | (xy[:, 0] > 0.85 * width)
    return xy, window

def simulate(n_nodes=12, days=30, grid_s=31, seed=0, dropout=True, return_coords=False, length_scale=10.0,
             start="2004-03-01"):
    """Simulated fallback generator (Section 3.3), written to mimic an indoor laboratory WSN:
    - nodes on a floor plan; static offsets and fast fluctuations are spatially correlated
      (exponential covariance, length_scale metres); window nodes see larger daily swings and daylight;
    - daily and weekly cycles, slow weather component, HVAC oscillation, weekday occupancy heat;
    - humidity moves against temperature with node-specific sensitivity plus a shared moisture component;
    - light follows office hours and daylight (window nodes) with cloud cover;
    - each node's battery declines, some die early; as voltage sags below 2.4 V the temperature and
      humidity readings become erratic (natural anomalies that cleaning must remove);
    - bursty packet loss (two-state link), late start-up of some nodes, small timestamp jitter.
    All values are modelling assumptions, not fitted to the Intel data; tune them once real data are inspected."""
    rng = np.random.default_rng(seed)
    T = int(days * 86400 / grid_s); t = np.arange(T) * grid_s
    xy, window = make_layout(n_nodes, rng)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    Lc = np.linalg.cholesky(np.exp(-d / length_scale) + 1e-6 * np.eye(n_nodes))
    hour = (t / 3600.0) % 24; dow = ((t // 86400) + pd.Timestamp(start).dayofweek) % 7
    workday = (dow < 5).astype(float)
    occupied = workday * ((hour > 8) & (hour < 18)) + 0.2 * (1 - workday) * ((hour > 10) & (hour < 15))
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)
    def ar1(sd, tau_s, size=None):
        phi = np.exp(-grid_s / tau_s)
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), (T,) if size is None else (T, size))
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -phi], e, axis=0)
    weather = ar1(1.0, 2 * 86400.0); moisture = ar1(2.0, 3 * 86400.0)
    cloud = np.clip(0.75 + 0.25 * ar1(1.0, 6 * 3600.0), 0.15, 1.0)
    local = ar1(0.3, 1200.0, n_nodes) @ Lc.T
    site = Lc @ rng.normal(0, 0.8, n_nodes)
    hvac_phase = rng.uniform(0, 2 * np.pi, n_nodes)
    rows = []
    for n in range(n_nodes):
        amp = 1.2 + 1.8 * window[n]
        temp = (21 + site[n] + amp * diurnal + 0.6 * occupied + weather
                + 0.3 * np.sin(2 * np.pi * t / 1500 + hvac_phase[n]) + local[:, n] + rng.normal(0, 0.06, T))
        hum = 40 - 1.6 * rng.normal(1.0, 0.15) * (temp - 21) + moisture + rng.normal(0, 0.4, T)
        light = (rng.uniform(80, 220) * occupied + window[n] * 600 * np.clip(diurnal, 0, None) * cloud) * np.exp(rng.normal(0, 0.05, T))
        death = int(T * rng.uniform(0.55, 0.97)) if (dropout and rng.random() < 0.25) else T + 1
        v0 = rng.normal(2.72, 0.03); frac = np.minimum(np.arange(T) / min(death, T), 1.0)
        volt = v0 - (v0 - 2.05) * frac ** rng.uniform(3, 6) + rng.normal(0, 0.003, T)
        bad = (volt < 2.4) & (rng.random(T) < np.clip((2.4 - volt) * 1.5, 0, 0.6))
        temp[bad] = rng.uniform(35, 125, bad.sum()); hum[bad] = rng.uniform(-5, 110, bad.sum())
        keep = _bursts(T, rng.uniform(0.03, 0.30), rng) if dropout else np.ones(T, bool)
        keep[min(death, T):] = False
        if dropout and rng.random() < 0.2: keep[: int(rng.integers(0, 3 * 86400 // grid_s))] = False
        ts = pd.Timestamp(start) + pd.to_timedelta(t + rng.uniform(0, grid_s) + rng.normal(0, 1.5, T), unit="s")
        rows.append(pd.DataFrame({"ts": ts[keep], "moteid": n + 1, "temperature": temp[keep], "humidity": hum[keep],
                                  "light": np.clip(light, 0, None)[keep], "voltage": volt[keep]}))
    df = pd.concat(rows).sort_values(["ts", "moteid"]).reset_index(drop=True)
    return (df, {n + 1: tuple(xy[n]) for n in range(n_nodes)}) if return_coords else df

def load_locations(path):
    """Intel mote_locs.txt: 'moteid x y' per line. Returns {moteid: (x, y)}."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            try: out[int(p[0])] = (float(p[1]), float(p[2]))
            except ValueError: pass
    return out

def knn_neighbours(coords, node_ids, k=5):
    """For each selected node, indices (within node_ids) of its k nearest selected nodes."""
    xy = np.array([coords[int(n)] for n in node_ids]); d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return [[int(i) for i in np.argsort(d[r])[:min(k, len(node_ids) - 1)]] for r in range(len(node_ids))]

def clean(df, bounds=BOUNDS, dead_voltage=DEAD_VOLTAGE):
    """Remove physically implausible readings and exhausted-battery node-periods (Section 3.4)."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    dead = df.groupby("moteid")["voltage"].transform(lambda v: (v < dead_voltage).cummax())
    df.loc[dead, CHANNELS] = np.nan
    return df

def align(df, grid_s):
    """Place observations on a common grid; returns (array [T,N,4], node ids, grid start)."""
    t0 = df["ts"].min().floor(f"{grid_s}s")
    df = df.assign(bin=((df["ts"] - t0).dt.total_seconds() // grid_s).astype(int))
    nodes = sorted(df["moteid"].unique()); T = int(df["bin"].max()) + 1
    X = np.full((T, len(nodes), len(CHANNELS)), np.nan)
    g = df.groupby(["bin", "moteid"])[CHANNELS].mean()
    bi = g.index.get_level_values(0).to_numpy(); ni = np.searchsorted(nodes, g.index.get_level_values(1).to_numpy())
    X[bi, ni, :] = g.to_numpy()
    return X, np.array(nodes), t0

def fill_short_gaps(X, max_gap):
    """Linear interpolation of gaps up to max_gap bins; longer gaps stay NaN (no forward fill)."""
    X = X.copy()
    for n in range(X.shape[1]):
        for c in range(X.shape[2]):
            s = pd.Series(X[:, n, c])
            X[:, n, c] = s.interpolate(method="linear", limit=max_gap, limit_area="inside").to_numpy()
    return X

def select_nodes_period(X, grid_s, min_nodes, min_days, min_cov, stride_days=1):
    """Decision rule of Section 3.3. Prefers the longest period (down to min_days, in one-week steps)
    in which at least min_nodes nodes have >= min_cov of temperature and humidity readings present.
    Returns (node indices, start, end) or None if the rule fails."""
    epd = int(round(86400 / grid_s)); T = X.shape[0]; Lmin = int(min_days * epd)
    present = ~np.isnan(X[:, :, :2]).any(axis=2)
    csum = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(present, axis=0)])
    # window lengths tried, longest first: the whole record, then one week shorter each time, and finally exactly min_days
    lengths = (list(range(T, Lmin, -7 * epd)) + [Lmin]) if T >= Lmin else []
    for L in lengths:
        best = None
        for s in range(0, T - L + 1, stride_days * epd):
            cov = (csum[s + L] - csum[s]) / L
            ok = np.where(cov >= min_cov)[0]
            if best is None or len(ok) > len(best[0]):
                best = (ok, s, s + L)
        if best is not None and len(best[0]) >= min_nodes:
            return best
    return None


def cleaning_sweep(df, cfg, cutoffs=(2.1, 2.2, 2.3, 2.4), bound_sets=None):
    """Compare cleaning rules on the same raw data: for each battery cut-off and set of plausibility bounds, the share of temperature readings
    removed, the most nodes meeting the coverage rule in one shared window of the minimum length, and whether the selection rule is met."""
    import dataclasses
    bound_sets = bound_sets or {"tight (10-40 C, 5-95 %)": dict(temp_min=10.0, temp_max=40.0, hum_min=5.0, hum_max=95.0),
                                "loose (0-50 C, 0-100 %)": dict(temp_min=0.0, temp_max=50.0, hum_min=0.0, hum_max=100.0)}
    epd = int(round(86400 / cfg.grid_s)); rows = []
    for bname, bs in bound_sets.items():
        for v in cutoffs:
            c = dataclasses.replace(cfg, dead_voltage=float(v), **bs)
            cl = clean(df, bounds=c.bounds(), dead_voltage=c.dead_voltage)
            X, nodes, t0 = align(cl, c.grid_s); X = fill_short_gaps(X, c.max_gap_bins)
            L = int(c.min_days * epd); T = X.shape[0]; present = ~np.isnan(X[:, :, :2]).any(axis=2)
            csum = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(present, axis=0)])
            best = max((int(((csum[s + L] - csum[s]) / L >= c.min_coverage).sum()) for s in range(0, max(T - L, 0) + 1, epd)), default=0) if T >= L else 0
            sel = select_nodes_period(X, c.grid_s, c.min_nodes, c.min_days, c.min_coverage)
            rows.append({"bounds": bname, "battery_cutoff_V": float(v), "temperature_readings_removed_%": round(100 * float(cl["temperature"].isna().mean()), 1),
                         "nodes_in_best_window": best, "rule_met": sel is not None,
                         "nodes_kept": len(sel[0]) if sel else 0, "window_days": round((sel[2] - sel[1]) / epd, 1) if sel else 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ diagnostics: why a file passes or fails the selection rule
def raw_summary(df):
    """What was read from the file: rows, motes and the time span."""
    return dict(rows=int(len(df)), motes=int(df["moteid"].nunique()), first=str(df["ts"].min()), last=str(df["ts"].max()))

def clean_report(df, bounds=BOUNDS, dead_voltage=DEAD_VOLTAGE):
    """What cleaning removes. Returns (per-channel out-of-range counts, per-mote table including when the battery rule starts)."""
    d = df.copy(); oor = {}
    for c, (lo, hi) in bounds.items():
        m = (d[c] < lo) | (d[c] > hi); oor[c] = int(m.sum()); d.loc[m, c] = np.nan
    low = d["voltage"] < dead_voltage
    d = d.assign(low=low); d["dead"] = d.groupby("moteid")["low"].cummax()
    g = d.groupby("moteid")
    tbl = pd.DataFrame({"readings": g.size(), "first_reading": g["ts"].min(), "last_reading": g["ts"].max(),
                        "battery_rule_from": d.loc[low].groupby("moteid")["ts"].min(), "rows_removed_by_battery_rule": g["dead"].sum().astype(int)}).reset_index()
    return oor, tbl

def diagnose_selection(X, nodes, t0, cfg, raw=None, clean_tbl=None, oor=None,
                       days_list=(28, 21, 14, 7), cov_list=(0.9, 0.8, 0.7, 0.6, 0.5)):
    """Everything needed to see why fewer than cfg.min_nodes nodes meet the rule, and what looser rules would give."""
    epd = int(round(86400 / cfg.grid_s)); T, N = X.shape[0], X.shape[1]
    present = ~np.isnan(X[:, :, :2]).any(axis=2)
    csum = np.vstack([np.zeros((1, N)), np.cumsum(present, axis=0)])
    def windows(L):
        if L > T or L < 1: return np.zeros((0, N))
        return np.array([(csum[s + L] - csum[s]) / L for s in range(0, T - L + 1, epd)])
    relax = pd.DataFrame({f"{int(round(c * 100))}%": [int((windows(d * epd) >= c).sum(axis=1).max()) if T >= d * epd else 0 for d in days_list] for c in cov_list},
                         index=[f"{d} days" for d in days_list]); relax.index.name = "window length"
    Wm = windows(int(cfg.min_days * epd)); best = Wm.max(axis=0) if len(Wm) else np.zeros(N)
    stamp = lambda i: str(t0 + pd.Timedelta(seconds=int(i) * cfg.grid_s))
    first = [stamp(np.argmax(present[:, n])) if present[:, n].any() else "" for n in range(N)]
    last = [stamp(T - 1 - np.argmax(present[::-1, n])) if present[:, n].any() else "" for n in range(N)]
    nodes_tbl = pd.DataFrame({"moteid": nodes.astype(int), "bins_with_data": present.sum(axis=0), "overall_coverage": present.mean(axis=0),
                              "first_data": first, "last_data": last, f"best_{cfg.min_days:g}_day_coverage": best,
                              "meets_rule_in_some_window": best >= cfg.min_coverage}).sort_values(f"best_{cfg.min_days:g}_day_coverage", ascending=False).reset_index(drop=True)
    best_common = int((Wm >= cfg.min_coverage).sum(axis=1).max()) if len(Wm) else 0     # most nodes meeting the rule in ONE shared window
    sug = []                                                                           # per window length: the strictest coverage that reaches min_nodes
    for d in days_list:
        for c in sorted(cov_list, reverse=True):
            n = int(relax.loc[f"{d} days", f"{int(round(c * 100))}%"])
            if n >= cfg.min_nodes: sug.append((d, c, n)); break
    return dict(rule=dict(min_nodes=cfg.min_nodes, min_days=cfg.min_days, min_coverage=cfg.min_coverage, grid_s=cfg.grid_s),
                span_days=T / epd, nodes_meeting_rule=int(nodes_tbl["meets_rule_in_some_window"].sum()), best_common_nodes=best_common, suggestions=sug,
                raw=raw or {}, out_of_range=oor or {}, relax=relax, nodes=nodes_tbl, cleaning=clean_tbl if clean_tbl is not None else pd.DataFrame())
