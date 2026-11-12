"""Export the labelled dataset behind an experiment, for analysis in other software.
    python -m wsnfd.export --source sim --scenario S2 --seed 0 --out dataset.csv [--features] [--test-only]
Long format: one row per node-epoch with the readings the models see (after change and fault injection), the labels,
the segment and, optionally, the features. Use --scenario baseline for the cleaned data before change and faults."""
import argparse
import numpy as np, pandas as pd
from .config import Config, CHANNELS, FAULT_TYPES
from . import run as R, features as Fe, interop as IO

def build_dataset(Xbase, info, cfg, scenario="S0", seed=0, features=False, test_only=False, neighbours=None):
    T, N, _ = Xbase.shape; t0 = pd.Timestamp(info["t0"]); start = int(info.get("start_bin", 0)); nodes = info.get("nodes", list(range(N)))
    if scenario == "baseline":
        X = Xbase; yb = np.zeros((T, N), np.int8); yt = yb.copy(); onset = -1; train_end = int(cfg.train_frac * T)
        valid = np.isfinite(Xbase).all(axis=2)
    else:
        X, yb, yt, onset, train_end, valid = R.make_stream(Xbase, cfg, scenario, seed)
    cols = {c: X[:, :, i] for i, c in enumerate(CHANNELS)}
    if features:
        F, names = Fe.build_features(X, train_end, cfg.feature_set, cfg.roll_window, neighbours)
        cols.update({n: F[:, :, i] for i, n in enumerate(names) if n not in CHANNELS})
    a = train_end if test_only else 0
    ep = np.arange(a, T); tt = np.repeat(ep, N); nn = np.tile(np.arange(N), len(ep))
    df = pd.DataFrame({"time": t0 + pd.to_timedelta((start + tt) * cfg.grid_s, unit="s"), "epoch": tt, "node": np.array(nodes)[nn],
                       "segment": np.where(tt < train_end, "train", "test"), "scenario": scenario, "seed": seed,
                       "valid": valid[a:].ravel().astype(int), "after_change": (tt >= onset).astype(int) if onset >= 0 else 0,
                       "y_fault": yb[a:].ravel(), "fault_type": np.array(["normal"] + FAULT_TYPES)[yt[a:].ravel()]})
    for c, v in cols.items(): df[c] = v[a:].ravel()
    return df[df.valid == 1].reset_index(drop=True)

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="sim"); ap.add_argument("--path"); ap.add_argument("--locs")
    ap.add_argument("--scenario", default="S0"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--features", action="store_true"); ap.add_argument("--test-only", action="store_true"); ap.add_argument("--feature-set", default="FS-A")
    ap.add_argument("--grid-s", type=int, default=31); ap.add_argument("--sim-nodes", type=int, default=30); ap.add_argument("--sim-days", type=int, default=36)
    ap.add_argument("--sim-seed", type=int, default=0); ap.add_argument("--fault-rate", type=float, default=0.10); ap.add_argument("--mag-scale", type=float, default=1.0)
    ap.add_argument("--min-nodes", type=int, default=10); ap.add_argument("--min-days", type=float, default=28); ap.add_argument("--min-coverage", type=float, default=0.80); ap.add_argument("--dead-voltage", type=float, default=2.2); ap.add_argument("--temp-range", default="10,40"); ap.add_argument("--hum-range", default="5,95")
    a = ap.parse_args(argv)
    cfg = Config(grid_s=a.grid_s, feature_set=a.feature_set, fault_rate=a.fault_rate, mag_scale=a.mag_scale, min_nodes=a.min_nodes, min_days=a.min_days, min_coverage=a.min_coverage, dead_voltage=a.dead_voltage,
                 temp_min=float(a.temp_range.split(',')[0]), temp_max=float(a.temp_range.split(',')[1]), hum_min=float(a.hum_range.split(',')[0]), hum_max=float(a.hum_range.split(',')[1]))
    X, info = R.prepare_baseline(cfg, a.source, a.path, dict(n_nodes=a.sim_nodes, days=a.sim_days, seed=a.sim_seed), a.locs)
    if X is None: raise SystemExit("Selection rule failed; use the simulated fallback.")
    df = build_dataset(X, info, cfg, a.scenario, a.seed, a.features, a.test_only, info.get("neighbours")); ext = a.out.rsplit(".", 1)[-1].lower()
    if ext == "mat": IO.dataset_to_mat(df, a.out)
    elif ext == "parquet": df.to_parquet(a.out, index=False)
    else: df.to_csv(a.out, index=False)
    print(a.out, len(df), "rows")

if __name__ == "__main__":
    main()
