"""Interactive live network for demonstrations: add or delete nodes (singly or in bulk), switch nodes off and on,
inject each fault type on demand, clear the recorded data, and watch a trained detector respond.

Readings come from the prepared baseline (the real Intel data, or the simulated fallback) played forward in time, so every
recorded row carries its source node and source timestamp. Added nodes beyond the baseline's own nodes are labelled clones
(a baseline series shifted by whole days, with extra sensor noise). Faults follow Table 4.2 (temperature and humidity)."""
from collections import deque, OrderedDict
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from .config import FAULT_TYPES
from . import run as R, features as Fe, faults as Fa, models as Mo, data as D, live as LV

FAULT_DEFAULTS = {  # typical magnitude (in node sigmas) and duration (ticks, None = until cleared) per type
    "spike": (5.0, 1), "noise": (2.5, 30), "bias": (2.0, None), "drift": (3.0, None), "stuck": (0.0, None)}
DRIFT_RAMP_TICKS = 60

@dataclass
class Node:
    id: int; xy: tuple; src: int; shift: int = 0; clone: bool = False; on: bool = True
    faults: dict = field(default_factory=dict)   # type -> dict(chs, mag, sign, start, dur, held)

def train_detector(Xb, info, cfg, seed=0):
    """Fit the Random Forest exactly as the experiments do (scenario S0 training segment), keeping the constants needed online."""
    X, yb, yt, onset, train_end, valid = R.make_stream(Xb, cfg, "S0", seed)
    F, names = Fe.build_features(X, train_end, cfg.feature_set, cfg.roll_window, info.get("neighbours"))
    ok = valid & np.isfinite(F).all(axis=2); tr = ok[:train_end]
    rf = Mo.make_rf(cfg, seed).fit(F[:train_end][tr], yb[:train_end][tr])
    resid = Fe.fit_residual(X, train_end) if cfg.feature_set == "FS-A" else None
    tol = 1e-3 * float(np.nanmean(np.nanstd(X[:train_end, :, :2], axis=0)))
    return dict(rf=rf, resid=resid, tol=tol, names=names, n_train=int(tr.sum()))

class Sandbox:
    def __init__(self, Xb, info, cfg, seed=0, n_nodes=12, train=True, max_log=300_000):
        self.Xb, self.info, self.cfg = Xb, info, cfg
        self.T, self.NB = Xb.shape[0], Xb.shape[1]; self.train_end = int(cfg.train_frac * self.T)
        self.sigma = Fa.node_sigma(Xb, self.train_end); self.rng = np.random.default_rng(seed)
        self.grid_s = cfg.grid_s; self.t0 = pd.Timestamp(info["t0"]) if info.get("t0") else None; self.start_bin = int(info.get("start_bin", 0))
        self.real = info.get("source") == "intel"
        self.base_ids = [int(n) for n in info.get("nodes", range(self.NB))]
        xy = info.get("xy"); a = np.linspace(0, 2 * np.pi, self.NB, endpoint=False)
        self.base_xy = [tuple(p) for p in xy] if xy and len(xy) == self.NB else [tuple(p) for p in np.column_stack([np.cos(a), np.sin(a)]) * 10]
        cov = np.isfinite(Xb[self.train_end:]).all(axis=2).mean(axis=0); self.order = [int(i) for i in np.argsort(-cov)]   # best coverage first
        self.next_clone_id = max(self.base_ids) + 1
        self.L = int(max(60, min(500, 500 * 31 / cfg.grid_s)))
        self.det = train_detector(Xb, info, cfg, seed) if train else None
        self.nodes = OrderedDict(); self.f = self.train_end; self.tick = 0
        self.reset_records(); self.events = []
        self.add_nodes(n_nodes)

    # ---------------------------------------------------------------- records
    def reset_records(self):
        self.log = deque(maxlen=300_000); self.hist = deque(maxlen=600); self.buf = deque(maxlen=self.L)
        self.counts = dict(tp=0, fp=0, fn=0, tn=0); self.by_type = {t: [0, 0] for t in FAULT_TYPES}   # [detected, rows]
        self.n_alarms = 0; self.nb_cache = None

    def clear_data(self):
        """Forget every recorded reading and the detector's history, as if the system had just been switched on. Nodes and faults stay."""
        self.reset_records(); self.events.append((self.tick, "recorded data cleared"))

    def jump_feed(self, where=None):
        """Move the data feed to another time; history is cleared because the stream is discontinuous."""
        self.f = int(where if where is not None else self.rng.integers(self.train_end, self.T)); self.buf.clear()
        self.events.append((self.tick, f"feed jumped to epoch {self.f}"))

    # ---------------------------------------------------------------- network
    def active_cols(self): return {n.src for n in self.nodes.values() if not n.clone}

    def add_nodes(self, k=1, near=None):
        """Add k nodes. Baseline nodes are used first (real data when the source is Intel); then labelled clones. Returns new ids."""
        new = []
        for _ in range(int(k)):
            free = [c for c in self.order if c not in self.active_cols()]
            if free:
                c = free[0]; nid = self.base_ids[c]; self.nodes[nid] = Node(nid, self.base_xy[c], c)
            else:
                srcs = [n.src for n in self.nodes.values()] or self.order[:1]; c = int(self.rng.choice(srcs)); nid = self.next_clone_id; self.next_clone_id += 1
                shift = int(self.rng.integers(1, max(2, self.T // self.cfg.epochs_per_day()))) * self.cfg.epochs_per_day()
                bx, by = self.base_xy[c]; sp = max(1e-6, np.ptp([p[0] for p in self.base_xy]) / 20)
                self.nodes[nid] = Node(nid, (bx + self.rng.normal(0, sp), by + self.rng.normal(0, sp)), c, shift, True)
            new.append(nid)
        self.nb_cache = None; self.events.append((self.tick, f"added {len(new)} node(s)")); return new

    def delete_nodes(self, ids):
        ids = [i for i in ids if i in self.nodes]
        for i in ids: del self.nodes[i]
        self.nb_cache = None; self.events.append((self.tick, f"deleted {len(ids)} node(s)")); return ids

    def delete_all(self): return self.delete_nodes(list(self.nodes))

    def set_on(self, ids, flag):
        for i in ids:
            if i in self.nodes: self.nodes[i].on = bool(flag)
        self.events.append((self.tick, f"switched {'on' if flag else 'off'} {len(ids)} node(s)"))

    def inject(self, ids, ftype, channels=("temperature",), mag=None, duration="default", sign=0):
        """Start a fault on the given nodes. mag in node sigmas; duration in ticks or None (until cleared); sign +1/-1 or 0 for random."""
        assert ftype in FAULT_TYPES
        dm, dd = FAULT_DEFAULTS[ftype]; mag = dm if mag is None else float(mag); dur = dd if duration == "default" else duration
        chs = [{"temperature": 0, "humidity": 1}[c] for c in channels]; n = 0
        for i in ids:
            if i not in self.nodes: continue
            sg = sign if sign else int(self.rng.choice([-1, 1]))
            self.nodes[i].faults[ftype] = dict(chs=chs, mag=mag, sign=sg, start=self.tick, dur=dur, held={}); n += 1
        self.events.append((self.tick, f"{ftype} on {n} node(s)")); return n

    def clear_faults(self, ids=None):
        for i in (list(self.nodes) if ids is None else ids):
            if i in self.nodes: self.nodes[i].faults.clear()
        self.events.append((self.tick, "faults cleared"))

    # ---------------------------------------------------------------- one tick
    def _reading(self, n):
        """(vector of 4 or NaN row, true-fault flag, list of active fault types, source epoch)."""
        fs = (self.f + n.shift) % self.T; nan = np.full(4, np.nan)
        if not n.on: return nan, 0, [], fs
        v = self.Xb[fs, n.src].copy()
        if not np.isfinite(v).all(): return nan, 0, [], fs
        if n.clone: v[:2] += self.rng.normal(0, 0.2, 2) * self.sigma[n.src]
        names = []
        for ft in list(n.faults):
            F = n.faults[ft]; age = self.tick - F["start"]
            if F["dur"] is not None and age >= F["dur"]: del n.faults[ft]; continue
            for c in F["chs"]:
                sg = self.sigma[n.src, c] * F["mag"] * F["sign"]
                if ft == "spike": v[c] += sg
                elif ft == "noise": v[c] += self.rng.normal(0, abs(sg))
                elif ft == "bias": v[c] += sg
                elif ft == "drift": v[c] += sg * min(age + 1, F["dur"] or DRIFT_RAMP_TICKS) / (F["dur"] or DRIFT_RAMP_TICKS)
                else: v[c] = F["held"].setdefault(c, v[c])
            names.append(ft)
        return v, int(bool(names)), names, fs

    def _neighbours(self, ids):
        key = tuple(ids)
        if self.nb_cache and self.nb_cache[0] == key: return self.nb_cache[1]
        xy = {i: self.nodes[i].xy for i in ids}; nb = D.knn_neighbours(xy, ids, self.cfg.knn); self.nb_cache = (key, nb); return nb

    def _detect(self, ids):
        """Predictions for the newest epoch: array of -1 (cannot decide), 0 or 1 per node."""
        N = len(ids); pred = np.full(N, -1, np.int8)
        if self.det is None or N < 2 or len(self.buf) < self.cfg.roll_window: return pred
        X = np.full((len(self.buf), N, 4), np.nan)
        for li, d in enumerate(self.buf):
            for ni, i in enumerate(ids):
                v = d.get(i)
                if v is not None: X[li, ni] = v
        F, _ = Fe.build_features(X, 0, self.cfg.feature_set, self.cfg.roll_window, self._neighbours(ids), self.det["resid"], self.det["tol"])
        row = F[-1]; ok = np.isfinite(row).all(axis=1)
        if ok.any(): pred[ok] = self.det["rf"].predict(row[ok])
        return pred

    def step(self):
        ids = list(self.nodes); rd = [self._reading(self.nodes[i]) for i in ids]
        self.buf.append({i: r[0] for i, r in zip(ids, rd)})
        pred = self._detect(ids); h = {}
        for k, i in enumerate(ids):
            v, y, yt, fs = rd[k]; n = self.nodes[i]; p = int(pred[k]); has = bool(np.isfinite(v).all())
            if has and p >= 0:
                self.counts["tp" if (y and p) else "fp" if p and not y else "fn" if y else "tn"] += 1
                for ft in yt: self.by_type[ft][1] += 1; self.by_type[ft][0] += p      # every active fault type is credited
                self.n_alarms += p
            stamp = (self.t0 + pd.Timedelta(seconds=(self.start_bin + fs) * self.grid_s)) if self.t0 is not None else None
            self.log.append((self.tick, i, *v, int(n.on), y, "+".join(yt), p, int(n.clone), n.src if not n.clone else -1, self.base_ids[n.src], fs, str(stamp) if stamp is not None else ""))
            h[i] = (v[0], v[1], y, p, int(n.on))
        self.hist.append(h); self.tick += 1; self.f += 1
        if self.f >= self.T: self.f = self.train_end; self.buf.clear(); self.events.append((self.tick, "feed reached its end and restarted"))
        return pred

    # ---------------------------------------------------------------- views
    def status(self, ids=None):
        """Status code per node for the most recent tick (uses the history of the last step)."""
        ids = ids or list(self.nodes); out = []
        for i in ids:
            n = self.nodes[i]; h = self.hist[-1].get(i) if self.hist else None
            if not n.on: out.append(LV.OFF)
            elif h is None or not np.isfinite(h[0]): out.append(LV.NO_DATA if h is not None else LV.WARM)
            elif h[3] < 0: out.append(LV.WARM)
            else: y, p = h[2], h[3]; out.append(LV.TP if y and p else LV.FP if p else LV.FN if y else LV.TN)
        return np.array(out, np.int8)

    def stats(self):
        c = self.counts; d = 2 * c["tp"] + c["fp"] + c["fn"]
        return dict(**c, f1=(2 * c["tp"] / d) if d else float("nan"), alarms=self.n_alarms, recorded_rows=len(self.log), ticks=len(self.hist))

    COLS = ["tick", "node", "temperature", "humidity", "light", "voltage", "on", "fault_present", "fault_type", "flagged", "clone", "source_col", "source_node", "source_epoch", "source_time"]
    def log_df(self): return pd.DataFrame(list(self.log), columns=self.COLS)
