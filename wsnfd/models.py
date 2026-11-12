"""Model configurations M1-M4 (Section 4.4). M1-M3 share one scikit-learn forest and one window W."""
import time, pickle
from collections import deque
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

def make_rf(cfg, seed):
    return RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                                  min_samples_leaf=cfg.min_samples_leaf, class_weight="balanced",
                                  n_jobs=1, random_state=seed)

def tune_rf(Xtr, ytr, cfg, seed, grid=None, log=None):
    """Hyperparameters chosen once, on the training segment only, by forward-chaining CV (Section 3.4)."""
    # The number of trees is a fixed standard (cfg.n_estimators, 100 in all reported runs); only depth and leaf size are tuned.
    grid = grid or [dict(n_estimators=cfg.n_estimators, max_depth=d, min_samples_leaf=l) for d in (10, None) for l in (1, 5)]
    best, best_f = None, -1
    for g in grid:
        fs = []
        for tr, va in TimeSeriesSplit(3).split(Xtr):
            m = RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed, **g).fit(Xtr[tr], ytr[tr])
            fs.append(f1_score(ytr[va], m.predict(Xtr[va]), zero_division=0))
        if log is not None: log.append(dict(**g, mean_f1=float(np.mean(fs)), fold_f1=[round(float(f), 4) for f in fs]))
        if np.mean(fs) > best_f: best, best_f = g, float(np.mean(fs))
    return best

class Base:
    name = "base"
    detector = False                      # True for models that raise drift alarms (only M3 records them)
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed
        self.fit_time = 0.0; self.predict_time = 0.0; self.n_updates = 0; self.n_train_rows = 0
        self.labels_consumed = 0; self.update_log = []; self.alarms = []
        self.buf = deque()  # revealed labelled batches: (epoch_end, X, y)
        self.buf_epochs = 0
    def _fit(self, X, y, when):
        t = time.process_time(); self.rf = make_rf(self.cfg, self.seed).fit(X, y)
        dt = time.process_time() - t; self.fit_time += dt
        self.n_updates += 1; self.n_train_rows += len(y); self.update_log.append((when, len(y), dt))
    def init(self, X, y):
        self._fit(X, y, -1); self.n_updates = 0; self.n_train_rows = 0; self.update_log = []; self.fit_time_init = self.fit_time; self.fit_time = 0.0
    def predict(self, X):
        t = time.process_time(); p = self.rf.predict(X); self.predict_time += time.process_time() - t; return p
    def _push(self, when, n_epochs, X, y):
        self.buf.append((when, n_epochs, X, y)); self.buf_epochs += n_epochs
        W = int(self.cfg.window_days * self.cfg.epochs_per_day())
        while self.buf_epochs - self.buf[0][1] >= W and len(self.buf) > 1:
            self.buf_epochs -= self.buf.popleft()[1]
    def _window(self):
        return np.vstack([b[2] for b in self.buf]), np.concatenate([b[3] for b in self.buf])
    def observe(self, when, n_epochs, X, y, yhat):  # labels revealed after the configured delay
        pass
    def model_bytes(self):
        return len(pickle.dumps(self.rf))

class Static(Base):
    name = "M1"

class Periodic(Base):
    name = "M2"
    def __init__(self, cfg, seed):
        super().__init__(cfg, seed); self.last = None; self.P = int(cfg.period_days * cfg.epochs_per_day())
    def observe(self, when, n_epochs, X, y, yhat):
        self.labels_consumed += len(y); self._push(when, n_epochs, X, y)
        if self.last is None: self.last = when - n_epochs        # the clock starts where the test stream starts, not at epoch 0
        if when - self.last >= self.P:
            Xw, yw = self._window()
            if len(np.unique(yw)) > 1: self._fit(Xw, yw, when); self.last = when

class DriftTriggered(Base):
    name = "M3"
    detector = True
    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        from river.drift import ADWIN
        self.det = ADWIN(delta=cfg.adwin_delta); self.cool = 0
    def observe(self, when, n_epochs, X, y, yhat):
        self.labels_consumed += len(y); self._push(when, n_epochs, X, y)
        err = (yhat != y).astype(int); alarm = False
        if self.cfg.adwin_input == "observation":                  # first design: one value per reading (readings of one batch are not independent)
            for e in err:
                self.det.update(int(e))
                if self.det.drift_detected: alarm = True
        else:                                                      # one value per batch: the share of the batch's readings that were wrong
            self.det.update(float(err.mean()))
            alarm = bool(self.det.drift_detected)
        self.cool = max(0, self.cool - 1)
        if alarm:
            self.alarms.append(when)
            if self.cool == 0:
                Xw, yw = self._window()
                if len(np.unique(yw)) > 1: self._fit(Xw, yw, when); self.cool = self.cfg.cooldown_batches

class StreamARF(Base):
    """Optional M4 (and M4b with detector disabled): River Adaptive Random Forest, learned per observation."""
    name = "M4"
    def __init__(self, cfg, seed, detector=True):
        super().__init__(cfg, seed)
        from river.forest import ARFClassifier
        self.name = "M4" if detector else "M4b"
        kw = dict(n_models=10, seed=seed)
        if not detector:  # None selects River's default ADWIN; NoDrift switches detection off (M4b)
            from river.drift import NoDrift
            kw.update(drift_detector=NoDrift(), warning_detector=NoDrift())
        self.rf = ARFClassifier(**kw); self.keys = None
    def _row(self, x): return {i: float(v) for i, v in enumerate(x)}
    def init(self, X, y):
        t = time.process_time()
        for x, yy in zip(X, y): self.rf.learn_one(self._row(x), int(yy))
        self.fit_time_init = time.process_time() - t
    def predict(self, X):
        t = time.process_time(); p = np.array([self.rf.predict_one(self._row(x)) or 0 for x in X]); self.predict_time += time.process_time() - t; return p
    def observe(self, when, n_epochs, X, y, yhat):
        t = time.process_time(); self.labels_consumed += len(y)
        for x, yy in zip(X, y): self.rf.learn_one(self._row(x), int(yy))
        self.fit_time += time.process_time() - t; self.n_train_rows += len(y)
    def model_bytes(self): return len(pickle.dumps(self.rf))

def make_model(name, cfg, seed):
    return {"M1": Static, "M2": Periodic, "M3": DriftTriggered,
            "M4": lambda c, s: StreamARF(c, s, True), "M4b": lambda c, s: StreamARF(c, s, False)}[name](cfg, seed)
