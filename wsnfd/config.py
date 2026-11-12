from dataclasses import dataclass

CHANNELS = ["temperature", "humidity", "light", "voltage"]
FAULT_TYPES = ["spike", "noise", "bias", "drift", "stuck"]  # type ids 1..5; 0 = normal

@dataclass
class Config:
    # data
    grid_s: int = 31                  # alignment grid in seconds (Intel nominal sampling interval)
    temp_min: float = 10.0            # plausibility bounds for an indoor laboratory (Section 3.4)
    temp_max: float = 40.0
    hum_min: float = 5.0
    hum_max: float = 95.0
    dead_voltage: float = 2.2         # battery cut-off (V): a mote is treated as degraded from its first reading below this (Section 3.4)
    max_gap_bins: int = 5             # longest gap filled by linear interpolation
    min_nodes: int = 10               # selection rule (Section 3.3)
    min_days: int = 28
    min_coverage: float = 0.80
    train_frac: float = 0.28          # initial training segment (Section 3.4)
    # faults (Section 4.3.1)
    fault_rate: float = 0.10
    mag_scale: float = 1.0            # pilot calibration multiplier on fault magnitudes
    # change (Section 4.3.2)
    transition_days: float = 5.0
    # features
    roll_window: int = 10
    knn: int = 5                      # nearest neighbours for the neighbour-deviation feature
    # models (Section 4.4)
    feature_set: str = "FS-A"         # "FS-A" or "FS-B"
    n_estimators: int = 100
    max_depth: object = None
    min_samples_leaf: int = 1
    period_days: float = 1.0          # P
    window_days: float = 3.0          # W
    adwin_delta: float = 0.002
    cooldown_batches: int = 24
    prefill_buffer: bool = True       # M2 and M3 start with the last W days of the training segment in their retraining buffer (False: empty buffer, the first design)
    adwin_input: str = "batch"        # M3: ADWIN receives one value per batch, the batch's mean error ("observation": one value per reading, the first design)
    label_delay_days: float = 0.0
    # protocol (Section 4.5)
    batch_epochs: int = 120           # micro-batch length (about one hour at 31 s)
    block_days: float = 1.0
    n_seeds: int = 10

    def bounds(self):
        return {"temperature": (self.temp_min, self.temp_max), "humidity": (self.hum_min, self.hum_max), "light": (0.0, 2500.0), "voltage": (2.0, 3.2)}

    def epochs_per_day(self):
        return int(round(86400 / self.grid_s))
