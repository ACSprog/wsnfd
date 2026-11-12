import numpy as np, pytest
from wsnfd.config import Config, FAULT_TYPES
from wsnfd import run as R, sandbox as S, live as L

@pytest.fixture(scope="module")
def base():
    cfg = Config(grid_s=300, n_estimators=30); X, info = R.prepare_baseline(cfg, "sim", None, dict(n_nodes=30, days=36, seed=0)); return cfg, X, info

@pytest.fixture()
def sb(base):
    cfg, X, info = base; return S.Sandbox(X, info, cfg, seed=1, n_nodes=12)

def run(sb, n):
    for _ in range(n): sb.step()

def test_bulk_add_delete_and_clones_are_labelled(sb, base):
    n0 = len(sb.nodes); new = sb.add_nodes(40)
    assert len(new) == 40 and len(sb.nodes) == n0 + 40 and len(set(sb.nodes)) == len(sb.nodes)       # unique ids
    clones = [n for n in sb.nodes.values() if n.clone]; assert clones and all(n.shift % sb.cfg.epochs_per_day() == 0 for n in clones)
    run(sb, 3); assert sb.log_df().query("clone == 1").source_col.eq(-1).all()
    gone = sb.delete_nodes(new[:25]); assert len(gone) == 25 and not (set(gone) & set(sb.nodes))
    assert sb.delete_nodes([10 ** 6]) == []                                                          # unknown id is ignored
    sb.delete_all(); assert len(sb.nodes) == 0; sb.step()                                            # an empty network must not crash
    assert len(sb.add_nodes(3)) == 3

def test_switching_off_gives_no_reading_and_off_status(sb):
    ids = list(sb.nodes)[:4]; sb.set_on(ids, False); run(sb, 2)
    d = sb.log_df().query("tick == @sb.tick - 1 and node in @ids"); assert d.temperature.isna().all() and (d.on == 0).all()
    assert (sb.status(ids) == L.OFF).all()
    sb.set_on(ids, True); sb.step(); assert not (sb.status(ids) == L.OFF).any()

@pytest.mark.parametrize("ft", FAULT_TYPES)
def test_each_fault_changes_the_reading_as_defined(sb, ft):
    # a clean twin sharing the same feed shows exactly what the fault added
    import copy; twin = copy.deepcopy(sb); i = list(sb.nodes)[0]
    run(sb, 5); run(twin, 5)
    sb.inject([i], ft, ("temperature",), mag=6.0 if ft != "stuck" else None, duration=40, sign=1)
    run(sb, 20); run(twin, 20)
    a = sb.log_df().query("node == @i").temperature.to_numpy(); b = twin.log_df().query("node == @i").temperature.to_numpy()
    ok = np.isfinite(a) & np.isfinite(b); diff = (a - b)[ok][-15:]
    assert sb.log_df().query("node == @i").fault_present.iloc[-1] == 1 and sb.nodes[i].faults[ft]["dur"] == 40
    if ft == "spike": assert np.allclose(diff, diff[0]) and diff[0] > 0         # a 40-tick spike setting is a burst of the same jump
    if ft == "bias": assert np.allclose(diff, diff[0]) and diff[0] > 0
    if ft == "drift": assert diff[-1] > diff[0] >= 0
    if ft == "noise": assert np.std(diff) > 0
    if ft == "stuck": assert np.ptp(a[ok][-10:]) == 0

def test_spike_is_one_tick_and_faults_expire(sb):
    i = list(sb.nodes)[0]; run(sb, 3); sb.inject([i], "spike", mag=5); run(sb, 4)
    f = sb.log_df().query("node == @i").fault_present.to_numpy(); assert f[3] == 1 and f[:3].sum() == 0 and f[4:].sum() == 0 and not sb.nodes[i].faults

def test_clear_data_empties_everything_but_keeps_network_and_faults(sb):
    i = list(sb.nodes)[0]; sb.inject([i], "bias", duration=None); run(sb, 30); n = len(sb.nodes)
    sb.clear_data(); st = sb.stats()
    assert st["recorded_rows"] == 0 and st["ticks"] == 0 and st["tp"] + st["fp"] + st["fn"] + st["tn"] == 0 and len(sb.log_df()) == 0
    assert len(sb.nodes) == n and "bias" in sb.nodes[i].faults
    sb.step(); assert (sb.log_df().flagged == -1).all()                  # detector warms up again: no verdicts from nothing
    run(sb, sb.cfg.roll_window + 2); assert (sb.log_df().flagged >= 0).any()

def test_detector_flags_a_strong_fault_and_stays_quiet_otherwise(sb):
    run(sb, 40); quiet = sb.stats(); assert quiet["alarms"] <= 0.05 * max(1, quiet["tn"] + quiet["fp"])
    ids = list(sb.nodes)[:4]; sb.inject(ids, "bias", ("temperature", "humidity"), mag=6, duration=None); run(sb, 25)
    d = sb.log_df().query("tick >= 55 and node in @ids"); assert d[d.fault_present == 1].flagged.mean() > 0.8

def test_provenance_is_recorded_and_traceable(sb, base):
    cfg, X, info = base; run(sb, 3); d = sb.log_df()
    assert {"source_node", "source_epoch", "source_time"} <= set(d.columns) and d.source_time.str.len().gt(0).all()
    r = d.dropna(subset=["temperature"]).iloc[0]; col = sb.base_ids.index(int(r.source_node))
    assert r.temperature == X[int(r.source_epoch), col, 0]               # the recorded value is the data file's value (no faults here)
    assert not sb.real                                                    # simulated source is declared as such

def test_feed_never_reads_the_training_segment_first(sb):
    assert sb.f == sb.train_end

def test_simultaneous_faults_are_each_counted_and_logged(sb):
    ids = list(sb.nodes)[:3]; run(sb, 15)
    sb.inject(ids, "noise", duration=None); sb.inject(ids, "bias", mag=4, duration=None); run(sb, 10)
    assert sb.by_type["noise"][1] > 0 and sb.by_type["bias"][1] == sb.by_type["noise"][1] and sb.by_type["drift"][1] == 0
    assert "noise+bias" in set(sb.log_df().fault_type)
