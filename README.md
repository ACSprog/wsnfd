# wsnfd: adaptive fault detection in IoT sensor networks

Experimental pipeline and analysis workbench for the MSc study "Adaptive Machine Learning for Fault Detection in IoT
Sensor Networks". Implements Chapters 3 to 5: controlled network-wide change, node-level fault injection, causal features,
three same-learner Random Forest configurations (M1 static, M2 periodic, M3 ADWIN-triggered), optional River ARF (M4/M4b),
a prequential engine with label delay, and the measures, tests and exports of Chapter 6.

Two ways to work. They use the same code and give identical outputs.

## Interface and command line do the same things

Every function of the workbench has a command-line equivalent that uses the same code and gives the same results; each tab shows its equivalent command in an "Equivalent command line" box.

| Function | Workbench | Command line |
|---|---|---|
| Load data, selection-rule report | Sidebar, tab 1 | `python -m wsnfd.run --source ... --check-data` |
| Export the labelled dataset (csv, mat, parquet) | Tab 1 | `python -m wsnfd.export --scenario S2 --seed 0 [--features] --out dataset.csv` |
| Run experiments (all settings, tuning) | Tab 2 | `python -m wsnfd.run ... [--tune]` |
| Summaries, recovery, recall by type, block curves | Tab 3 | `python -m wsnfd.analyse FOLDERS --what summary\|recovery\|recall\|blocks` |
| Any measure, any grouping, charts, sensitivity | Tab 4 | `python -m wsnfd.analyse FOLDERS --what explore\|sensitivity ... [--chart bar\|box]` |
| Friedman, Holm-Wilcoxon, "worthwhile", change effects against S0, cost, gain | Tab 5 | `python -m wsnfd.analyse FOLDERS --what tests\|worthwhile\|change\|cost\|gain` |
| All tables, workbook, figures, zip | Tab 6 | `python -m wsnfd.report FOLDERS --out report_out` |
| Live view, live sandbox | Tabs 7 and 8 | none: interactive demonstrations that produce no results for the dissertation |

Settings accept the same values in both: any time grid, seed count, fault rate, label delay, fault-magnitude multiplier, P, W, ADWIN delta, cooldown, detector input and number of trees.

The two routes have different starting values in the boxes (the workbench opens on the simulated source with a 300 s grid, 3 seeds, 30 trees and scenarios S0 and S2, for a quick check; the command line defaults to the Intel-scale values 31 s, 10 seeds, 100 trees and all five scenarios). Set every value explicitly for a reported run and read it back from `environment.json`.

## A. Interface (workbench)
    pip install -r requirements.txt
    streamlit run app.py
1. **Data and dataset export**: choose the source, see the selection-rule result, plot channels, and download the labelled
   dataset (baseline or any scenario/seed, optionally with features) as CSV for other software.
2. **Run**: set scenarios, models, seeds, feature set, label delay, P, W, ADWIN delta, cooldown, detector input and number of trees. The equivalent command line is shown.
3. **Workspace**: load several result folders side by side (for example FS-A and FS-B, or different label delays), filter, see summaries
   with 95% CI, block-F1 curves with the change onset, recall by fault type, recovery.
4. **Explorer**: any measure, any grouping, chosen statistics, bar or box charts, sensitivity across experiments. Every table and chart downloads.
5. **Statistics**: Friedman and Holm-corrected Wilcoxon with effect sizes, the Section 3.5 "worthwhile" check, cost and gain per CPU second.
6. **Export**: one click builds all tables (CSV), an Excel workbook, all figures (PNG) and a zip.
7. **Live view** and 8. **Live sandbox**: described below.

## B. Manual route (command line and Python)
    python -m pytest tests                                     # 99 tests
    python -m wsnfd.run --source intel --path data.txt.gz --locs mote_locs.txt --feature-set FS-A --tune --out results/fsA     # --tune: choose the forest hyperparameters once on the training segment
    python -m wsnfd.run --source intel --path data.txt.gz --locs mote_locs.txt --feature-set FS-B --out results/fsB
    python -m wsnfd.report results/fsA results/fsB --out report_out        # tables, results.xlsx, figures, bundle.zip
    python -m wsnfd.run --source intel --path data.txt --locs mote_locs.txt --feature-set FS-A --tune-only --out results/Tuning-FS-A     # scores of every depth and leaf-size combination, written to tuning_scores.csv; nothing else is run
    # defaults: M2 and M3 start with the last W days of the training segment in their buffer, and M3's ADWIN receives one value per batch; the first design is reproduced with --cold-start --adwin-input observation (not used for reported runs)
    # staged procedure and pass conditions: see RUNBOOK.md and run_all.sh; the Run tab has every setting (including fault-magnitude multiplier and --tune); matching command-line flags: --mag-scale --period-days --window-days --adwin-delta --cooldown
    python -m wsnfd.export --source intel --path data.txt.gz --scenario S2 --seed 0 --features --out dataset.csv
    # simulated fallback or a quick pilot: --source sim --grid-s 600 --scenarios S0,S2 --seeds 3 --trees 20
In Python:

    import pandas as pd; from wsnfd import analyse as A
    runs, blocks = A.load_folders(["results/fsA", "results/fsB"])
    A.summary(runs); A.tests(runs); A.worthwhile(A.tests(runs), runs); A.explore(runs, "post_change_f1", ["scenario", "model"])

Outputs of a run: `runs.csv` (one row per scenario x seed x model), `blocks.csv` (block-F1 curves), `environment.json` (parameters, selection outcome, library versions, hardware; written by both the command line and the workbench). Drift-alarm measures (`n_alarms`, `false_alarms`, `detection_delay`, in epochs) are recorded for M3 only; other models show them as empty, not zero.
Intel data: http://db.csail.mit.edu/labdata/labdata.html (data.txt.gz, mote_locs.txt).

## Simulator (fallback)
Floor plan with spatially correlated offsets and fluctuations, window nodes, daily and weekly cycles, HVAC oscillation, occupancy heat,
cloud-modulated daylight, battery decline with erratic readings on brown-out, bursty packet loss, early node death and late start-up.
Parameters are modelling assumptions, not fitted to the Intel data. Defaults (30 nodes, 36 days) satisfy the selection rule.

## Status
Complete. The reported experiments were run on the Intel Berkeley file (10 nodes, 78,036 epochs on a 31 s grid), and every
results folder records its settings, selection outcome, library versions and hardware in `environment.json`. The simulated
source remains available as a fallback and for quick checks of the machinery; its output must not be reported as a finding.

## Live view (tab 7)
Replays the test stream of one scenario and seed, epoch by epoch, with the nodes drawn on the floor plan (real `mote_locs.txt` coordinates, or the simulator's layout). Each node is coloured by the chosen model's decision: normal, fault detected, fault missed, false alarm, or no reading. Up to three models are shown side by side with a trailing-F1 curve that marks the change and each model update. Use Play, or drag the position slider. It is a replay of the experiment, not a hardware connection. Engine: `wsnfd/live.py`.

## Hand-over to other software
The export bundle (Export tab, or `python -m wsnfd.report`) now contains, besides CSV tables, figures and `results.xlsx`:
`results.mat` (MATLAB, one struct per table), `analysis.ipynb` (Jupyter, runs on the exported tables), and `load_in_matlab.m`, `load_in_R.R`, `load_in_python.py`.
The labelled dataset can be exported as CSV (Excel, SPSS, Stata, Origin, Tableau, Power BI, R), `.mat` (MATLAB) or Parquet (Python, R, Power BI), from the Data tab or `python -m wsnfd.export --out dataset.mat`. Excel holds at most 1,048,576 rows; use a coarser grid, test-stream-only, or mat/parquet for larger sets. Engine: `wsnfd/interop.py`. Parquet needs `pyarrow`, the .mat export needs `scipy` (already required).

## Live sandbox (tab 8)
A hands-on network for demonstrations. Create the network, then press Run.
- **Nodes:** add any number (single or bulk), delete the selected nodes or all of them, switch the selected nodes off and on. Select nodes in the list, or use Select all, Select none, Select k at random.
- **Faults:** buttons for spike, noise, bias, drift and stuck-at on the selected nodes. Choose the channel (temperature, humidity), magnitude (in node standard deviations), duration (typical, fixed number of ticks, until cleared) and direction. Several faults can run on one node. Clear faults on the selected nodes or on all.
- **Recorded data:** "Clear recorded data" empties the log and the detector's history while nodes and faults stay, so the audience watches data appear from nothing (the detector warms up again). Every row records its source node, source epoch and source timestamp; with the Intel file loaded these can be checked against `data.txt`. The recording can be downloaded as CSV. "Jump feed to a random time" shows the feed is not scripted.
- **Detector:** the Random Forest is trained once, on the training segment, exactly as in the experiments (S0, current feature set). The feed starts after the training segment, so the detector has not seen the readings it is judging.
- **Honesty about data:** with the simulated fallback the readings are simulated, and the tab says so. Nodes beyond the prepared data's own nodes are clones (a real series shifted by whole days plus a little sensor noise) and are marked `*`.
Engine: `wsnfd/sandbox.py` (tests: `tests/test_sandbox.py`).

## Choosing the Intel files
In the sidebar, choose "Intel Berkeley file", then use any of:
- **Browse**: opens the normal file window (Windows Explorer, Finder) and fills in the path. It runs on the computer that runs the app, which is your own computer when you start the app locally.
- **Upload**: drag the file in. It is copied into an `uploads` folder next to the app (uploads up to 1 GB, see `.streamlit/config.toml`).
- **Paste a path**: the file, or its folder. Quotes from "Copy as path" are accepted. `mote_locs.txt` is picked up automatically when it sits next to the data file.
The loader reads the 2.3-million-row file with the fast parser (seconds), falling back to a slower tolerant one. Engine: `wsnfd/filepick.py`.

## When the selection rule fails on your file
The first tab then shows a report: how many readings and motes were read (the real Intel file has about 2.3 million readings from 54 motes, 28 Feb to 5 Apr 2004), what cleaning removed (impossible values and the battery rule, cut-off 2.2 V), each node's coverage, and a grid of how many nodes would qualify under looser rules (shorter window, lower coverage). The rule itself can be changed in the sidebar under "Selection rule (Chapter 3)", or on the command line with `--min-nodes`, `--min-days`, `--min-coverage`. Changing it changes your method: record the new rule and the reason in Chapter 3.
`mote_locs.txt`: the "Download it" button fetches the original from the Intel Lab site (needs internet) and saves it next to the data file. Engine: `wsnfd/fetch.py`; reports: `diagnose_selection` and `clean_report` in `wsnfd/data.py`.

### How the selection rule works
It tries window lengths from the whole record downwards in one-week steps and finally exactly `min_days`, and keeps the longest window in which at least `min_nodes` nodes have at least `min_coverage` of temperature and humidity present. The nodes must satisfy this in the same window. For the Intel file, readings with mote ids outside 1 to 54 (the dataset documents ids 1 to 54) are ignored and counted in the report.

## Cleaning (Section 3.4)
Readings are set to missing when they fall outside indoor-laboratory plausibility bounds (temperature 10-40 C, humidity 5-95%, light 0-2500 lux, voltage 2.0-3.2 V), and a mote is treated as degraded from its first reading below the battery cut-off (default 2.2 V; sidebar "Battery cut-off", command line `--dead-voltage`; bounds: sidebar and `--temp-range`, `--hum-range`). On the full Intel file, 2.2 V is the strictest of the cut-offs 2.1 to 2.4 V at which the selection rule is still met (2.3 V and above leave fewer than 10 nodes). Changing the cleaning changes your method: record it and the reason in Chapter 3. The values are stored with each run in `environment.json`.

To choose the cleaning, compare several rules on your own file: tab 1, "Compare cleaning rules", or `python -m wsnfd.run --source intel --path data.txt.gz --sweep-cleaning`. The table shows, for battery cut-offs of 2.1 to 2.4 V with tight and loose bounds, the share of temperature readings removed and the nodes that still meet the selection rule.

## Older results without block counts
Results written before blocks carried observation counts are scored by dropping empty trailing blocks and the final scored block. To exclude every thin block, add the counts without rerunning any model: Workspace tab, "Older results without block counts" (button), or
`python -m wsnfd.run --source intel --path data.txt --locs mote_locs.txt --grid-s 31 --add-block-counts results/FS-B-Main-Run` (same data, grid, selection and cleaning flags as the run). The counts are checked against every run's confusion counts; on any mismatch nothing is written. The original curves are kept as `blocks_before_counts.csv`.
