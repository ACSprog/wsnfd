# Test and run procedure (workbench)

Work in the Streamlit workbench (`streamlit run app.py`). Follow the stages in order; each has a pass condition. Every run saves a folder containing `runs.csv`, `blocks.csv` and `environment.json`; keep them all. The "Equivalent command line" box on the Run tab shows the exact command for any run, so it can be pasted into your records or repeated later. Everything in the workbench can also be done from the command line with the same results (table in `README.md`); `run_all.sh` runs the stages that way and is optional.

## 0. Setup (once)
1. `pip install -r requirements.txt`, then `streamlit run app.py`.
2. Run the automated tests once from a terminal: `python -m pytest tests` (about 15 minutes). **Pass:** all tests pass. Repeat after any change to the code.

## 1. Load and check the data (tab 1, sidebar)
1. Source: *Intel Berkeley file*. Use Browse, Upload or paste the path of `data.txt.gz`; `mote_locs.txt` is picked up if it sits next to it (otherwise Download it).
2. Time grid: **31 s** for the main runs (coarser grids only for pilots).
3. Read tab 1: readings and motes read, what cleaning removed (implausible values and the battery rule, cut-off 2.2 V in the sidebar), and the selection rule (10 nodes, 28 days, 80% coverage). **Pass:** the rule is satisfied. If not, read the table of looser rules, and if you relax the rule record the new rule and the reason in Chapter 3.

## 2. Pilot calibration (tab 2, sidebar)
Pilot settings: grid 300 s or 600 s, scenarios S0 and S2, models M1 and M3, 3 seeds, FS-A, label delay 0, a different "Save this run to" folder each time (e.g. `results/pilot/mag_0.8`).
1. Run one pilot for each *Fault magnitude multiplier* (0.6, 0.8, 1.0, 1.25, 1.5). In tab 3 (Workspace) load the pilot folders together and compare M1 on S0. **Choose the multiplier at which M1 has non-trivial but unsaturated F1** (Section 4.3.1).
2. With that multiplier, try ADWIN delta and Cooldown. In S2, M3 should raise alarms after the change and few alarms in S0 (the Workspace summary shows alarms and false alarms for M3). Change one setting at a time.
3. **Pass:** one value each for multiplier, ADWIN delta and cooldown; P and W as specified in Chapter 4. Write them down. **Pilot results are calibration, not findings; do not report them.**

## 3. Main experiments (tab 2)
Design defaults for every reported run: the retraining buffer of M2 and M3 is prefilled with the last W days of the training segment, and M3's ADWIN receives one value per batch. Do not use --cold-start or --adwin-input observation (they reproduce the first design only).
Settings that stay fixed from now on: the values chosen in stage 2, Time grid 31 s, Fault rate 0.10, Seeds 10, scenarios S0 to S4, **Trees 100 in every run** (the number of trees is never tuned; only depth and leaf size are).
1. **FS-A:** Models M1, M2, M3 (M4 and M4b are excluded from the study); Trees 100; tick **Choose the forest settings once (--tune)**; Label delay 0; save to `results/FS-A-Main-Run`; Run experiments.
2. **FS-B:** same settings, Feature set FS-B, save to `results/FS-B-Main-Run`.
3. Long runs: keep the computer awake and the browser tab open (each run shows progress by scenario and seed; results are written as they complete).
4. **Pass:** each folder has the three files; `runs.csv` has scenarios x seeds x models rows; alarm measures are empty for M1 and M2 by design.

## 4. Label-delay and sensitivity runs (tab 2)
Same fixed settings as the main run of the same feature set, including Trees 100 and **--tune ticked**: the tuning uses only the training segment, which does not depend on label delay, so it reproduces the depth and leaf size of the main run (check them in `environment.json`).
- **Label delay** (models M2 and M3; M1 is unaffected and is taken from the main run), for each feature set: `Delay-1d-FS-B`, `Delay-3d-FS-B`, `Delay-1d-FS-A`, `Delay-3d-FS-A`.
- **Design-choice sensitivity**, on each feature set in turn (FS-B first, then FS-A), with the feature-set suffix in every folder name: `Sens-P0.5` and `Sens-P2` (M2, P = 0.5 and 2 days); `Sens-W1` and `Sens-W7` (M2 and M3, W = 1 and 7 days); `Sens-ADWIN0.001`, `Sens-ADWIN0.01` and `Sens-ADWIN0.05` (M3, ADWIN confidence). Change one setting at a time; all else as in the main run.
- **Optional, only if time allows:** `Sens-Fault5` and `Sens-Fault20` (all three models, fault rate 0.05 and 0.20); detector-trigger runs `Sens-ObsInput`, `Sens-ObsInput-CD6` and `Sens-ADWIN0.2-CD6` (all three models; the first two set the detector input to one value per reading); and the equal-budget control `Sens-P0.25` (all three models, P = 0.25 day).
- The authoritative list of runs, with the settings and the record of what was run when, is `Run_Plan.docx`.
- Every run: Trees 100, --tune ticked, 10 seeds, S0 to S4, one run at a time with the computer otherwise idle.
- **Pass:** one folder per setting.

## 5. Analysis and tables (tabs 3 to 6; each tab shows its command line equivalent, e.g. `python -m wsnfd.analyse results/fsA results/fsB --what tests --margin 0.02`)
1. Workspace: load `fsA` and `fsB` (and the delay and sensitivity folders) together; check the filters.
2. Statistics: Friedman, Holm-corrected Wilcoxon, effect sizes, the "worthwhile" check, cost and gain per CPU second.
3. Explorer: any extra comparison (e.g. recall by fault type, sensitivity across the experiments).
4. Export: *Build export bundle* gives all tables (CSV), the Excel workbook, figures and a zip.
5. **Pass:** every table and figure cited in Chapter 6 exists and matches the numbers quoted in the text.

## 6. Reproducibility check
Repeat one scenario with seed count 1 and the same settings, saving to `results/repeat`; compare the S0 (or chosen scenario) seed-0 row with `results/fsA/runs.csv`. Detection measures must match exactly; CPU times will differ slightly. Keep `environment.json` of every folder: it records parameters, the selection outcome, library versions and hardware (reported in Chapters 5 and 6).

## 7. Keep for the dissertation
The frozen settings and why; the pilot comparison; every results folder; the export bundle; the exact commands shown on the Run tab.

## Parity check (once)
Run any experiment in tab 2, copy the command from "Equivalent command line", run it in a terminal with a different `--out`, and compare the two `runs.csv` files: precision, recall, F1 and update counts must be identical (CPU times differ). This is also tested automatically.

## Quick check of the machinery
Source *Simulated (fallback)*, grid 600 s, 2 seeds, 20 trees. This tests the workflow only; never report it.

## Result folders
Each run saves to its own folder (`results/run1`, `run2`, ...; the interface proposes the next free name). A run never replaces an existing folder unless you tick "Overwrite this folder" (interface) or add `--overwrite` (command line, or `OVERWRITE=1 ./run_all.sh STAGE`). Give main runs a descriptive name, for example `results/FS-B-Main-Run`. In the Workspace tab nothing is loaded until you pick the folders; do not load pilot folders together with main runs.
