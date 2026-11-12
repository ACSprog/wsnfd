#!/usr/bin/env bash
# Reproduces the thirty-two experiments of Appendix A. Each run writes to the folder
# name used there and in the wsnfd_reports bundles, so Chapter 6 can be regenerated.
# Usage:  ./run_all.sh STAGE     STAGE = check | pilot | main | delay | sensitivity | faultrate | exploratory | analyse | report | experiments | all
# Run one stage at a time and read the output before the next. See RUNBOOK.md.
set -euo pipefail
OVR="${OVERWRITE:+--overwrite}"      # OVERWRITE=1 replaces folders that already hold results
DATA=${DATA:-data.txt.gz}; LOCS=${LOCS:-mote_locs.txt}        # Intel Berkeley files
SRC="--source intel --path $DATA --locs $LOCS"                # simulated fallback: SRC="--source sim"
# ---- shared by every reported run; frozen after the pilot ----
BASE="$SRC --grid-s 31 --seeds 10 --trees 100 --mag-scale 1.0 --tune"
mkdir -p results logs

run () {  # run NAME <flags...>
  local name=$1; shift
  if [ -z "${OVR}" ] && [ -d "results/$name" ]; then echo "skip $name (exists; OVERWRITE=1 to replace)"; return; fi
  echo "== $name"
  python -m wsnfd.run $OVR $BASE "$@" --out "results/$name" 2>&1 | tee "logs/$name.log"
}

stage=${1:-}
case "$stage" in
check)       python -m pytest tests -q 2>&1 | tee logs/01_tests.log                 # expect 99 passed (about 15 min)
             python -m wsnfd.run $OVR $SRC --scenarios S0 --seeds 1 --models M1 --out results/_selection_check 2>&1 | tee logs/02_selection.log ;;
pilot)       for m in 0.6 0.8 1.0 1.25 1.5; do                                      # calibrate fault magnitude on S0 and S2
               python -m wsnfd.run $OVR $SRC --scenarios S0,S2 --seeds 3 --mag-scale $m --out results/pilot_mag_$m 2>&1 | tee logs/pilot_mag_$m.log; done ;;
main)        # Runs 1-2: main runs, both feature sets
             run FS-A-Main-Run            --scenarios S0,S2,S1,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run FS-B-Main-Run            --scenarios S0,S2,S1,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 ;;
delay)       # Runs 3-6: label-delay analysis
             run Delay-1d-FS-A            --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 1.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Delay-1d-FS-B            --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 1.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Delay-3d-FS-A            --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 3.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Delay-3d-FS-B            --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 3.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 ;;
sensitivity) # Runs 7-20: period, window and detector-confidence sensitivity
             run Sens-ADWIN0.001-FS-A     --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.001 --cooldown 24
             run Sens-ADWIN0.001-FS-B     --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 0.1 --window-days 3.0 --adwin-delta 0.001 --cooldown 24
             run Sens-ADWIN0.01-FS-A      --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.01 --cooldown 24
             run Sens-ADWIN0.01-FS-B      --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.01 --cooldown 24
             run Sens-ADWIN0.05-FS-A      --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.05 --cooldown 24
             run Sens-ADWIN0.05-FS-B      --scenarios S0,S1,S2,S3,S4 --models M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.05 --cooldown 24
             run Sens-P0.5-FS-A           --scenarios S0,S1,S2,S3,S4 --models M2 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 0.5 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-P0.5-FS-B           --scenarios S0,S1,S2,S3,S4 --models M2 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 0.5 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-P2-FS-A             --scenarios S0,S1,S2,S3,S4 --models M2 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 2.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-P2-FS-B             --scenarios S0,S1,S2,S3,S4 --models M2 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 2.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-W1-FS-A             --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 1.0 --adwin-delta 0.002 --cooldown 24
             run Sens-W1-FS-B             --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 1.0 --adwin-delta 0.002 --cooldown 24
             run Sens-W7-FS-A             --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 7.0 --adwin-delta 0.002 --cooldown 24
             run Sens-W7-FS-B             --scenarios S0,S1,S2,S3,S4 --models M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 7.0 --adwin-delta 0.002 --cooldown 24 ;;
faultrate)   # Runs 21-24: fault-rate analysis (optional in Table 4.1)
             run Sens-Fault20-FS-A        --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.2 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-Fault20-FS-B        --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.2 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-Fault5-FS-A         --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.05 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-Fault5-FS-B         --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.05 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 ;;
exploratory) # Runs 25-32: added after the main results were seen - reported as exploratory (Section 9.2.2)
             run Sens-ADWIN0.2-CD6-FS-A   --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.2 --cooldown 6
             run Sens-ADWIN0.2-CD6-FS-B   --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.2 --cooldown 6
             run Sens-ObsInput-CD6-FS-A   --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 6 --adwin-input observation
             run Sens-ObsInput-CD6-FS-B   --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 6 --adwin-input observation
             run Sens-ObsInput-FS-A       --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 --adwin-input observation
             run Sens-ObsInput-FS-B       --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 1.0 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 --adwin-input observation
             run Sens-P0.25-FS-A          --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-A --fault-rate 0.1 --delay-days 0.0 --period-days 0.25 --window-days 3.0 --adwin-delta 0.002 --cooldown 24
             run Sens-P0.25-FS-B          --scenarios S0,S1,S2,S3,S4 --models M1,M2,M3 --feature-set FS-B --fault-rate 0.1 --delay-days 0.0 --period-days 0.25 --window-days 3.0 --adwin-delta 0.002 --cooldown 24 ;;
experiments) for s in main delay sensitivity faultrate exploratory; do "$0" $s; done ;;
analyse)     python -m wsnfd.analyse results/FS-A-Main-Run results/FS-B-Main-Run --what tests --out report_out_tests.csv 2>&1 | tee logs/analyse.log
             python -m wsnfd.analyse results/FS-A-Main-Run results/FS-B-Main-Run --what worthwhile 2>&1 | tee -a logs/analyse.log ;;
report)      python -m wsnfd.report results/*-Main-Run results/Delay-* results/Sens-* --out report_out 2>&1 | tee logs/report.log ;;
all)         for s in check pilot; do "$0" $s; done
             echo "Stop: freeze the pilot settings, then run: experiments analyse report" ;;
*)           sed -n 3,4p "$0"; exit 1 ;;
esac
