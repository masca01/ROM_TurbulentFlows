#!/bin/zsh
# Measure the capacity ceilings (run_capacity.py) from a Terminal window, either alongside a run
# already going in IDLE or queued behind it.
#
#   zsh run_capacity_terminal.sh            # start now, in parallel with whatever else is running
#   zsh run_capacity_terminal.sh --after    # wait for the 25-snapshot subsets run to finish first
#
# Training here is CPU-only (no MPS/CUDA in the recipe) and torch defaults to 12 threads on a
# 16-core machine, so a second full-thread process would oversubscribe the cores and slow both
# jobs. This script caps itself to 6 threads, which leaves the IDLE run its share.

CODE="/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/CODE"
CONV="/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/convergence"
LOG="$CONV/capacity_run.log"
SUBSETS_CSV="$CONV/lowdata_results.csv"
SUBSETS_TARGET=131                       # rows when the 8-subset run is complete

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 OPENBLAS_NUM_THREADS=6

rows() { [ -f "$SUBSETS_CSV" ] && awk 'END{print NR-1}' "$SUBSETS_CSV" || echo 0; }

caffeinate -i -w $$ &                    # no idle sleep while this window is open

if [ "$1" = "--after" ]; then
  echo "$(date '+%F %H:%M')  waiting for the subsets run: $(rows)/$SUBSETS_TARGET rows" | tee -a "$LOG"
  while [ "$(rows)" -lt $SUBSETS_TARGET ]; do
    sleep 120
    if [ $(( $(date +%s) - $(stat -f %m "$SUBSETS_CSV") )) -gt 5400 ]; then
      echo "$(date '+%F %H:%M')  subsets run has been quiet for 90 min ($(rows)/$SUBSETS_TARGET); starting anyway" | tee -a "$LOG"
      break
    fi
  done
fi

echo "$(date '+%F %H:%M')  starting run_capacity.py with $OMP_NUM_THREADS threads" | tee -a "$LOG"
cd "$CODE" || exit 1
python3 -u run_capacity.py 2>&1 | tee -a "$LOG"
status=${pipestatus[1]}

if [ $status -eq 0 ]; then
  echo "$(date '+%F %H:%M')  capacity run finished; refreshing every corrected headroom" | tee -a "$LOG"
  python3 -u capacity_headroom.py 2>&1 | tee -a "$LOG"
  echo "$(date '+%F %H:%M')  done. Results: $CONV/capacity_ceilings.csv and headroom_corrected*.csv" | tee -a "$LOG"
else
  echo "$(date '+%F %H:%M')  run_capacity.py exited with status $status - see $LOG" | tee -a "$LOG"
  exit $status
fi
