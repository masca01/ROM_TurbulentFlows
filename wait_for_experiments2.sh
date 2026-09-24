#!/bin/zsh
# Keeps the Mac awake while run_experiments2.py runs and reports when it ends.
#   zsh wait_for_experiments2.sh
# Leave this terminal window open; launch the run itself in another window or in IDLE.

CONV="/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/convergence"
CSV="$CONV/experiments2_results.csv"
TOTAL=52                       # 17 real-only + 35 augmented trainings
STALE=10800                    # 3 h without a new row = the run died

count() { [ -f "$CSV" ] && python3 -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))" "$CSV" || echo 0; }
alive() { [ ! -f "$CSV" ] || [ $(( $(date +%s) - $(stat -f %m "$CSV") )) -lt $STALE ]; }

caffeinate -i -w $$ &                      # no idle sleep while this script runs
echo "$(date '+%F %H:%M')  waiting: $(count)/$TOTAL trainings written"
while [ "$(count)" -lt $TOTAL ] && alive; do
  sleep 300
  echo "$(date '+%F %H:%M')  $(count)/$TOTAL"
done

N=$(count)
if [ "$N" -lt $TOTAL ]; then
  echo "$(date '+%F %H:%M')  no new row for 3 h, stopped at $N/$TOTAL. Check the run window for an error."
  echo "run_experiments2.py resumes where it left off: just relaunch it."
  exit 1
fi
echo "$(date '+%F %H:%M')  run complete: $N/$TOTAL rows in $CSV"
