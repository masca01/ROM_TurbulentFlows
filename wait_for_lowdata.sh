#!/bin/zsh
# Keeps the Mac awake while run_lowdata.py runs and reports when it ends.
#   zsh wait_for_lowdata.sh
# Leave this window open; launch the run itself in another window or in IDLE.

CONV="/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/convergence"
A="$CONV/lowdata_screen.csv"; A_TOTAL=540        # stage A rows
B="$CONV/lowdata_results.csv"; B_TOTAL=66        # stage B rows
STALE=10800                                      # 3 h without a new row = the run died

rows() { [ -f "$1" ] && awk 'END{print NR-1}' "$1" || echo 0; }
fresh() { [ -f "$1" ] && [ $(( $(date +%s) - $(stat -f %m "$1") )) -lt $STALE ]; }

caffeinate -i -w $$ &
echo "$(date '+%F %H:%M')  waiting: stage A $(rows $A)/$A_TOTAL, stage B $(rows $B)/$B_TOTAL"
while [ "$(rows $B)" -lt $B_TOTAL ]; do
  sleep 300
  echo "$(date '+%F %H:%M')  A $(rows $A)/$A_TOTAL   B $(rows $B)/$B_TOTAL"
  if [ -f "$B" ] && ! fresh "$B"; then
    echo "$(date '+%F %H:%M')  no new row for 3 h, stopped at $(rows $B)/$B_TOTAL. Check the run window."
    echo "run_lowdata.py resumes where it left off: just relaunch it."
    exit 1
  fi
done
echo "$(date '+%F %H:%M')  complete: stage A $(rows $A)/$A_TOTAL, stage B $(rows $B)/$B_TOTAL"
