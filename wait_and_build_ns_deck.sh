#!/bin/zsh
# Wait for the Navier-Stokes grid run (run_re100_grid.py in IDLE) to finish, keeping the
# Mac awake, then wake Claude Code in THIS terminal on the same conversation to build the deck.
#   zsh wait_and_build_ns_deck.sh
# Leave the terminal window open. The Claude desktop app can be closed.

CONV="/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/convergence"
CSV="$CONV/re100_tol_modes_grid.csv"
SESSION="b5cfc37b-9fe4-4108-acb0-f09f6ef780da"     # this conversation
LOG="$CONV/ns_deck_build.log"
TOTAL=145

count() { python3 -c "import csv; print(sum(r['generator']=='galerkin_ns' for r in csv.DictReader(open('$CSV'))))"; }
# "alive" = a new result was written in the last 3 hours (the longest training is ~70 min).
# Checking for the IDLE process is not enough: IDLE stays open when the grid script crashes.
alive() { [ $(( $(date +%s) - $(stat -f %m "$CSV") )) -lt 10800 ]; }

caffeinate -i -w $$ &                                # no idle sleep while this script runs
echo "$(date '+%F %H:%M')  waiting: $(count)/$TOTAL NS trainings done"
while [ "$(count)" -lt $TOTAL ] && alive; do
  sleep 300
  echo "$(date '+%F %H:%M')  $(count)/$TOTAL"
done

N=$(count)
if [ "$N" -lt $TOTAL ]; then
  echo "$(date '+%F %H:%M')  no new result for 3 hours, stopped at $N/$TOTAL. Not building the deck."
  echo "Check the IDLE window for an error."
  echo "Relaunch run_re100_grid.py (it resumes), then run this script again."
  exit 1
fi

echo "$(date '+%F %H:%M')  run complete. Waking Claude Code (log: $LOG)"
cd /Users/vmascarilla
claude -p --resume "$SESSION" \
  --allowedTools "Bash,Read,Write,Edit,Skill" \
  "The Navier-Stokes grid run (run_re100_grid.py, generator galerkin_ns) has finished with all $TOTAL trainings in re100_tol_modes_grid.csv. Build the PowerPoint presentation of these results, same structure as '08_Re100 Augmentation Sweep.pptx', and compare them with the data-regression grid. Save it in the Presentations folder and summarise the conclusions." \
  2>&1 | tee "$LOG"
