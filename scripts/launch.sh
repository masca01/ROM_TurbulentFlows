#!/usr/bin/env bash
# Launch any study from a Terminal window: thread cap, no idle sleep (macOS), a log file,
# optionally queued behind another run, optionally followed by more scripts.
#
#   scripts/launch.sh [options] <script.py> [args ...] [--then <script.py> [args ...]] ...
#
# Options (before the script):
#   --threads N      cap BLAS / OpenMP / torch threads (default 6, the old capacity launcher:
#                    it leaves the rest of a 16-core Mac to a run already going in IDLE)
#   --after X        wait until process X has finished first. X is a PID, or a pattern matched
#                    against running command lines (pgrep -f), e.g. --after run_lowdata_subsets.
#                    A script started from IDLE does not show its name in any command line:
#                    give the PID instead, or start that run with this launcher too.
#   --log FILE       log file (default: <RESULTS>/logs/<script>_<date>.log; RESULTS is
#                    rom.paths.RESULTS, i.e. ../../convergence unless ROM_RESULTS_DIR is set)
#   --no-caffeinate  do not keep the Mac awake
#
# Every script runs from the repository root with `python3 -u`; its output goes to the terminal
# and to the log. A --then script runs only if everything before it exited with status 0, and
# the launcher exits with the status of the first script that failed.
#
# Examples
#   scripts/launch.sh studies/4_region_map/run_region_map.py
#   scripts/launch.sh --after run_lowdata_subsets studies/6_capacity/run_capacity.py \
#                     --then studies/6_capacity/capacity_headroom.py
#   scripts/launch.sh --threads 12 studies/5_low_data/run_lowdata.py --stage B
#
# (Replaces run_capacity_terminal.sh. That zsh script assigned `status=${pipestatus[1]}`;
#  `status` is a read-only variable in zsh, so the script died there and capacity_headroom.py
#  never ran. Here the exit code goes to `rc`, taken from bash's PIPESTATUS.)

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
THREADS=6
AFTER=""
LOG=""
CAFFEINATE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --threads)       THREADS="$2"; shift 2 ;;
    --after)         AFTER="$2"; shift 2 ;;
    --log)           LOG="$2"; shift 2 ;;
    --no-caffeinate) CAFFEINATE=0; shift ;;
    -h|--help)       sed -n '2,34p' "$0"; exit 0 ;;
    --)              shift; break ;;
    -*)              echo "launch.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
    *)               break ;;
  esac
done
[ $# -gt 0 ] || { echo "launch.sh: no script given (see --help)" >&2; exit 2; }

# split the rest into jobs separated by --then
JOBS=()
job=""
for a in "$@"; do
  if [ "$a" = "--then" ]; then
    [ -n "$job" ] && JOBS+=("$job"); job=""
  else
    job+="$(printf '%q ' "$a")"
  fi
done
[ -n "$job" ] && JOBS+=("$job")

cd "$REPO" || exit 1
if [ -z "$LOG" ]; then
  RESULTS="$(cd / && "$PYTHON" -c 'from rom import paths; print(paths.RESULTS)' 2>/dev/null)" || {
    echo "launch.sh: cannot import rom — run 'python3 -m pip install -e .' in $REPO first" >&2; exit 1; }
  first="$(eval "set -- ${JOBS[0]}"; basename "$1" .py)"
  LOG="$RESULTS/logs/${first}_$(date '+%Y%m%d_%H%M').log"
fi
mkdir -p "$(dirname "$LOG")"

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       VECLIB_MAXIMUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

say() { echo "$(date '+%F %H:%M')  $*" | tee -a "$LOG"; }

if [ "$CAFFEINATE" = 1 ] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &                         # no idle sleep while this launcher lives
fi

if [ -n "$AFTER" ]; then
  # this launcher and the shells that started it have the pattern in their command line too
  SELF=" $$ "
  p=$$
  while [ "${p:-1}" -gt 1 ]; do
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    SELF+="$p "
  done
  running() {
    if [[ "$AFTER" =~ ^[0-9]+$ ]]; then kill -0 "$AFTER" 2>/dev/null; return; fi
    local q
    for q in $(pgrep -f -- "$AFTER"); do
      [[ "$SELF" == *" $q "* ]] || return 0
    done
    return 1
  }
  if running; then
    say "waiting for '$AFTER' to finish"
    while running; do sleep 60; done
    say "'$AFTER' has finished"
  else
    say "'$AFTER' is not running; starting now"
  fi
fi

for job in "${JOBS[@]}"; do
  eval "set -- $job"
  say "starting: $* ($THREADS threads, log $LOG)"
  "$PYTHON" -u "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    say "$1 exited with status $rc — stopping here (see $LOG)"
    exit "$rc"
  fi
  say "finished: $1"
done
say "all done"
