#!/usr/bin/env bash
# Wait for the confirmation sweep (E2) to finish, then run E3 and E4 back to
# back so the GPU is not idle between Sunday noon and Monday.
#
# Order is E3 first, deliberately. E3 with arm D produces a new result --- the
# value of timing as a function of the time budget --- and E4 closes a named
# reviewer objection. If the deadline forces a cut, E4 is the one that goes, so
# it runs second.
#
#   nohup bash scripts/chain_e3_e4.sh > results/runs/chain_e3_e4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

while pgrep -f "scripts/run_program.py" > /dev/null; do sleep 120; done
echo "E2 done at $(date -Is)"

echo "=== E3: time-budget sweep, 4 arms x 5 protocols x T{5,10,50} x 3 seeds ==="
python scripts/run_e3.py --workers 6
e3=$?
echo "E3 exit $e3 at $(date -Is)"

if [ -f scripts/run_e4.py ]; then
  echo "=== E4: gated-neuron threshold sweep ==="
  python scripts/run_e4.py --workers 6
  echo "E4 exit $? at $(date -Is)"
else
  echo "run_e4.py not present; stopping after E3."
fi
