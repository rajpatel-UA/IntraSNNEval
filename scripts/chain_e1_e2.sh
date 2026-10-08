#!/usr/bin/env bash
# Wait for E1 to finish, then run the full confirmation sweep in the order
# registered in the addendum: cheapest protocol first, so the ranking becomes
# computable on four protocols long before CIC-IDS2017 finishes.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_e1.py" > /dev/null; do sleep 60; done
echo "E1 done at $(date -Is); starting E2" 
exec python scripts/run_program.py --phases E2a E2b E2c E2d E2e
