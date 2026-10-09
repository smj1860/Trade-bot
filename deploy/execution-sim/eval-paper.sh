#!/usr/bin/env bash
# Replays the paper-trade signals against the recorded order book on the droplet.
#
#   bash deploy/execution-sim/eval-paper.sh                 # every recorded day so far, all arms
#   bash deploy/execution-sim/eval-paper.sh 2026/10/08 2026/10/09
#   ARMS=ext-fib-m4 bash deploy/execution-sim/eval-paper.sh
#
# Needs /root/recorder.env (Spaces) and /root/paper.env (SUPABASE_DB_URL). Files are cached in
# /root/rec-cache. Runs at the lowest CPU priority; the full report goes to /root/eval-last.txt.
set -euo pipefail

FIRST_DAY="2026-10-07"   # the recorder's first day
if [ "$#" -gt 0 ]; then
  DAYS=("$@")
else
  DAYS=(); d="$FIRST_DAY"; today="$(date -u +%F)"
  while [ "$d" != "$(date -u -d "$today + 1 day" +%F)" ]; do DAYS+=("${d//-//}"); d="$(date -u -d "$d + 1 day" +%F)"; done
fi

for m in boto3 psycopg2; do
  python3 -c "import $m" 2>/dev/null || apt-get install -y "python3-$m"
done

set -a
. /root/recorder.env
. /root/paper.env
set +a

cd "$(dirname "$0")/../../python-strategy"
nice -n 19 python3 -m execution_sim.evaluate --fetch "${DAYS[@]}" --cache /root/rec-cache \
  --since "${SINCE:-${FIRST_DAY}T08:00}" ${ARMS:+--arms "$ARMS"} > /root/eval-last.txt
head -40 /root/eval-last.txt
echo "(full report: /root/eval-last.txt)"
