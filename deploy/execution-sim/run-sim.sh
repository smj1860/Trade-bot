#!/usr/bin/env bash
# Replays recorded order-book data through the execution simulator on the droplet.
#
#   bash deploy/execution-sim/run-sim.sh BTC/USD 2026/10/08
#   bash deploy/execution-sim/run-sim.sh ETH/USD 2026/10/08 --queue-factor 0.5 --cancel-credit 0.5
#
# Downloads that day's files from Spaces into /root/rec-cache (kept, so a rerun is fast;
# delete with: rm -rf /root/rec-cache), runs at the lowest CPU priority so the recorder is
# never starved, saves the full report to /root/sim-last.txt and shows the first table.
set -euo pipefail

SYMBOL="${1:-BTC/USD}"
DAY="${2:-$(date -u -d yesterday +%Y/%m/%d)}"
shift $(( $# < 2 ? $# : 2 ))

python3 -c "import boto3" 2>/dev/null || apt-get install -y python3-boto3

set -a
. /root/recorder.env
set +a

cd "$(dirname "$0")/../../python-strategy"
nice -n 19 python3 -m execution_sim.run --fetch "$DAY" --cache /root/rec-cache --symbol "$SYMBOL" \
  --fees 0.004:0.008 "$@" > /root/sim-last.txt
head -14 /root/sim-last.txt
echo "(full report: /root/sim-last.txt)"
