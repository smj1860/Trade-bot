#!/usr/bin/env bash
# How often did the recorder disconnect or fail a checksum? (reads the cached files)
#   bash deploy/execution-sim/gap-report.sh
set -euo pipefail
cd "$(dirname "$0")/../../python-strategy"
nice -n 19 python3 -m execution_sim.gaps --data /root/rec-cache > /root/gaps-last.txt
cat /root/gaps-last.txt
