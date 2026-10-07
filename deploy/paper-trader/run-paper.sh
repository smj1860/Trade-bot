#!/bin/bash
# Runs the paper trader (or its report) once, in a throwaway container, at low
# priority and with a memory cap so it can never starve the order-book
# recorder on the same 1 vCPU / 1 GB droplet. Called by cron (see README.md).
#   run-paper.sh            hourly: resolve open trades, open new ones
#   run-paper.sh dry        score and print only; writes nothing
#   run-paper.sh report     print the per-arm performance report
set -u
MODE="${1:-trade}"
case "$MODE" in
  trade)  CMD="python scripts/paper_trade.py --arm all" ;;
  dry)    CMD="python scripts/paper_trade.py --arm all --dry-run" ;;
  report) CMD="python scripts/paper_report.py --arm all" ;;
  *) echo "usage: $0 [trade|dry|report]"; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) paper-trade $MODE ==="
# flock: skip (don't pile up) if the previous run is somehow still going
exec flock -n /var/lock/paper-trade.lock \
  timeout 600 nice -n 10 \
  docker run --rm --memory 450m --memory-swap 900m --cpus 0.5 \
    --env-file /root/paper.env kraken-paper $CMD
