#!/usr/bin/env python3
"""
Computes the next backfill-deepening window for each configured symbol,
based on ohlc_backfill_state, for the scheduled
`.github/workflows/historical-backfill-deepening.yml` workflow.

This is the automated, self-continuing counterpart to
`backfill_ohlc_from_trades.py`'s manual, one-symbol-at-a-time
`--since-days`/`--before-days` dispatch (see that script's module
docstring, and `historical-backfill-trades.yml`'s header, for why this
pulls from Kraken's Trades endpoint and why majors are slow): each run
looks at how deep every symbol's history already goes and, for any
symbol still short of --target-days, plans exactly one more 90-day-older
window — the same shape of window every prior deepening round so far has
been dispatched by hand for. Printed as a JSON list to stdout (an empty
list `[]` once every symbol has reached the target) so the workflow can
feed it straight into a GitHub Actions matrix job; a symbol that already
reached --target-days is simply left out of the list, which is how this
stops dispatching more work on its own instead of needing to be told to
stop.

A symbol with no ohlc_backfill_state row yet (current_depth_days == 0)
still gets planned starting from day 0 — this only actually matters if a
new symbol is added to config.example.toml before ever being backfilled;
every symbol currently configured already has at least 6 months of
history as of 2026-09-27/28.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 plan_backfill_deepening.py --target-days 1080
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import db
from symbols import SymbolSpec, load_symbols

EXCHANGE = "kraken"
WINDOW_DAYS = 90.0


@dataclass(frozen=True)
class PlannedWindow:
    symbol: str
    since_days: float
    before_days: float

    def as_dispatch_inputs(self) -> dict:
        # GitHub Actions matrix/workflow inputs are strings; keep whole-day
        # values as plain integers (e.g. "360" not "360.0") to match how
        # every manual dispatch this session has passed them.
        def fmt(days: float) -> str:
            return str(int(days)) if days == int(days) else str(days)

        return {"symbol": self.symbol, "since_days": fmt(self.since_days), "before_days": fmt(self.before_days)}


def next_window_for(symbol: str, current_depth_days: float, target_days: float) -> PlannedWindow | None:
    """Pure planning logic, kept separate from the DB/CLI glue below so it
    can be unit-tested without a real Postgres connection."""
    if current_depth_days >= target_days:
        return None
    before_days = round(current_depth_days)
    since_days = min(before_days + WINDOW_DAYS, target_days)
    if since_days <= before_days:
        return None
    return PlannedWindow(symbol=symbol, since_days=since_days, before_days=before_days)


def current_depth_days(state: tuple | None, now: float) -> float:
    if state is None or state[0] is None:
        return 0.0
    earliest_unix = state[0]
    return max(0.0, (now - earliest_unix) / 86400.0)


def build_plan(specs: list[SymbolSpec], depths: dict[str, float], target_days: float) -> list[PlannedWindow]:
    plan = []
    for spec in specs:
        window = next_window_for(spec.symbol, depths.get(spec.symbol, 0.0), target_days)
        if window is not None:
            plan.append(window)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=60, help="Candle resolution in minutes (default 60).")
    parser.add_argument(
        "--target-days",
        type=float,
        default=1080.0,
        help="Target depth in days for every symbol (default 1080 = 36 months).",
    )
    args = parser.parse_args()

    now = time.time()
    specs = load_symbols()
    conn = db.connect()
    try:
        depths = {
            spec.symbol: current_depth_days(db.get_backfill_state(conn, EXCHANGE, spec.symbol, args.interval), now)
            for spec in specs
        }
    finally:
        conn.close()

    plan = build_plan(specs, depths, args.target_days)
    print(json.dumps([w.as_dispatch_inputs() for w in plan]))


if __name__ == "__main__":
    main()
