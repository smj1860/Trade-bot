"""
Unit tests for plan_backfill_deepening.py's pure planning logic — no real
database connection. Covers the automated deepening workflow's core
decision: what window (if any) does a symbol need next, given its
current depth and the target.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_backfill_deepening import (
    PlannedWindow,
    build_plan,
    current_depth_days,
    next_window_for,
)
from symbols import SymbolSpec


def test_next_window_for_advances_by_one_90_day_slice():
    window = next_window_for("BTC-USD", current_depth_days=270.0, target_days=1080.0)
    assert window == PlannedWindow(symbol="BTC-USD", since_days=360.0, before_days=270.0)


def test_next_window_for_caps_at_target_when_the_last_slice_would_overshoot():
    # 1020 days deep, target 1080 -> only 60 more days needed, not a full 90.
    window = next_window_for("BTC-USD", current_depth_days=1020.0, target_days=1080.0)
    assert window == PlannedWindow(symbol="BTC-USD", since_days=1080.0, before_days=1020.0)


def test_next_window_for_returns_none_once_target_is_reached():
    assert next_window_for("BTC-USD", current_depth_days=1080.0, target_days=1080.0) is None
    assert next_window_for("BTC-USD", current_depth_days=1200.0, target_days=1080.0) is None


def test_next_window_for_starts_from_zero_for_a_brand_new_symbol():
    window = next_window_for("NEW-USD", current_depth_days=0.0, target_days=1080.0)
    assert window == PlannedWindow(symbol="NEW-USD", since_days=90.0, before_days=0.0)


def test_planned_window_dispatch_inputs_format_whole_days_without_decimals():
    window = PlannedWindow(symbol="BTC-USD", since_days=360.0, before_days=270.0)
    assert window.as_dispatch_inputs() == {"symbol": "BTC-USD", "since_days": "360", "before_days": "270"}


def test_current_depth_days_is_zero_with_no_backfill_state_row_yet():
    assert current_depth_days(state=None, now=1_000_000.0) == 0.0


def test_current_depth_days_computed_from_earliest_ts():
    now = 1_000_000.0
    ninety_days_ago = now - 90 * 86400
    # state is (earliest_unix, latest_unix); latest_unix is irrelevant here.
    assert current_depth_days(state=(ninety_days_ago, now), now=now) == 90.0


def test_build_plan_skips_symbols_already_at_target_and_plans_the_rest():
    specs = [
        SymbolSpec(symbol="BTC-USD", rest_native_symbol="XBTUSD"),
        SymbolSpec(symbol="PENDLE-USD", rest_native_symbol="PENDLEUSD"),
    ]
    depths = {"BTC-USD": 1080.0, "PENDLE-USD": 360.0}
    plan = build_plan(specs, depths, target_days=1080.0)
    assert plan == [PlannedWindow(symbol="PENDLE-USD", since_days=450.0, before_days=360.0)]


def test_build_plan_defaults_missing_depth_to_zero():
    specs = [SymbolSpec(symbol="NEW-USD", rest_native_symbol="NEWUSD")]
    plan = build_plan(specs, depths={}, target_days=1080.0)
    assert plan == [PlannedWindow(symbol="NEW-USD", since_days=90.0, before_days=0.0)]
