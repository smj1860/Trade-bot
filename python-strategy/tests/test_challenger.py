import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import paper_trade
from scripts.challenger import bootstrap_ci, format_report, paired_diffs, plan_retirements


def d(day):
    return datetime(2026, 10, day, tzinfo=timezone.utc)


def test_retire_keeps_newest_and_never_touches_champions():
    arms = [
        ("ext-fib-m4", d(1), False),
        ("ch-1", d(2), False),
        ("ch-2", d(3), False),
        ("ch-3", d(4), False),
        ("ch-0", d(1), True),  # already retired: not listed again
    ]
    assert plan_retirements(arms, keep=2) == ["ch-1"]
    assert plan_retirements(arms, keep=5) == []
    assert plan_retirements(arms, keep=0) == ["ch-3", "ch-2", "ch-1"]


def test_paired_diffs_only_common_trades():
    a = {("BTC-USD", 1): 0.02, ("ETH-USD", 1): -0.01, ("SOL-USD", 2): 0.5}
    b = {("BTC-USD", 1): 0.01, ("ETH-USD", 1): -0.03, ("XRP-USD", 9): 0.9}
    assert sorted(paired_diffs(a, b)) == pytest.approx([0.01, 0.02])


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02] * 10
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi
    assert (m, lo, hi) == bootstrap_ci(vals)
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_report_flags_clear_difference_and_empty_case():
    champ = {("A", i): 0.0 for i in range(40)}
    better = {("A", i): 0.02 for i in range(40)}
    txt = format_report("champ", champ, {"ch-1": better})
    assert "better" in txt and "paired n=40" in txt
    assert "no challengers" in format_report("champ", champ, {})
    assert "no paired closed trades" in format_report("champ", champ, {"ch-2": {("Z", 1): 0.1}})


def test_retired_arm_without_open_trades_does_no_work(monkeypatch, capsys):
    monkeypatch.setattr(paper_trade, "load_meta", lambda conn, arm: {"retired": True, "interval_minutes": 60})
    monkeypatch.setattr(paper_trade, "load_open_trades", lambda conn, arm: [])
    monkeypatch.setattr(paper_trade, "load_model", lambda *a: (_ for _ in ()).throw(AssertionError("loaded model")))
    monkeypatch.setattr(paper_trade, "fetch_candles", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    paper_trade.run_arm(None, "ch-old", {}, 0, True)
    assert "retired" in capsys.readouterr().out


def test_retired_arm_with_open_trade_resolves_but_opens_nothing(monkeypatch, capsys):
    meta = {"retired": True, "interval_minutes": 60, "symbols": ["BTC-USD"]}
    open_t = [{"symbol": "BTC-USD", "entry_ts": 0, "direction": 1, "entry_price": 100.0,
               "barrier_pct": 0.01, "horizon_bars": 4, "round_trip_cost": 0.0}]
    monkeypatch.setattr(paper_trade, "load_meta", lambda conn, arm: meta)
    monkeypatch.setattr(paper_trade, "load_open_trades", lambda conn, arm: open_t)
    monkeypatch.setattr(paper_trade, "load_model", lambda *a: (_ for _ in ()).throw(AssertionError("loaded model")))
    monkeypatch.setattr(paper_trade, "decide_signals", lambda *a: (_ for _ in ()).throw(AssertionError("scored")))
    H = 3600
    rows = [[i * H, 100, 100, 100, 100, 100, 1, 1] for i in range(8)]
    rows[2] = [2 * H, 100, 102, 100, 102, 100, 1, 1]  # long take-profit touched
    monkeypatch.setattr(paper_trade, "fetch_candles", lambda *a, **k: rows)
    paper_trade.run_arm(None, "ch-old", {"BTC-USD": "XBTUSD"}, 8 * H + 60, True, {})
    out = capsys.readouterr().out
    assert "resolved 1 trade" in out and "0 new signal" in out
