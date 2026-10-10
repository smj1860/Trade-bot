import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution_sim.engine import NS, Fees, SimConfig  # noqa: E402
from execution_sim.evaluate import build_specs, format_report, load_config  # noqa: E402
from execution_sim.lifecycle import SCENARIOS, TradeSpec, returns, run_lifecycles  # noqa: E402
from execution_sim.replay import MultiReplayer  # noqa: E402
from test_execution_sim import SNAP, SYM, book_msg, trade_msg, write_rec  # noqa: E402

CFG = {SYM: SimConfig(tick=0.1, price_decimals=1)}
FEES = Fees(0.004, 0.008)
SCEN = {s.name: s for s in SCENARIOS}


def filler(start, end):
    """A far-away level flickering once a second: gives the engine its clock."""
    return [(t, book_msg([(90.0, 1 + t % 2)], [])) for t in range(start, end)]


def recording(tmp_path, extra, end=1200):
    items = [(0, {"_event": "connect", "depth": 25}), (1, SNAP)] + filler(2, end) + extra
    write_rec(tmp_path / "2026/10/07/00-run.tsv.gz", sorted(items, key=lambda x: x[0]))
    return MultiReplayer([tmp_path], [SYM])


def spec(direction=1, barrier=0.01, hold_s=1000, t0=10):
    return TradeSpec(0, SYM, direction, t0 * NS, barrier, hold_s * NS)


def run(tmp_path, extra, scen, variants=((1.0, 0.0),), **kw):
    rep = recording(tmp_path, extra)
    outs, stats = run_lifecycles([kw.pop("sp", spec())], rep, CFG, list(variants), [SCEN[s] for s in scen], 1000.0)
    return outs, stats


# bids rise to 101.5 with plenty of size and the old asks are replaced
UP = [(90, book_msg([(101.5, 50.0)], [(100.2, 0), (100.3, 0), (100.4, 0), (101.6, 50.0)]))]
DOWN = [(90, book_msg([(100.0, 0), (99.9, 0), (99.8, 0), (98.5, 50.0)], [(98.6, 50.0)]))]


def test_all_taker_target_exit(tmp_path):
    outs, stats = run(tmp_path, UP + [(100, trade_msg("buy", 101.2, 0.1))], ["all-taker"])
    assert stats.started == 1 and len(outs) == 1
    o = outs[0]
    assert o.reason == "target" and o.long
    assert [f.maker for f in o.entry_fills + o.exit_fills] == [False, False]
    gross, fee, net = returns(o, FEES)
    entry_avg = (5 * 100.2 + (1000 / 100.1 - 5) * 100.3) / (1000 / 100.1)  # walks two ask levels
    assert gross == pytest.approx((101.5 - entry_avg) / 100.1, rel=1e-3)
    assert fee == pytest.approx(0.008 * 2, rel=0.02)
    assert net == pytest.approx(gross - fee)


def test_all_taker_stop_exit_for_a_short(tmp_path):
    ups = [(100, trade_msg("buy", 101.3, 0.1))]
    outs, _ = run(tmp_path, UP + ups, ["all-taker"], sp=spec(direction=-1))
    o = outs[0]
    assert o.reason == "stop" and not o.long
    assert returns(o, FEES)[0] < -0.01  # shorted at ~100.0, bought back at ~101.6


def test_timeout_exit_when_nothing_happens(tmp_path):
    outs, _ = run(tmp_path, [], ["all-taker"])
    o = outs[0]
    assert o.reason == "timeout"
    assert o.exit_fills[0].t_ns >= (10 + 1000) * NS


def test_maker_target_rests_and_fills_as_maker(tmp_path):
    # the target price (101.1) is not displayed, so a trade through it fills the resting order
    outs, _ = run(tmp_path, [(100, trade_msg("buy", 101.2, 0.1))], ["exit-passive"])
    o = outs[0]
    assert o.reason == "target"
    assert o.exit_fills[0].maker and o.exit_fills[0].price == pytest.approx(100.1 * 1.01)
    assert not o.entry_fills[0].maker  # exit-passive enters at market
    gross, fee, _ = returns(o, FEES)
    assert gross > 0.008 and fee == pytest.approx(0.008 + 0.004, rel=0.02)  # taker in, maker out


def test_stop_cancels_the_resting_target(tmp_path):
    outs, _ = run(tmp_path, DOWN + [(100, trade_msg("sell", 99.0, 0.1))], ["exit-passive"])
    o = outs[0]
    assert o.reason == "stop" and not o.exit_fills[0].maker


def test_chase_window_then_cross_at_timeout(tmp_path):
    outs, _ = run(tmp_path, [], ["exit-passive"])
    o = outs[0]
    assert o.reason == "timeout"
    assert o.exit_fills[-1].t_ns < (10 + 1000) * NS  # crossed inside the window, before the timeout
    assert not o.exit_fills[-1].maker


def test_chase_window_can_fill_as_maker(tmp_path):
    # window opens at 410 s; a buyer lifts our resting ask at 100.2 after it is posted
    outs, _ = run(tmp_path, [(420, trade_msg("buy", 100.2, 500.0))], ["exit-passive"])
    o = outs[0]
    assert o.reason == "timeout" and o.exit_fills[0].maker


def test_passive_entry_fills_at_touch(tmp_path):
    outs, _ = run(tmp_path, [(15, trade_msg("sell", 99.9, 500.0))], ["entry-passive"])
    o = outs[0]
    assert o.entry_fills[0].maker and o.entry_fills[0].price == 100.0


def test_gap_excludes_the_trade(tmp_path):
    outs, stats = run(tmp_path, [(300, {"_event": "disconnect", "reason": "x"})], ["all-taker"])
    assert outs == [] and stats.aborted == 1


def test_trade_without_recording_at_entry_is_skipped(tmp_path):
    rep = recording(tmp_path, [])
    outs, stats = run_lifecycles([spec(t0=-100)], rep, CFG, [(1.0, 0.0)], [SCEN["all-taker"]], 1000.0)
    assert outs == [] and stats.no_data_at_entry == 1


def test_taker_scenarios_ignore_variants_and_passive_ones_get_each(tmp_path):
    outs, _ = run(tmp_path, UP + [(100, trade_msg("buy", 101.2, 0.1))], ["all-taker", "all-passive"], variants=((1.0, 0.0), (0.5, 0.5)))
    keys = sorted((o.scenario, o.variant) for o in outs)
    assert keys == [("all-passive", 0), ("all-passive", 1), ("all-taker", 0)]


def test_build_specs_dedupes_and_report_runs(tmp_path):
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text('[[symbols]]\nsymbol="BTC-USD"\nexchange_native_symbol="BTC/USD"\ntick_size="0.1"\n')
    cfg = load_config(cfg_path)
    base = {"symbol": "BTC-USD", "entry_ts": 0, "direction": 1, "barrier_pct": 0.01, "horizon_bars": 1, "interval_minutes": 1,
            "status": "closed", "exit_reason": "timeout", "net_return": -0.017, "round_trip_cost": 0.017}
    rows = [dict(base, arm="a"), dict(base, arm="b"), dict(base, arm="a", entry_ts=60, direction=-1)]
    specs, owners, notes = build_specs(rows, cfg)
    assert len(specs) == 2 and len(owners[specs[0].uid]) == 2 and not notes
    assert specs[0].t0_ns == 60 * NS and specs[0].hold_ns == 60 * NS and specs[0].symbol == SYM
    outs, stats = run(tmp_path, UP + [(100, trade_msg("buy", 101.2, 0.1))], [s.name for s in SCENARIOS],
                      sp=TradeSpec(specs[0].uid, SYM, 1, 10 * NS, 0.01, 1000 * NS))
    text = format_report(outs, owners, [("fees t1", FEES)], [(1.0, 0.0)], stats)
    assert "all-taker" in text and "all-passive" in text and "validation" in text and "per arm" in text


def test_gap_summary(tmp_path):
    from execution_sim.gaps import format_summary, scan
    rep = recording(tmp_path, [(300, {"_event": "disconnect", "reason": "boom"}), (310, {"_event": "connect", "depth": 25}),
                               (320, {"_event": "checksum_mismatch", "symbol": SYM})])
    s = scan([tmp_path])
    assert s["events"]["disconnect"] == 1 and s["mismatch"][SYM] == 1 and s["downtimes"] == [10.0]
    assert "boom" in format_summary(s)
