"""
Unit tests for ohlc_from_trades.py's trade-to-candle aggregation — pure
logic, no real Kraken calls or database. Trade fixtures are built by hand
so each expected OHLCVT value can be hand-verified against them.
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kraken_client import Trade
from ohlc_from_trades import aggregate_trades_to_candles, bucket_start


def _trade(trade_id: int, ts_unix: float, price: str, volume: str) -> Trade:
    return Trade(
        trade_id=trade_id,
        ts_unix_ns=int(ts_unix * 1_000_000_000),
        price=Decimal(price),
        volume=Decimal(volume),
        side="buy",
        order_type="market",
    )


def test_bucket_start_floors_to_interval_boundary():
    # 3723s = 1h2m3s -> the 60-minute bucket starting at 3600 (the top of that hour)
    assert bucket_start(3723, 60) == 3600
    assert bucket_start(3600, 60) == 3600  # exactly on a boundary stays in that bucket
    assert bucket_start(0, 60) == 0


def test_bucket_start_aligned_to_unix_epoch_not_first_trade():
    # Regardless of where a stream of trades starts, buckets align to
    # wall-clock interval boundaries from the epoch.
    assert bucket_start(125, 60) == 0
    assert bucket_start(3599, 60) == 0
    assert bucket_start(3600, 60) == 3600


def test_single_bucket_aggregates_ohlcvt_correctly():
    trades = [
        _trade(1, 100.0, "10.0", "1.0"),   # open
        _trade(2, 110.0, "12.0", "2.0"),   # new high
        _trade(3, 120.0, "9.0", "1.0"),    # new low
        _trade(4, 130.0, "11.0", "1.0"),   # close
    ]
    candles = list(aggregate_trades_to_candles(trades, interval_minutes=60))
    assert len(candles) == 1
    c = candles[0]
    assert c.ts_unix == 0
    assert c.open == Decimal("10.0")
    assert c.high == Decimal("12.0")
    assert c.low == Decimal("9.0")
    assert c.close == Decimal("11.0")
    assert c.volume == Decimal("5.0")  # 1+2+1+1
    assert c.trade_count == 4
    # vwap = sum(price*volume)/sum(volume)
    expected_vwap = (Decimal("10.0") * 1 + Decimal("12.0") * 2 + Decimal("9.0") * 1 + Decimal("11.0") * 1) / Decimal("5.0")
    assert c.vwap == expected_vwap


def test_trades_spanning_multiple_buckets_yield_one_candle_each():
    trades = [
        _trade(1, 100.0, "10.0", "1.0"),    # bucket 0 (0-3600)
        _trade(2, 200.0, "11.0", "1.0"),    # bucket 0
        _trade(3, 3700.0, "20.0", "1.0"),   # bucket 3600 (3600-7200)
        _trade(4, 8000.0, "30.0", "1.0"),   # bucket 7200
    ]
    candles = list(aggregate_trades_to_candles(trades, interval_minutes=60))
    assert [c.ts_unix for c in candles] == [0, 3600, 7200]
    assert candles[0].open == Decimal("10.0") and candles[0].close == Decimal("11.0")
    assert candles[1].open == Decimal("20.0") and candles[1].close == Decimal("20.0")
    assert candles[2].open == Decimal("30.0") and candles[2].close == Decimal("30.0")


def test_final_open_bucket_is_still_yielded():
    # Even a single trailing trade with no follow-up trade in its bucket
    # should still produce a (potentially partial) candle -- the caller
    # decides whether to trust the very last one as complete.
    trades = [_trade(1, 100.0, "10.0", "1.0")]
    candles = list(aggregate_trades_to_candles(trades, interval_minutes=60))
    assert len(candles) == 1
    assert candles[0].trade_count == 1


def test_empty_trade_stream_yields_no_candles():
    assert list(aggregate_trades_to_candles([], interval_minutes=60)) == []


def test_vwap_falls_back_to_close_when_volume_is_zero():
    # Degenerate case (shouldn't happen with real trade data, but the
    # division-by-zero guard should hold regardless).
    trades = [_trade(1, 100.0, "10.0", "0")]
    candles = list(aggregate_trades_to_candles(trades, interval_minutes=60))
    assert candles[0].volume == Decimal("0")
    assert candles[0].vwap == Decimal("10.0")


def test_one_minute_interval_buckets_finer_than_hourly():
    trades = [
        _trade(1, 30.0, "10.0", "1.0"),   # minute bucket 0 (0-60)
        _trade(2, 90.0, "11.0", "1.0"),   # minute bucket 60 (60-120)
        _trade(3, 91.0, "12.0", "1.0"),   # same minute bucket as above
    ]
    candles = list(aggregate_trades_to_candles(trades, interval_minutes=1))
    assert [c.ts_unix for c in candles] == [0, 60]
    assert candles[1].open == Decimal("11.0")
    assert candles[1].close == Decimal("12.0")
    assert candles[1].trade_count == 2
