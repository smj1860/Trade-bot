"""
Unit tests for resample_ohlc.py's pure aggregation logic — no real
database connection. Covers building a coarser-timeframe candle (e.g.
daily from 24 hourly candles) correctly from already-stored hourly data.
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from kraken_client import Candle
from resample_ohlc import resample_candles


def _candle(ts_unix, open_, high, low, close, volume, trade_count=1, vwap=None):
    return Candle(
        ts_unix=ts_unix,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        vwap=Decimal(str(vwap if vwap is not None else close)),
        volume=Decimal(str(volume)),
        trade_count=trade_count,
    )


def test_resample_two_hourly_candles_into_one_2hr_bucket():
    hour0 = _candle(0, open_=100, high=110, low=95, close=105, volume=10, trade_count=3)
    hour1 = _candle(3600, open_=105, high=108, low=101, close=103, volume=20, trade_count=5)

    result = resample_candles([hour0, hour1], source_interval_minutes=60, target_interval_minutes=120)

    assert len(result) == 1
    bucket = result[0]
    assert bucket.ts_unix == 0
    assert bucket.open == Decimal("100")
    assert bucket.high == Decimal("110")
    assert bucket.low == Decimal("95")
    assert bucket.close == Decimal("103")
    assert bucket.volume == Decimal("30")
    assert bucket.trade_count == 8


def test_resample_24_hourly_candles_into_one_daily_bucket():
    hours = [_candle(i * 3600, open_=100 + i, high=100 + i + 1, low=100 + i - 1, close=100 + i, volume=1) for i in range(24)]

    result = resample_candles(hours, source_interval_minutes=60, target_interval_minutes=1440)

    assert len(result) == 1
    bucket = result[0]
    assert bucket.ts_unix == 0
    assert bucket.open == hours[0].open
    assert bucket.close == hours[-1].close
    assert bucket.high == max(h.high for h in hours)
    assert bucket.low == min(h.low for h in hours)
    assert bucket.volume == Decimal("24")
    assert bucket.trade_count == 24


def test_resample_splits_candles_spanning_bucket_boundary_into_separate_buckets():
    # Two full 2hr buckets worth of hourly candles (4 total) should produce 2 daily-2hr buckets, not 1.
    hours = [_candle(i * 3600, open_=100, high=101, low=99, close=100, volume=1) for i in range(4)]

    result = resample_candles(hours, source_interval_minutes=60, target_interval_minutes=120)

    assert [c.ts_unix for c in result] == [0, 7200]


def test_resample_vwap_is_volume_weighted():
    # 10 volume @ vwap 100, 30 volume @ vwap 200 -> weighted average = (10*100 + 30*200)/40 = 175
    hour0 = _candle(0, open_=100, high=100, low=100, close=100, volume=10, vwap=100)
    hour1 = _candle(3600, open_=100, high=100, low=100, close=100, volume=30, vwap=200)

    result = resample_candles([hour0, hour1], source_interval_minutes=60, target_interval_minutes=120)

    assert result[0].vwap == Decimal("175")


def test_resample_vwap_falls_back_to_close_when_bucket_has_zero_volume():
    hour0 = _candle(0, open_=100, high=100, low=100, close=104, volume=0, vwap=0)

    result = resample_candles([hour0], source_interval_minutes=60, target_interval_minutes=120)

    assert result[0].vwap == Decimal("104")


def test_resample_rejects_target_not_a_multiple_of_source():
    with pytest.raises(ValueError):
        resample_candles([], source_interval_minutes=60, target_interval_minutes=90)


def test_resample_rejects_target_not_greater_than_source():
    with pytest.raises(ValueError):
        resample_candles([], source_interval_minutes=60, target_interval_minutes=60)
