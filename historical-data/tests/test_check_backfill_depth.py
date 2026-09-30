import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_backfill_depth import DepthRow, classify, depth_days, expected_candles, format_row

DAY = 86400
NOW = 1_800_000_000


def row(min_ts, max_ts=NOW, count=0):
    return DepthRow("BTC-USD", min_ts, max_ts, min_ts, max_ts, count)


def test_depth_days():
    assert depth_days(NOW - 10 * DAY, NOW) == 10
    assert depth_days(None, NOW) is None


def test_expected_candles_hourly():
    assert expected_candles(NOW - 23 * 3600, NOW, 60) == 24
    assert expected_candles(None, None, 60) == 0


def test_classify():
    assert classify(row(None), NOW, 1080, 60) == "EMPTY"
    assert classify(row(NOW - 1080 * DAY), NOW, 1080, 60) == "OK"
    assert classify(row(NOW - 1079 * DAY), NOW, 1080, 60) == "OK"
    assert classify(row(NOW - 900 * DAY), NOW, 1080, 60) == "SHORT"


def test_format_row_mentions_symbol_and_status():
    out = format_row(row(NOW - 900 * DAY, count=10), NOW, 1080, 60)
    assert "BTC-USD" in out and "SHORT" in out
