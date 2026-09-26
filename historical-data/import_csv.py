#!/usr/bin/env python3
"""
Bulk-imports OHLC candles from an already-downloaded CSV file into the
same ohlc_candles table backfill_ohlc.py writes to — for loading a large
historical dataset all at once (e.g. Kraken's own downloadable per-pair
OHLCVT dumps, which go back much further than the public REST API's
retention window lets backfill_ohlc.py reach — see the "retention"
section of docs/historical-data-pipeline.md), rather than waiting for
history to accumulate one REST call at a time.

This is meant to run ALONGSIDE backfill_ohlc.py / the scheduled GitHub
Actions job, not instead of it: a CSV import gets deep history in one
shot; the recurring job keeps things current afterward. Both write to the
same table with the same upsert-by-(exchange, symbol, interval, ts) key,
so importing a CSV that overlaps what's already there is safe — it just
updates those rows in place.

Two input shapes are supported:

  --format kraken-dump
      Kraken's own downloadable per-pair CSV dumps: no header row, columns
      in order (unix_timestamp, open, high, low, close, volume, trades).
      One file per (pair, interval) pair — the file itself doesn't say
      what resolution it is, so you tell this script with --interval.

  --format generic
      Any CSV with a header row. Column names are matched case-
      insensitively against common variants:
        timestamp: "timestamp" | "time" | "date" | "datetime"
        open/high/low/close: "open" | "high" | "low" | "close"
        volume: "volume" | "vol"
        vwap (optional): "vwap"
        trade count (optional): "trades" | "trade_count" | "count"
      The timestamp column can be either unix seconds (a plain number) or
      a date/time string pandas can parse (e.g. "2024-01-15 00:00:00").

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 import_csv.py --symbol BTC-USD --interval 1 --format kraken-dump --file XBTUSD_1.csv
    python3 import_csv.py --symbol BTC-USD --interval 60 --format generic --file my_export.csv
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal

import pandas as pd

import db
from kraken_client import Candle
from symbols import load_symbols

EXCHANGE = "kraken"
CHUNK_ROWS = 20_000  # stream large files in chunks rather than loading the whole thing into memory

KRAKEN_DUMP_COLUMNS = ["ts_unix", "open", "high", "low", "close", "volume", "trade_count"]

GENERIC_COLUMN_ALIASES = {
    "ts": ["timestamp", "time", "date", "datetime"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
    "volume": ["volume", "vol"],
    "vwap": ["vwap"],
    "trade_count": ["trades", "trade_count", "count"],
}


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    # Numeric-looking column -> already unix seconds. Otherwise, let
    # pandas parse it as a date/time string and convert to unix seconds.
    #
    # NOTE: pandas' internal datetime64 resolution isn't guaranteed to be
    # nanoseconds (it varies by pandas version and by what to_datetime
    # infers — this project has seen microsecond resolution in practice).
    # `.astype("int64") // 1_000_000_000` silently assumes nanoseconds and
    # is wrong-by-1000x on a microsecond-resolution series. Going through
    # numpy's `datetime64[s]` cast instead lets numpy do the unit
    # conversion explicitly, so it's correct regardless of the source
    # resolution.
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64")
    parsed = pd.to_datetime(series, utc=True)
    return pd.Series(parsed.values.astype("datetime64[s]").astype("int64"), index=series.index)


def _row_to_candle(ts_unix, open_, high, low, close, volume, vwap, trade_count) -> Candle:
    return Candle(
        ts_unix=int(ts_unix),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        vwap=Decimal(str(vwap)) if vwap is not None and not pd.isna(vwap) else Decimal(str(close)),
        volume=Decimal(str(volume)),
        trade_count=int(trade_count) if trade_count is not None and not pd.isna(trade_count) else 0,
    )


def import_kraken_dump(path: str, symbol: str, interval: int, conn) -> int:
    total = 0
    earliest = None
    latest = None
    for chunk in pd.read_csv(path, header=None, names=KRAKEN_DUMP_COLUMNS, chunksize=CHUNK_ROWS):
        candles = [
            _row_to_candle(r.ts_unix, r.open, r.high, r.low, r.close, r.volume, None, r.trade_count)
            for r in chunk.itertuples(index=False)
        ]
        total += db.upsert_candles(conn, EXCHANGE, symbol, interval, candles)
        for c in candles:
            earliest = c.ts_unix if earliest is None else min(earliest, c.ts_unix)
            latest = c.ts_unix if latest is None else max(latest, c.ts_unix)
    if earliest is not None:
        db.update_backfill_state(conn, EXCHANGE, symbol, interval, earliest, latest)
    return total


def import_generic(path: str, symbol: str, interval: int, conn) -> int:
    # Read just the header first to resolve column names once.
    header_df = pd.read_csv(path, nrows=0)
    columns = list(header_df.columns)

    ts_col = _find_column(columns, GENERIC_COLUMN_ALIASES["ts"])
    open_col = _find_column(columns, GENERIC_COLUMN_ALIASES["open"])
    high_col = _find_column(columns, GENERIC_COLUMN_ALIASES["high"])
    low_col = _find_column(columns, GENERIC_COLUMN_ALIASES["low"])
    close_col = _find_column(columns, GENERIC_COLUMN_ALIASES["close"])
    volume_col = _find_column(columns, GENERIC_COLUMN_ALIASES["volume"])
    vwap_col = _find_column(columns, GENERIC_COLUMN_ALIASES["vwap"])
    trades_col = _find_column(columns, GENERIC_COLUMN_ALIASES["trade_count"])

    missing = [
        name
        for name, col in [
            ("timestamp", ts_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
            ("volume", volume_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"couldn't find required column(s) {missing} in {path}. "
            f"Found columns: {columns}. Rename them or use --format kraken-dump instead."
        )

    total = 0
    earliest = None
    latest = None
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS):
        ts_series = _parse_timestamp_series(chunk[ts_col])
        candles = []
        for i, row in enumerate(chunk.itertuples(index=False)):
            row_dict = row._asdict()
            candles.append(
                _row_to_candle(
                    ts_series.iloc[i],
                    row_dict[open_col],
                    row_dict[high_col],
                    row_dict[low_col],
                    row_dict[close_col],
                    row_dict[volume_col],
                    row_dict.get(vwap_col) if vwap_col else None,
                    row_dict.get(trades_col) if trades_col else None,
                )
            )
        total += db.upsert_candles(conn, EXCHANGE, symbol, interval, candles)
        for c in candles:
            earliest = c.ts_unix if earliest is None else min(earliest, c.ts_unix)
            latest = c.ts_unix if latest is None else max(latest, c.ts_unix)
    if earliest is not None:
        db.update_backfill_state(conn, EXCHANGE, symbol, interval, earliest, latest)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Normalized symbol (e.g. BTC-USD) to store these candles under.")
    parser.add_argument("--interval", type=int, required=True, help="Candle resolution in minutes (this CSV's own resolution).")
    parser.add_argument("--format", choices=["kraken-dump", "generic"], required=True)
    parser.add_argument("--file", required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    known_symbols = {s.symbol for s in load_symbols()}
    if args.symbol not in known_symbols:
        print(
            f"warning: {args.symbol} isn't in config/config.example.toml's enabled symbols — "
            "importing anyway, but double-check the symbol name is what you intend.",
            file=sys.stderr,
        )

    conn = db.connect()
    try:
        if args.format == "kraken-dump":
            total = import_kraken_dump(args.file, args.symbol, args.interval, conn)
        else:
            total = import_generic(args.file, args.symbol, args.interval, conn)
    finally:
        conn.close()

    print(f"[{args.symbol}] interval={args.interval}min: upserted {total} candles from {args.file}", file=sys.stderr)


if __name__ == "__main__":
    main()
