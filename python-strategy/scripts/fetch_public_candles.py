"""Downloads public exchange candles to CSV for out-of-sample checks of the rule backtest.

    python scripts/fetch_public_candles.py --symbols LTC,BCH --interval 4h --start 2020-01-01 --out candles/

Source: Binance's public market-data mirror (data-api.binance.vision), USDT pairs, no key. Writes
<BASE>-USD.csv with columns ts (unix seconds), high, low, close, volume, which rule_backtest.py
--csv-dir reads. Prices differ slightly from Kraken's USD pairs; for direction/volatility tests
that is immaterial, and the original 14 coins are fetched too so the two sources can be compared.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://data-api.binance.vision/api/v3/klines"
STEP_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int, session: requests.Session) -> list[list]:
    out, t = [], start_ms
    while t < end_ms:
        for attempt in range(4):
            r = session.get(BASE, params={"symbol": f"{symbol}USDT", "interval": interval, "startTime": t, "endTime": end_ms, "limit": 1000}, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code == 400:  # unknown symbol
                return out
            time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"{symbol}: HTTP {r.status_code} {r.text[:120]}")
        rows = r.json()
        if not rows:
            break
        out += rows
        t = rows[-1][0] + STEP_MS[interval]
        if len(rows) < 1000:
            break
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", required=True, help="comma-separated base assets, e.g. LTC,BCH")
    p.add_argument("--interval", default="4h", choices=list(STEP_MS))
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None, help="default: now")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    start = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.end else int(time.time() * 1000)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    for base in [x.strip().upper() for x in args.symbols.split(",") if x.strip()]:
        rows = fetch(base, args.interval, start, end, s)
        if not rows:
            print(f"{base}: no data", file=sys.stderr)
            continue
        # keep only closed candles
        rows = [r for r in rows if r[6] < end]
        with (out / f"{base}-USD.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "high", "low", "close", "volume"])
            for r in rows:
                w.writerow([r[0] // 1000, r[2], r[3], r[4], r[5]])
        first = datetime.fromtimestamp(rows[0][0] / 1000, timezone.utc).date()
        print(f"{base}: {len(rows)} bars from {first}", file=sys.stderr)


if __name__ == "__main__":
    main()
