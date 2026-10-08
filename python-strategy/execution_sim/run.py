"""Run the baseline execution policies over recorded data.

    python -m execution_sim.run --data /srv/recorder --symbol BTC/USD
    python -m execution_sim.run --fetch 2026/10/07 --cache /tmp/rec --symbol ETH/USD --notional 1000

Prints, per fee tier, each policy's cost against arrival mid and its paired
difference from crossing immediately. See engine.py for what the fill model
assumes: all of it is counterfactual (our orders never changed the recorded book).
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

from .engine import NS, Fees, SimConfig, simulate
from .policies import default_policies
from .replay import Replayer, find_recordings
from .report import full_report

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.example.toml"
DEFAULT_FEES = "0.004:0.008,0.003:0.006,0.0022:0.0038"


def symbol_info(symbol: str, config: Path) -> tuple[float, int]:
    with config.open("rb") as f:
        raw = tomllib.load(f)
    for e in raw.get("symbols", []):
        if symbol in (e.get("exchange_native_symbol"), e.get("symbol")):
            tick = e["tick_size"]
            return float(tick), len(tick.split(".")[1]) if "." in tick else 0
    raise SystemExit(f"error: {symbol} is not in {config}")


def parse_fees(text: str) -> list[tuple[str, Fees]]:
    out = []
    for part in text.split(","):
        maker, taker = (float(x) for x in part.split(":"))
        out.append((f"fees {part}", Fees(maker, taker)))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*", default=[], help="recorder files or directories")
    p.add_argument("--fetch", default=None, help='download this day/month prefix (e.g. 2026/10/07) from the bucket in RECORDER_S3_* first')
    p.add_argument("--cache", default="/tmp/recorder-cache", help="where --fetch stores files")
    p.add_argument("--symbol", default="BTC/USD", help="Kraken v2 symbol as recorded, e.g. BTC/USD")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--notional", type=float, default=1000.0, help="quote-currency size of each parent order")
    p.add_argument("--deadline-s", type=float, default=600.0, help="time allowed to complete an order")
    p.add_argument("--every-s", type=float, default=300.0, help="a new pair of buy/sell orders starts this often")
    p.add_argument("--latency-ms", type=float, default=100.0)
    p.add_argument("--queue-factor", type=float, default=1.0, help="share of the displayed size at our level assumed ahead of us (1.0 = all, conservative)")
    p.add_argument("--cancel-credit", type=float, default=0.0, help="share of untraded displayed-size drops at our level that moves us up the queue (0 = conservative)")
    p.add_argument("--fees", default=DEFAULT_FEES, help="maker:taker per leg, comma separated (Kraken Tier 1/2/3 by default)")
    args = p.parse_args(argv)

    paths = list(args.data)
    if args.fetch:
        from .fetch import download
        paths += [str(x) for x in download(args.fetch, args.cache)]
    files = find_recordings(paths)
    if not files:
        raise SystemExit("error: no recorder files (.tsv.gz / .tsv.xz) found; pass --data or --fetch")
    tick, decimals = symbol_info(args.symbol, Path(args.config))
    cfg = SimConfig(tick=tick, price_decimals=decimals, latency_ns=int(args.latency_ms * 1e6),
                    queue_factor=args.queue_factor, cancel_credit=args.cancel_credit)
    rep = Replayer(files, args.symbol)
    t0 = time.time()
    results = simulate(rep.events(), rep.book, default_policies(), cfg, args.notional,
                       int(args.every_s * NS), int(args.deadline_s * NS))
    print(f"{args.symbol}: {len(files)} file(s), tick {tick}, notional {args.notional:g}, deadline {args.deadline_s:g}s, "
          f"latency {args.latency_ms:g}ms, queue_factor {args.queue_factor:g}, cancel_credit {args.cancel_credit:g}; "
          f"simulated in {time.time() - t0:.0f}s", file=sys.stderr)
    print(full_report(results, parse_fees(args.fees)))


if __name__ == "__main__":
    main()
