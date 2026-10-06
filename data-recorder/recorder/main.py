"""Kraken spot order-book and trade recorder.

    python -m recorder.main --out ./data [--depth 25] [--duration 120]

Subscribes to the v2 `book` (level 2) and `trade` channels for every symbol in
config/config.example.toml and appends every message, as received, to hourly
gzip files (see writer.py). Runs forever by default: reconnects with backoff,
re-subscribes (which delivers a fresh snapshot), and reconnects if the feed goes
quiet or a symbol's local book fails its checksum repeatedly. It never trades
and shares nothing with the trading core, so it can run on any machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path

import websockets

from .book import LocalBook
from .config import SymbolInfo, load_symbols
from .uploader import Uploader
from .writer import RotatingWriter

log = logging.getLogger("recorder")
KRAKEN_WS = "wss://ws.kraken.com/v2"


class Resync(Exception):
    """Local state diverged from the feed; reconnect for fresh snapshots."""


class Recorder:
    def __init__(
        self,
        symbols: dict[str, SymbolInfo],
        writer: RotatingWriter,
        uploader: Uploader | None = None,
        depth: int = 100,
        url: str = KRAKEN_WS,
        stall_seconds: float = 30.0,
        max_consecutive_mismatches: int = 3,
    ):
        self.symbols = symbols
        self.writer = writer
        self.uploader = uploader
        self.depth = depth
        self.url = url
        self.stall_seconds = stall_seconds
        self.max_consecutive_mismatches = max_consecutive_mismatches
        self.stop = False
        self._started = time.monotonic()
        self.books: dict[str, LocalBook] = {}
        self._bad_streak: dict[str, int] = {}
        self.stats = {
            "messages": 0, "book_msgs": 0, "trade_msgs": 0, "checksum_checks": 0,
            "checksum_mismatches": 0, "reconnects": 0,
        }
        self.per_symbol_book_msgs: dict[str, int] = {}
        self._reset_books()

    # ---- state -----------------------------------------------------------
    def _reset_books(self) -> None:
        self.books = {ws: LocalBook(self.depth, s.price_decimals, s.qty_decimals) for ws, s in self.symbols.items()}
        self._bad_streak = {ws: 0 for ws in self.symbols}

    def event(self, name: str, **fields) -> None:
        self.writer.write(time.time_ns(), json.dumps({"_event": name, **fields}, separators=(",", ":")))

    # ---- message handling ------------------------------------------------
    def handle(self, raw: str, recv_ns: int) -> None:
        """Record one raw message and, for book messages, validate it."""
        msg = json.loads(raw, parse_float=Decimal)
        channel = msg.get("channel") if isinstance(msg, dict) else None
        if channel == "heartbeat":
            return  # one per second per connection; carries no market data
        self.writer.write(recv_ns, raw)
        self.stats["messages"] += 1
        if channel == "trade":
            self.stats["trade_msgs"] += 1
        elif channel == "book":
            self.stats["book_msgs"] += 1
            snapshot = msg.get("type") == "snapshot"
            for item in msg.get("data", []):
                self._apply_book(item, snapshot)
        elif isinstance(msg, dict) and msg.get("success") is False:
            log.error("subscription rejected: %s", raw[:300])

    def _apply_book(self, item: dict, snapshot: bool) -> None:
        sym = item.get("symbol")
        book = self.books.get(sym)
        if book is None:
            return
        self.per_symbol_book_msgs[sym] = self.per_symbol_book_msgs.get(sym, 0) + 1
        bids = [(Decimal(l["price"]), Decimal(l["qty"])) for l in item.get("bids", [])]
        asks = [(Decimal(l["price"]), Decimal(l["qty"])) for l in item.get("asks", [])]
        book.apply(bids, asks, snapshot)
        expected = item.get("checksum")
        if expected is None:
            return
        self.stats["checksum_checks"] += 1
        if book.checksum() == int(expected):
            self._bad_streak[sym] = 0
            return
        self.stats["checksum_mismatches"] += 1
        self._bad_streak[sym] += 1
        self.event("checksum_mismatch", symbol=sym, streak=self._bad_streak[sym], snapshot=snapshot)
        if self._bad_streak[sym] >= self.max_consecutive_mismatches:
            raise Resync(f"{sym}: {self._bad_streak[sym]} consecutive checksum mismatches")

    # ---- connection ------------------------------------------------------
    def _subscriptions(self) -> list[dict]:
        syms = list(self.symbols)
        return [
            {"method": "subscribe", "params": {"channel": "book", "symbol": syms, "depth": self.depth, "snapshot": True}},
            {"method": "subscribe", "params": {"channel": "trade", "symbol": syms, "snapshot": False}},
        ]

    async def _session(self, ws, deadline: float | None) -> None:
        for sub in self._subscriptions():
            await ws.send(json.dumps(sub))
        last_msg = time.monotonic()
        while not self.stop:
            if deadline is not None and time.monotonic() >= deadline:
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.monotonic() - last_msg > self.stall_seconds:
                    raise
                continue
            last_msg = time.monotonic()
            if isinstance(raw, bytes):
                raw = raw.decode()
            self.handle(raw, time.time_ns())

    async def run(self, duration: float | None = None) -> None:
        deadline = time.monotonic() + duration if duration else None
        backoff = 1.0
        housekeeping = asyncio.create_task(self._housekeeping())
        try:
            while not self.stop and (deadline is None or time.monotonic() < deadline):
                started = time.monotonic()
                self._reset_books()
                try:
                    async with websockets.connect(self.url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
                        self.event("connect", url=self.url, depth=self.depth, symbols=len(self.symbols))
                        log.info("connected")
                        await self._session(ws, deadline)
                except (Resync, OSError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as exc:
                    self.stats["reconnects"] += 1
                    self.event("disconnect", reason=repr(exc))
                    log.warning("disconnected: %r", exc)
                    backoff = 1.0 if time.monotonic() - started > 60 else min(backoff * 2, 60.0)
                    wait = backoff if deadline is None else max(0.0, min(backoff, deadline - time.monotonic()))
                    await asyncio.sleep(wait)
        finally:
            housekeeping.cancel()
            self.writer.close()
            if self.uploader:
                await asyncio.to_thread(self.uploader.sweep, set())

    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(60)
            log.info("stats %s", json.dumps(self.stats))
            if self.uploader:
                current = self.writer.current_path
                await asyncio.to_thread(self.uploader.sweep, {current} if current else set())

    def summary(self, root: Path) -> dict:
        files = list(Path(root).rglob("*.tsv.gz"))
        return {
            **self.stats,
            "uncompressed_bytes": self.writer.bytes_in,
            "compressed_bytes": sum(f.stat().st_size for f in files),
            "files": len(files),
            "cpu_seconds": round(time.process_time(), 2),
            "wall_seconds": round(time.monotonic() - self._started, 2),
            "book_msgs_by_symbol": dict(sorted(self.per_symbol_book_msgs.items())),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("RECORDER_OUT", "./data"), help="directory for the hourly files")
    p.add_argument("--depth", type=int, default=25, choices=[10, 25, 100, 500, 1000], help="book depth to subscribe to")
    p.add_argument("--config", default=None, help="config TOML with [[symbols]] (default: repo config/config.example.toml)")
    p.add_argument("--symbols", default="", help="comma-separated subset, e.g. BTC/USD,ETH/USD (default: all enabled)")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds (smoke tests); default run forever")
    p.add_argument("--url", default=KRAKEN_WS)
    p.add_argument("--log-level", default="INFO")
    return p


async def amain(args: argparse.Namespace) -> int:
    only = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    symbols = load_symbols(args.config, only)
    if not symbols:
        log.error("no symbols selected")
        return 2
    root = Path(args.out)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    rec = Recorder(symbols, RotatingWriter(root, run_id), Uploader.from_env(root), depth=args.depth, url=args.url)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: setattr(rec, "stop", True))
        except NotImplementedError:  # Windows
            pass
    log.info("recording %d symbols at depth %d to %s", len(symbols), args.depth, root)
    await rec.run(args.duration)
    print(json.dumps(rec.summary(root), indent=2))
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
