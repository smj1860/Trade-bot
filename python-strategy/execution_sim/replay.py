"""Replays recorder files (``<recv_ns>\\t<json>`` lines, .tsv.gz or .tsv.xz) for one symbol.

``Replayer.book`` is a live, mutable book that is already updated when an
event is yielded; consumers read it immediately and must not keep it. Timing
is always our receive time (``recv_ns``).

Gaps: a ``disconnect`` event invalidates every book, a ``checksum_mismatch``
invalidates that symbol's book, and a book is valid again only after the next
snapshot. A ``GapEvent`` is yielded so anything in flight can be discarded.
"""

from __future__ import annotations

import gzip
import json
import lzma
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

BID, ASK = 0, 1


def open_recording(path: str | Path):
    p = str(path)
    return lzma.open(p, "rb") if p.endswith(".xz") else gzip.open(p, "rb")


def find_recordings(paths: Iterable[str | Path]) -> list[Path]:
    """Files and directories -> chronologically sorted recording files.
    Names are YYYY/MM/DD/HH-<run id>.tsv.*, and run ids start with a UTC
    timestamp, so sorting the trailing path parts sorts by time."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += [f for f in p.rglob("*") if f.name.endswith((".tsv.gz", ".tsv.xz"))]
        else:
            out.append(p)
    return sorted(out, key=lambda f: f.parts[-4:])


class SymbolBook:
    """Top-``depth`` book for one symbol (float prices; dict level -> qty)."""

    __slots__ = ("depth", "sides", "valid")

    def __init__(self, depth: int = 25):
        self.depth = depth
        self.sides: tuple[dict[float, float], dict[float, float]] = ({}, {})
        self.valid = False

    def reset(self) -> None:
        self.sides[BID].clear()
        self.sides[ASK].clear()
        self.valid = False

    def apply(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]], snapshot: bool) -> list[tuple[int, float, float, float]]:
        """Applies one message; returns the changes as (side, price, old_qty, new_qty)."""
        if snapshot:
            self.sides[BID].clear()
            self.sides[ASK].clear()
        changes = []
        for side, levels in ((BID, bids), (ASK, asks)):
            book = self.sides[side]
            for price, qty in levels:
                old = book.get(price, 0.0)
                if qty == 0:
                    book.pop(price, None)
                else:
                    book[price] = qty
                if old != qty:
                    changes.append((side, price, old, qty))
            if len(book) > self.depth:  # Kraken does not announce levels leaving the subscribed depth
                keep = set(sorted(book, reverse=(side == BID))[: self.depth])
                for price in [p for p in book if p not in keep]:
                    del book[price]
        if snapshot:
            self.valid = True
        return changes

    def best(self, side: int) -> tuple[float, float] | None:
        book = self.sides[side]
        if not book:
            return None
        price = max(book) if side == BID else min(book)
        return price, book[price]

    def qty_at(self, side: int, price: float) -> float:
        return self.sides[side].get(price, 0.0)

    def levels(self, side: int, n: int | None = None) -> list[tuple[float, float]]:
        """Best first."""
        book = self.sides[side]
        prices = sorted(book, reverse=(side == BID))
        if n is not None:
            prices = prices[:n]
        return [(p, book[p]) for p in prices]

    def ready(self) -> bool:
        return self.valid and bool(self.sides[BID]) and bool(self.sides[ASK])


@dataclass(slots=True)
class BookEvent:
    t_ns: int
    changes: list[tuple[int, float, float, float]]
    snapshot: bool


@dataclass(slots=True)
class TradeEvent:
    t_ns: int
    aggressor_buy: bool  # True: a buyer lifted asks. A resting BID is hit by aggressor_buy == False.
    price: float
    qty: float


@dataclass(slots=True)
class GapEvent:
    t_ns: int
    reason: str


class MultiReplayer:
    """One pass over the files for several symbols at once. Yields
    ``(symbol, event)``; ``books[symbol]`` is that symbol's live book."""

    def __init__(self, paths: Iterable[str | Path], symbols: Iterable[str], depth: int = 25):
        self.files = find_recordings(paths)
        self.symbols = list(symbols)
        self.books = {s: SymbolBook(depth) for s in self.symbols}
        self._needles = [(s, f'"{s}"'.encode()) for s in self.symbols]

    def events(self, start_ns: int | None = None, end_ns: int | None = None) -> Iterator[tuple[str, BookEvent | TradeEvent | GapEvent]]:
        books, needles = self.books, self._needles
        for path in self.files:
            with open_recording(path) as f:
                for line in f:
                    tab = line.find(b"\t")
                    if tab < 0:
                        continue
                    payload = line[tab + 1:]
                    is_event = b'"_event"' in payload
                    if not is_event and not any(n in payload for _, n in needles):
                        continue
                    t = int(line[:tab])
                    if end_ns is not None and t > end_ns:
                        return
                    try:
                        msg = json.loads(payload)
                    except ValueError:
                        continue  # truncated final line of a crashed hour
                    if is_event:
                        name = msg.get("_event")
                        if name == "connect":
                            d = msg.get("depth")
                            for b in books.values():
                                b.reset()
                                if d:
                                    b.depth = int(d)
                        elif name == "disconnect":
                            for sym, b in books.items():
                                b.reset()
                                yield sym, GapEvent(t, "disconnect")
                        elif name == "checksum_mismatch" and msg.get("symbol") in books:
                            books[msg["symbol"]].reset()
                            yield msg["symbol"], GapEvent(t, "checksum_mismatch")
                        continue
                    channel = msg.get("channel")
                    if channel == "book":
                        snapshot = msg.get("type") == "snapshot"
                        for item in msg.get("data", ()):
                            sym = item.get("symbol")
                            book = books.get(sym)
                            if book is None:
                                continue
                            if not snapshot and not book.valid:
                                continue  # state unknown until a snapshot arrives
                            bids = [(l["price"], l["qty"]) for l in item.get("bids", ())]
                            asks = [(l["price"], l["qty"]) for l in item.get("asks", ())]
                            changes = book.apply(bids, asks, snapshot)
                            if start_ns is None or t >= start_ns:
                                yield sym, BookEvent(t, changes, snapshot)
                    elif channel == "trade" and msg.get("type") == "update" and (start_ns is None or t >= start_ns):
                        for tr in msg.get("data", ()):
                            if tr.get("symbol") in books:
                                yield tr["symbol"], TradeEvent(t, tr.get("side") == "buy", float(tr["price"]), float(tr["qty"]))


class Replayer:
    """Single-symbol convenience wrapper around MultiReplayer."""

    def __init__(self, paths: Iterable[str | Path], symbol: str, depth: int = 25):
        self.symbol = symbol
        self._multi = MultiReplayer(paths, [symbol], depth)
        self.files = self._multi.files
        self.book = self._multi.books[symbol]

    def events(self, start_ns: int | None = None, end_ns: int | None = None) -> Iterator[BookEvent | TradeEvent | GapEvent]:
        for _, ev in self._multi.events(start_ns, end_ns):
            yield ev
