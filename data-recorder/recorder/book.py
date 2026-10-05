"""Local copy of a Kraken v2 order book, kept only to validate the feed.

The recorder stores raw messages; this book exists so every update can be
checked against the checksum Kraken sends with it. A mismatch means our view
of the book diverged from Kraken's (missed message, bad parse), and the data
recorded since the last good snapshot cannot be trusted on replay.

Algorithm (https://docs.kraken.com/api/docs/guides/spot-ws-book-v2/), same as
rust-core/src/checksum.rs: top 10 asks (low to high) then top 10 bids (high to
low); each price and qty is padded to the pair's fixed precision, stripped of
its decimal point and leading zeros; concatenated; CRC-32.
"""

from __future__ import annotations

import zlib
from decimal import Decimal

Level = tuple[Decimal, Decimal]


def decimals_of(text: str) -> int:
    """Fractional digits in a config value such as "0.00000001" -> 8."""
    return len(text.split(".")[1]) if "." in text else 0


def format_component(value: Decimal, decimals: int) -> str:
    stripped = f"{value:.{decimals}f}".replace(".", "").lstrip("0")
    return stripped or "0"


def compute_checksum(asks: list[Level], bids: list[Level], price_decimals: int, qty_decimals: int) -> int:
    parts = []
    for price, qty in list(asks) + list(bids):
        parts.append(format_component(price, price_decimals))
        parts.append(format_component(qty, qty_decimals))
    return zlib.crc32("".join(parts).encode("ascii")) & 0xFFFFFFFF


class LocalBook:
    """Top-`depth` book. Kraken does not tell you when a level falls out of the
    subscribed depth, so after every message the book is trimmed back to
    `depth` levels per side (skipping that makes checksums drift)."""

    def __init__(self, depth: int, price_decimals: int, qty_decimals: int):
        self.depth = depth
        self.price_decimals = price_decimals
        self.qty_decimals = qty_decimals
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}

    def apply(self, bids: list[Level], asks: list[Level], snapshot: bool) -> None:
        if snapshot:
            self.bids.clear()
            self.asks.clear()
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for price, qty in levels:
                if qty == 0:
                    side.pop(price, None)
                else:
                    side[price] = qty
        if len(self.asks) > self.depth:
            for price in sorted(self.asks)[self.depth :]:
                del self.asks[price]
        if len(self.bids) > self.depth:
            for price in sorted(self.bids, reverse=True)[self.depth :]:
                del self.bids[price]

    def top(self, n: int = 10) -> tuple[list[Level], list[Level]]:
        asks = [(p, self.asks[p]) for p in sorted(self.asks)[:n]]
        bids = [(p, self.bids[p]) for p in sorted(self.bids, reverse=True)[:n]]
        return asks, bids

    def checksum(self) -> int:
        asks, bids = self.top(10)
        return compute_checksum(asks, bids, self.price_decimals, self.qty_decimals)
