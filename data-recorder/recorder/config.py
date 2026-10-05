"""Symbol list for the recorder, read from the same config/config.example.toml
the trading core and historical-data pipeline use."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .book import decimals_of

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "config.example.toml"


@dataclass(frozen=True)
class SymbolInfo:
    ws_symbol: str  # Kraken v2 pair, e.g. "BTC/USD"
    price_decimals: int
    qty_decimals: int


def load_symbols(path: str | Path | None = None, only: list[str] | None = None) -> dict[str, SymbolInfo]:
    with Path(path or DEFAULT_CONFIG).open("rb") as f:
        raw = tomllib.load(f)
    out: dict[str, SymbolInfo] = {}
    for entry in raw.get("symbols", []):
        if not entry.get("enabled", True):
            continue
        ws = entry["exchange_native_symbol"]
        if only and ws not in only and entry["symbol"] not in only:
            continue
        out[ws] = SymbolInfo(ws, decimals_of(entry["tick_size"]), decimals_of(entry["lot_size"]))
    return out
