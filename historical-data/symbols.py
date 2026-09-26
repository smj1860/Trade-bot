"""
Loads the trading symbol universe from the same config/config.example.toml
the Rust core and Python strategy layer already use, rather than
duplicating the BASE-USD <-> Kraken-REST-altname mapping a second time.
Every [[symbols]] block's `symbol` and `rest_native_symbol` fields are all
this module needs — the Kraken public OHLC/Trades endpoints take the same
`rest_native_symbol` altname (e.g. "XBTUSD") the execution client already
uses for AddOrder, since it's the same REST API surface.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.example.toml"


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str  # normalized "BASE-USD", e.g. "BTC-USD"
    rest_native_symbol: str  # Kraken REST altname, e.g. "XBTUSD"


def load_symbols(config_path: str | Path | None = None) -> list[SymbolSpec]:
    if config_path is None:
        config_path = os.environ.get("TRADING_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    path = Path(config_path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    specs = []
    for entry in raw.get("symbols", []):
        if not entry.get("enabled", True):
            continue
        specs.append(SymbolSpec(symbol=entry["symbol"], rest_native_symbol=entry["rest_native_symbol"]))
    return specs


if __name__ == "__main__":
    for s in load_symbols():
        print(s.symbol, s.rest_native_symbol)
