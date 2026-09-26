# Training / trading universe

Candidate multi-asset universe for eventual model training and/or expanded
live coverage, dropped by Stephen on 2026-09-26. Grouped by role, with each
symbol checked against Kraken's real `/0/public/AssetPairs` on that date —
this is deliberately verified, not assumed, per the project's standing
"trust but verify" rule on Kraken symbol naming.

| Group | Symbols | Rationale (as given) |
|---|---|---|
| Macro Anchor Baseline | BTC, ETH | Baseline market direction, low-noise trends, flight-to-safety signals |
| Specialized Macro | XRP, DOGE | Decoupled news-driven flows (XRP) and extreme retail sentiment/pumps (DOGE) |
| High-Beta L1 Momentum | SOL, AVAX, NEAR, SUI | High-speed breakout continuation, volatility expansion, sharp pullback entries |
| DeFi & Structural Rotations | UNI, AAVE, PENDLE | Mean-reversion in range-bound markets (UNI/AAVE) vs. decoupled protocol trends (PENDLE) |
| Cross-Ecosystem / Infrastructure | LINK, TAO | High-liquidity infrastructure trends, distinct correlation cycles vs. BTC |

## Kraken availability (checked live, 2026-09-26)

All 13 base assets are listed on Kraken against both USD and USDT. Naming
by asset:

| Base | Kraken WS v2 pair | Kraken REST altname (USD) | Kraken REST altname (USDT) | Notes |
|---|---|---|---|---|
| BTC | `BTC/USD` (verified live) | `XXBTZUSD` | `XBTUSDT` | Classic REST/AssetPairs still calls this `XBT`; WS v2 uses `BTC`. Only WS v2 naming has been confirmed live so far — see kraken.rs. |
| ETH | `ETH/USD` (verified live) | `XETHZUSD` | `ETHUSDT` | |
| XRP | `XRP/USD` (unverified) | `XXRPZUSD` | `XRPUSDT` | |
| DOGE | `DOGE/USD` (unverified — Kraken's WS v2 docs suggest modern naming, but this project has NOT confirmed it live; classic REST still calls it `XDG`) | `XDGUSD` | `XDGUSDT` | Same legacy/modern split as BTC — verify with a live WS subscribe before enabling, exactly like BTC was verified |
| SOL | `SOL/USD` (unverified) | `SOLUSD` | `SOLUSDT` | |
| AVAX | `AVAX/USD` (unverified) | `AVAXUSD` | `AVAXUSDT` | |
| NEAR | `NEAR/USD` (unverified) | `NEARUSD` | no USDT pair listed | |
| SUI | `SUI/USD` (unverified) | `SUIUSD` | no USDT pair listed | |
| UNI | `UNI/USD` (unverified) | `UNIUSD` | no USDT pair listed | |
| AAVE | `AAVE/USD` (unverified) | `AAVEUSD` | no USDT pair listed | |
| PENDLE | `PENDLE/USD` (unverified) | `PENDLEUSD` | no USDT pair listed | |
| LINK | `LINK/USD` (unverified) | `LINKUSD` | `LINKUSDT` | |
| TAO | `TAO/USD` (unverified) | `TAOUSD` | no USDT pair listed | |

**"Unverified" above means**: confirmed to exist in Kraken's public
AssetPairs listing (a real, tradeable pair), but this project has not yet
opened a live WS v2 subscription or placed a REST order against it the way
BTC and ETH have been. Treat the WS v2 pair name as a strong inference
(same pattern as BTC/ETH, which do match), not a confirmed fact, until
each one is actually exercised live — same standard the project already
holds itself to for `rest_native_symbol` elsewhere.

**USD vs. USDT — not yet decided.** The matrix as given quotes everything
in USDT; the bot's config today (`config/config.example.toml`) quotes
everything in USD, and only USD pairs have been used in any live test so
far. Six of the thirteen (NEAR, SUI, UNI, AAVE, PENDLE, TAO) don't even
have a Kraken USDT pair — only USD. Whether to standardize on USD, USDT
where available, or mix per-symbol is an open decision, not yet made.

## Not yet done

- No historical data collection exists for any of these symbols — this
  project has no historical/backtesting pipeline at all yet, live-only.
- No config entries exist for anything beyond BTC-USD/ETH-USD.
- No decision on quote currency (see above).
- No live verification of WS v2 naming beyond BTC and ETH.
