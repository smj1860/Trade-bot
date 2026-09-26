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
by asset — **all 13 WS v2 pair names below are now verified live**, not
inferred: each one was subscribed to on a real `wss://ws.kraken.com/v2`
public `book` channel, got a `success: true` ack, and produced real
snapshot data (bid/ask prices actually observed). `rest_native_symbol`,
`tick_size` and `lot_size` were pulled from a live query against
`/0/public/AssetPairs` (`altname`, `pair_decimals`, `lot_decimals`), not
copied from documentation.

| Base | Kraken WS v2 pair | Kraken REST altname (USD) | Kraken REST altname (USDT) | Notes |
|---|---|---|---|---|
| BTC | `BTC/USD` (verified live) | `XBTUSD` | `XBTUSDT` | Classic REST/AssetPairs still calls this `XBT`; WS v2 uses `BTC`. |
| ETH | `ETH/USD` (verified live) | `ETHUSD` | `ETHUSDT` | |
| XRP | `XRP/USD` (verified live) | `XRPUSD` | `XRPUSDT` | |
| DOGE | `DOGE/USD` (verified live) | `XDGUSD` | `XDGUSDT` | Same legacy/modern naming split as BTC: AssetPairs' `wsname` field misleadingly reports `XDG/USD`, but the real WS v2 name — confirmed by live subscription — is `DOGE/USD`, exactly like BTC/XBT. **Do not trust the `wsname` field for this project's WS v2 naming; it does not match the real WS v2 feed for either BTC or DOGE.** |
| SOL | `SOL/USD` (verified live) | `SOLUSD` | `SOLUSDT` | |
| AVAX | `AVAX/USD` (verified live) | `AVAXUSD` | `AVAXUSDT` | |
| NEAR | `NEAR/USD` (verified live) | `NEARUSD` | no USDT pair listed | |
| SUI | `SUI/USD` (verified live) | `SUIUSD` | no USDT pair listed | |
| UNI | `UNI/USD` (verified live) | `UNIUSD` | no USDT pair listed | |
| AAVE | `AAVE/USD` (verified live) | `AAVEUSD` | no USDT pair listed | |
| PENDLE | `PENDLE/USD` (verified live) | `PENDLEUSD` | no USDT pair listed | |
| LINK | `LINK/USD` (verified live) | `LINKUSD` | `LINKUSDT` | |
| TAO | `TAO/USD` (verified live) | `TAOUSD` | no USDT pair listed | |

**USD vs. USDT — decided: USD for all 13.** Six of the thirteen (NEAR, SUI,
UNI, AAVE, PENDLE, TAO) don't even have a Kraken USDT pair — only USD —
so USDT could never have covered the whole universe. Standardizing on USD
for every symbol keeps the config uniform and matches every live test run
so far. This is now reflected in `config/config.example.toml`.

## Not yet done

- No historical data collection exists for any of these symbols — this
  project has no historical/backtesting pipeline at all yet, live-only.
  (Next up: designing that pipeline.)
- No live smoke test yet of the full 13-symbol `config.example.toml`
  actually loading and streaming through the real binary (config file
  itself has been updated; the process hasn't been run against it yet).

## Done (2026-09-26)

- All 13 symbols added to `config/config.example.toml`, `enabled = true`,
  with live-verified `exchange_native_symbol`, `rest_native_symbol`,
  `tick_size`, and `lot_size`, grouped under the same five categories as
  the table above.
- Quote currency decided: USD for all 13 (see above).
- `[risk.global].max_total_position_usd` raised from `10000` to `20000` so
  the global cap stays meaningfully binding against the new sum of
  per-symbol caps (~24,500) rather than either blocking multi-symbol
  exposure or becoming decorative.
- All 13 symbols' WS v2 naming verified live (not just BTC/ETH) via real
  subscription + snapshot data.
