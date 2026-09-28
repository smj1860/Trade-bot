//! Kraken WebSocket API v2 ingestion: connects, subscribes to both the
//! `book` and `trade` channels for the configured symbols, maintains
//! per-symbol order book state, and publishes `MarketDataEvent`s onto a
//! broadcast channel that `MarketDataServiceImpl` streams out to gRPC
//! subscribers.
//!
//! Kraken specifics (as of WS API v2): pairs are slash-delimited modern
//! asset codes (e.g. "BTC/USD", not the legacy REST v0 "XXBTZUSD"). The
//! `book` channel sends a `snapshot` message once per symbol on subscribe,
//! then `update` deltas; a quantity of 0 in an update means "remove this
//! price level." The `trade` channel sends one message per batch of
//! executed trades — real traded price/size/side, not a book level — which
//! is what lets strategy.bars.BarAggregator on the Python side build live
//! bars (close/high/low) and VWAP from actual executions instead of
//! approximating them from mid-price book snapshots (see that module's
//! docstring and this crate's TradeUpdate proto message for why this
//! matters for train/live feature parity).

use std::collections::HashMap;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;

use crate::checksum::compute_book_checksum;
use crate::config::{ExchangeConfig, SymbolConfig};
use crate::orderbook::{OrderBook, SharedBooks};
use crate::proto::pb::market_data_event::Event;
use crate::proto::pb::{
    Decimal as PbDecimal, MarketDataEvent, OrderBookLevel, OrderBookUpdate, OrderSide, TradeUpdate,
};

const RECONNECT_DELAY: Duration = Duration::from_secs(3);
const BOOK_DEPTH: usize = 10;
const PUBLISHED_LEVELS: usize = 50;

/// Runs forever, reconnecting on any error. A dropped connection or a
/// parse error on one message must never take down market data for every
/// other symbol — that's why this loops rather than propagating errors up
/// to main().
pub async fn run(
    exchange: ExchangeConfig,
    symbols: Vec<SymbolConfig>,
    tx: broadcast::Sender<MarketDataEvent>,
    books: SharedBooks,
) {
    if symbols.is_empty() {
        tracing::warn!(exchange = %exchange.name, "no symbols configured for this exchange, not starting ingestion");
        return;
    }

    loop {
        tracing::info!(exchange = %exchange.name, url = %exchange.ws_url, "connecting");
        match connect_and_stream(&exchange, &symbols, &tx, &books).await {
            Ok(()) => tracing::warn!(exchange = %exchange.name, "stream ended, reconnecting"),
            Err(e) => tracing::error!(exchange = %exchange.name, error = %e, "stream error, reconnecting"),
        }
        tokio::time::sleep(RECONNECT_DELAY).await;
    }
}

async fn connect_and_stream(
    exchange: &ExchangeConfig,
    symbols: &[SymbolConfig],
    tx: &broadcast::Sender<MarketDataEvent>,
    books: &SharedBooks,
) -> anyhow::Result<()> {
    let (ws_stream, _) = tokio_tungstenite::connect_async(&exchange.ws_url).await?;
    let (mut write, mut read) = ws_stream.split();

    let native_symbols: Vec<&str> = symbols.iter().map(|s| s.exchange_native_symbol.as_str()).collect();
    let book_subscribe_msg = serde_json::json!({
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": native_symbols,
            "depth": BOOK_DEPTH,
        }
    });
    write.send(Message::Text(book_subscribe_msg.to_string())).await?;
    tracing::info!(symbols = ?native_symbols, "sent book subscription");

    // A separate subscribe message for the public trade tape — Kraken WS v2
    // treats each channel as its own subscription even over one connection.
    let trade_subscribe_msg = serde_json::json!({
        "method": "subscribe",
        "params": {
            "channel": "trade",
            "symbol": native_symbols,
        }
    });
    write.send(Message::Text(trade_subscribe_msg.to_string())).await?;
    tracing::info!(symbols = ?native_symbols, "sent trade subscription");

    // native symbol ("BTC/USD") -> normalized symbol ("BTC-USD"). Actual book
    // state lives in the shared `books` map, keyed by normalized symbol, so
    // the risk engine can read current prices without going through the
    // broadcast channel.
    let native_to_normalized: HashMap<String, String> = symbols
        .iter()
        .map(|s| (s.exchange_native_symbol.clone(), s.symbol.clone()))
        .collect();

    // native symbol -> (price_decimals, qty_decimals), for padding checksum
    // input to the pair's fixed precision (see checksum.rs's top-level
    // docs for why this padding is required, not optional).
    let checksum_decimals: HashMap<String, (u32, u32)> = symbols
        .iter()
        .map(|s| (s.exchange_native_symbol.clone(), (s.price_decimals(), s.qty_decimals())))
        .collect();

    while let Some(msg) = read.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Ping(payload) => {
                write.send(Message::Pong(payload)).await?;
                continue;
            }
            Message::Close(frame) => {
                tracing::warn!(?frame, "kraken closed the connection");
                return Ok(());
            }
            _ => continue,
        };

        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, raw = %text, "failed to parse kraken message, skipping");
                continue;
            }
        };

        // Anything besides book/trade data messages (subscribe ack,
        // heartbeat, status) is expected and not an error — just nothing to
        // publish.
        let channel = value.get("channel").and_then(|c| c.as_str());
        let msg_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let Some(entries) = value.get("data").and_then(|d| d.as_array()) else {
            continue;
        };

        match channel {
            Some("book") => {
                for entry in entries {
                    handle_book_entry(
                        entry,
                        msg_type,
                        exchange,
                        &native_to_normalized,
                        &checksum_decimals,
                        books,
                        tx,
                    )
                    .await?;
                }
            }
            Some("trade") => {
                for entry in entries {
                    handle_trade_entry(entry, exchange, &native_to_normalized, tx);
                }
            }
            _ => continue,
        }
    }

    Ok(())
}

/// Applies one book entry to local state and publishes the resulting
/// levels. Returns `Err` only for a checksum mismatch (see checksum.rs) —
/// deliberately propagated up through `connect_and_stream` to force a full
/// reconnect+resubscribe, which is the simplest way to guarantee every
/// symbol's book gets a fresh, trustworthy snapshot again. A narrower
/// per-symbol resubscribe was considered and rejected for a first version:
/// it would need its own unsubscribe/subscribe round-trip whose behavior
/// under Kraken's real v2 API has not been exercised here, whereas the
/// full-reconnect path reuses `run()`'s existing, already-tested backoff
/// loop. The cost is momentarily dropping every symbol's book on one
/// symbol's desync, not just the affected one — an accepted trade-off for
/// how rare a checksum mismatch should be in practice.
async fn handle_book_entry(
    entry: &serde_json::Value,
    msg_type: &str,
    exchange: &ExchangeConfig,
    native_to_normalized: &HashMap<String, String>,
    checksum_decimals: &HashMap<String, (u32, u32)>,
    books: &SharedBooks,
    tx: &broadcast::Sender<MarketDataEvent>,
) -> anyhow::Result<()> {
    let Some(native_symbol) = entry.get("symbol").and_then(|s| s.as_str()) else {
        return Ok(());
    };
    let Some(normalized_symbol) = native_to_normalized.get(native_symbol) else {
        tracing::warn!(%native_symbol, "book update for a symbol we didn't subscribe to");
        return Ok(());
    };
    // Defaults to (0, 0) — i.e. no padding — only if a symbol somehow has
    // no decimals entry, which can't happen via connect_and_stream's
    // construction (same `symbols` slice builds both maps) but would
    // otherwise silently reintroduce the unpadded-checksum bug rather than
    // failing loudly, so tests that build this map by hand must populate it.
    let &(price_decimals, qty_decimals) = checksum_decimals.get(native_symbol).unwrap_or(&(0, 0));

    let bids = parse_levels(entry.get("bids"));
    let asks = parse_levels(entry.get("asks"));
    // Kraken sends this as a JSON number; `arbitrary_precision` (see
    // parse_levels' docs) doesn't affect integers, but `as_u64` handles it
    // regardless of the exact JSON number representation.
    let received_checksum = entry.get("checksum").and_then(|c| c.as_u64());

    // Hold the lock only long enough to apply the delta, validate the
    // checksum, and read back the levels we're about to publish — never
    // across an await point, so a slow subscriber can't stall ingestion.
    let (published_bids, published_asks, checksum_mismatch) = {
        let mut guard = books.lock().await;
        let book = guard
            .entry(normalized_symbol.clone())
            .or_insert_with(|| OrderBook::with_depth(normalized_symbol.clone(), BOOK_DEPTH));

        match msg_type {
            "snapshot" => book.apply_snapshot(bids, asks),
            "update" => book.apply_update(bids, asks),
            other => {
                tracing::debug!(msg_type = other, "unhandled book message type");
                return Ok(());
            }
        }

        let computed_checksum =
            compute_book_checksum(&book.ask_levels(10), &book.bid_levels(10), price_decimals, qty_decimals);
        let mismatch = match received_checksum {
            Some(expected) if u64::from(computed_checksum) != expected => {
                tracing::error!(
                    symbol = %normalized_symbol,
                    msg_type,
                    expected,
                    computed = computed_checksum,
                    top_asks = ?book.ask_levels(10),
                    top_bids = ?book.bid_levels(10),
                    raw_entry = %entry,
                    "order book checksum mismatch — local book is desynced from Kraken's, \
                     dropping local state and forcing a reconnect+resubscribe"
                );
                true
            }
            // No checksum field on this message (or one that already
            // matches) — nothing wrong here.
            _ => false,
        };

        let published_bids = book.bid_levels(PUBLISHED_LEVELS);
        let published_asks = book.ask_levels(PUBLISHED_LEVELS);
        // `book`'s borrow of `guard` ends with the reads above, so
        // mutating `guard` directly below (still under the same lock
        // hold) is fine.
        if mismatch {
            guard.remove(normalized_symbol);
        }

        (published_bids, published_asks, mismatch)
    };

    if checksum_mismatch {
        anyhow::bail!("order book checksum mismatch for {normalized_symbol}");
    }

    let event = MarketDataEvent {
        event: Some(Event::OrderBookUpdate(OrderBookUpdate {
            symbol: normalized_symbol.clone(),
            exchange: exchange.name.clone(),
            exchange_timestamp_ns: 0, // Kraken's timestamp string isn't parsed yet; not needed for the pipe to work
            received_timestamp_ns: now_ns(),
            bids: to_levels(published_bids),
            asks: to_levels(published_asks),
            sequence: 0, // Kraken v2's own book checksum (validated above) serves this role instead
        })),
    };

    // No subscribers yet is not an error — the ingestion loop keeps the
    // book warm regardless of whether anyone's listening.
    let _ = tx.send(event);
    Ok(())
}

fn handle_trade_entry(
    entry: &serde_json::Value,
    exchange: &ExchangeConfig,
    native_to_normalized: &HashMap<String, String>,
    tx: &broadcast::Sender<MarketDataEvent>,
) {
    let Some(native_symbol) = entry.get("symbol").and_then(|s| s.as_str()) else {
        return;
    };
    let Some(normalized_symbol) = native_to_normalized.get(native_symbol) else {
        tracing::warn!(%native_symbol, "trade for a symbol we didn't subscribe to");
        return;
    };
    let Some(trade) = parse_trade_entry(entry) else {
        tracing::warn!(%native_symbol, raw = %entry, "failed to parse trade entry, skipping");
        return;
    };

    let event = MarketDataEvent {
        event: Some(Event::TradeUpdate(TradeUpdate {
            symbol: normalized_symbol.clone(),
            exchange: exchange.name.clone(),
            exchange_timestamp_ns: 0, // Kraken's timestamp string isn't parsed yet, same simplification as book updates
            received_timestamp_ns: now_ns(),
            price: Some(PbDecimal { value: trade.price.to_string() }),
            quantity: Some(PbDecimal { value: trade.quantity.to_string() }),
            side: trade.side as i32,
            trade_id: trade.trade_id,
        })),
    };

    let _ = tx.send(event);
}

/// Kraken sends price/qty as JSON numbers. `arbitrary_precision` on
/// serde_json preserves their exact original text, which we parse straight
/// into Decimal rather than round-tripping through f64 and losing
/// precision.
fn parse_levels(value: Option<&serde_json::Value>) -> Vec<(Decimal, Decimal)> {
    let Some(arr) = value.and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|lvl| {
            let price = lvl.get("price")?;
            let qty = lvl.get("qty")?;
            let price = Decimal::from_str(&price.to_string()).ok()?;
            let qty = Decimal::from_str(&qty.to_string()).ok()?;
            Some((price, qty))
        })
        .collect()
}

struct ParsedTrade {
    price: Decimal,
    quantity: Decimal,
    side: OrderSide,
    trade_id: String,
}

/// Parses one entry of a Kraken `trade` channel message, e.g.:
/// `{"symbol": "BTC/USD", "side": "buy", "price": 43000.1, "qty": 0.001,
/// "ord_type": "market", "trade_id": 123456789, "timestamp": "..."}`.
/// Returns None if price or qty is missing/unparseable — a trade without a
/// real price/size isn't one this pipeline can do anything useful with, so
/// it's dropped (and logged by the caller) rather than published with a
/// zero/placeholder value that could silently corrupt a bar's VWAP.
fn parse_trade_entry(entry: &serde_json::Value) -> Option<ParsedTrade> {
    let price = entry.get("price")?;
    let qty = entry.get("qty")?;
    let price = Decimal::from_str(&price.to_string()).ok()?;
    let quantity = Decimal::from_str(&qty.to_string()).ok()?;
    let side = match entry.get("side").and_then(|s| s.as_str()) {
        Some("buy") => OrderSide::Buy,
        Some("sell") => OrderSide::Sell,
        _ => OrderSide::Unspecified,
    };
    // trade_id may come through as a JSON number or (on some feeds) a
    // string; either way it's carried as an opaque string for logging/dedup
    // on the Python side, never parsed as a number here.
    let trade_id = entry
        .get("trade_id")
        .map(|v| match v.as_str() {
            Some(s) => s.to_string(),
            None => v.to_string(),
        })
        .unwrap_or_default();
    Some(ParsedTrade { price, quantity, side, trade_id })
}

fn to_levels(levels: Vec<(Decimal, Decimal)>) -> Vec<OrderBookLevel> {
    levels
        .into_iter()
        .map(|(price, qty)| OrderBookLevel {
            price: Some(PbDecimal { value: price.to_string() }),
            quantity: Some(PbDecimal { value: qty.to_string() }),
        })
        .collect()
}

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_trade_entry_buy_side() {
        let entry = serde_json::json!({
            "symbol": "BTC/USD",
            "side": "buy",
            "price": 43000.1,
            "qty": 0.001,
            "ord_type": "market",
            "trade_id": 123456789,
            "timestamp": "2023-09-25T07:49:37.708299Z",
        });
        let trade = parse_trade_entry(&entry).expect("should parse");
        assert_eq!(trade.price, Decimal::from_str("43000.1").unwrap());
        assert_eq!(trade.quantity, Decimal::from_str("0.001").unwrap());
        assert_eq!(trade.side, OrderSide::Buy);
        assert_eq!(trade.trade_id, "123456789");
    }

    #[test]
    fn parse_trade_entry_sell_side() {
        let entry = serde_json::json!({
            "symbol": "ETH/USD",
            "side": "sell",
            "price": 2500.55,
            "qty": 0.25,
            "trade_id": 42,
        });
        let trade = parse_trade_entry(&entry).expect("should parse");
        assert_eq!(trade.side, OrderSide::Sell);
    }

    #[test]
    fn parse_trade_entry_unrecognized_side_is_unspecified() {
        let entry = serde_json::json!({
            "symbol": "BTC/USD",
            "price": 43000.1,
            "qty": 0.001,
        });
        let trade = parse_trade_entry(&entry).expect("should still parse without a side");
        assert_eq!(trade.side, OrderSide::Unspecified);
        assert_eq!(trade.trade_id, ""); // missing trade_id defaults to empty, not an error
    }

    #[test]
    fn parse_trade_entry_string_trade_id_kept_verbatim() {
        let entry = serde_json::json!({
            "price": 100.0,
            "qty": 1.0,
            "trade_id": "abc-123",
        });
        let trade = parse_trade_entry(&entry).expect("should parse");
        assert_eq!(trade.trade_id, "abc-123");
    }

    #[test]
    fn parse_trade_entry_missing_price_returns_none() {
        let entry = serde_json::json!({ "qty": 1.0 });
        assert!(parse_trade_entry(&entry).is_none());
    }

    #[test]
    fn parse_trade_entry_missing_qty_returns_none() {
        let entry = serde_json::json!({ "price": 100.0 });
        assert!(parse_trade_entry(&entry).is_none());
    }

    #[test]
    fn parse_trade_entry_unparseable_price_returns_none() {
        let entry = serde_json::json!({ "price": "not-a-number", "qty": 1.0 });
        assert!(parse_trade_entry(&entry).is_none());
    }

    fn test_exchange() -> ExchangeConfig {
        ExchangeConfig {
            name: "kraken".into(),
            ws_url: "wss://ws.kraken.com/v2".into(),
            rest_url: "https://api.kraken.com".into(),
            enabled: true,
        }
    }

    // Built via serde_json::from_str on real JSON text, not the `json!`
    // macro — deliberately, so this exercises the exact same
    // arbitrary_precision string-preserving parse path production code
    // uses (see parse_levels' docs), which a float literal built through
    // `json!` would not reliably go through.
    fn book_snapshot_entry(checksum: &str) -> serde_json::Value {
        let raw = format!(
            r#"{{"symbol": "BTC/USD", "bids": [{{"price": 99.0, "qty": 1.0}}], "asks": [{{"price": 100.0, "qty": 1.0}}], "checksum": {checksum}}}"#
        );
        serde_json::from_str(&raw).unwrap()
    }

    // Both test levels are already at 1 decimal place, so price_decimals=1,
    // qty_decimals=1 is a no-op pad here — realistic per-pair values are
    // exercised separately in checksum.rs and config.rs.
    const TEST_PRICE_DECIMALS: u32 = 1;
    const TEST_QTY_DECIMALS: u32 = 1;

    fn test_checksum_decimals() -> HashMap<String, (u32, u32)> {
        let mut m = HashMap::new();
        m.insert("BTC/USD".to_string(), (TEST_PRICE_DECIMALS, TEST_QTY_DECIMALS));
        m
    }

    fn expected_checksum_for_test_book() -> u32 {
        let d = |s: &str| Decimal::from_str(s).unwrap();
        crate::checksum::compute_book_checksum(
            &[(d("100.0"), d("1.0"))],
            &[(d("99.0"), d("1.0"))],
            TEST_PRICE_DECIMALS,
            TEST_QTY_DECIMALS,
        )
    }

    #[tokio::test]
    async fn handle_book_entry_accepts_a_matching_checksum_and_keeps_the_book() {
        let books = crate::orderbook::new_shared_books();
        let (tx, _rx) = broadcast::channel(16);
        let mut native_to_normalized = HashMap::new();
        native_to_normalized.insert("BTC/USD".to_string(), "BTC-USD".to_string());
        let checksum_decimals = test_checksum_decimals();

        let entry = book_snapshot_entry(&expected_checksum_for_test_book().to_string());
        let result = handle_book_entry(
            &entry,
            "snapshot",
            &test_exchange(),
            &native_to_normalized,
            &checksum_decimals,
            &books,
            &tx,
        )
        .await;

        assert!(result.is_ok());
        assert!(books.lock().await.contains_key("BTC-USD"));
    }

    #[tokio::test]
    async fn handle_book_entry_rejects_a_mismatched_checksum_and_drops_the_book() {
        let books = crate::orderbook::new_shared_books();
        let (tx, _rx) = broadcast::channel(16);
        let mut native_to_normalized = HashMap::new();
        native_to_normalized.insert("BTC/USD".to_string(), "BTC-USD".to_string());
        let checksum_decimals = test_checksum_decimals();

        // Deliberately wrong — one off the real value.
        let wrong = expected_checksum_for_test_book().wrapping_add(1);
        let entry = book_snapshot_entry(&wrong.to_string());
        let result = handle_book_entry(
            &entry,
            "snapshot",
            &test_exchange(),
            &native_to_normalized,
            &checksum_decimals,
            &books,
            &tx,
        )
        .await;

        assert!(result.is_err(), "a checksum mismatch should force a reconnect via an Err");
        assert!(
            !books.lock().await.contains_key("BTC-USD"),
            "the desynced book should be dropped, not left for something else to trust"
        );
    }

    #[tokio::test]
    async fn handle_book_entry_with_no_checksum_field_is_accepted() {
        // Defensive: if a message type ever omits checksum, this must not
        // be treated as a mismatch.
        let books = crate::orderbook::new_shared_books();
        let (tx, _rx) = broadcast::channel(16);
        let mut native_to_normalized = HashMap::new();
        native_to_normalized.insert("BTC/USD".to_string(), "BTC-USD".to_string());
        let checksum_decimals = test_checksum_decimals();

        let raw = r#"{"symbol": "BTC/USD", "bids": [{"price": 99.0, "qty": 1.0}], "asks": [{"price": 100.0, "qty": 1.0}]}"#;
        let entry: serde_json::Value = serde_json::from_str(raw).unwrap();
        let result = handle_book_entry(
            &entry,
            "snapshot",
            &test_exchange(),
            &native_to_normalized,
            &checksum_decimals,
            &books,
            &tx,
        )
        .await;

        assert!(result.is_ok());
        assert!(books.lock().await.contains_key("BTC-USD"));
    }

    #[tokio::test]
    async fn handle_book_entry_pads_to_configured_decimals_so_kraken_trailing_zero_stripping_matches() {
        // Regression test for the real production incident this padding
        // fix addresses: a book entry whose price/qty arrive with FEWER
        // digits than the pair's configured precision (exactly what
        // Kraken's own wire messages do — see checksum.rs's docs) must
        // still validate correctly once padded to that precision, not be
        // treated as a mismatch just because the wire text was shorter
        // than the fully-padded form.
        let books = crate::orderbook::new_shared_books();
        let (tx, _rx) = broadcast::channel(16);
        let mut native_to_normalized = HashMap::new();
        native_to_normalized.insert("BTC/USD".to_string(), "BTC-USD".to_string());
        let mut checksum_decimals = HashMap::new();
        checksum_decimals.insert("BTC/USD".to_string(), (7u32, 8u32));

        let d = |s: &str| Decimal::from_str(s).unwrap();
        // Expected checksum computed as Kraken would: pad "99" -> 7 decimals
        // and "1" -> 8 decimals before stripping, even though the wire
        // entry below sends bare integers.
        let expected =
            crate::checksum::compute_book_checksum(&[(d("100"), d("1"))], &[(d("99"), d("1"))], 7, 8);

        let raw = format!(
            r#"{{"symbol": "BTC/USD", "bids": [{{"price": 99, "qty": 1}}], "asks": [{{"price": 100, "qty": 1}}], "checksum": {expected}}}"#
        );
        let entry: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let result = handle_book_entry(
            &entry,
            "snapshot",
            &test_exchange(),
            &native_to_normalized,
            &checksum_decimals,
            &books,
            &tx,
        )
        .await;

        assert!(result.is_ok(), "padding to configured decimals should make this checksum match: {result:?}");
        assert!(books.lock().await.contains_key("BTC-USD"));
    }
}
