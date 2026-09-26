//! Kraken WebSocket API v2 ingestion: connects, subscribes to the `book`
//! channel for the configured symbols, maintains per-symbol order book
//! state, and publishes `MarketDataEvent`s onto a broadcast channel that
//! `MarketDataServiceImpl` streams out to gRPC subscribers.
//!
//! Kraken specifics (as of WS API v2): pairs are slash-delimited modern
//! asset codes (e.g. "BTC/USD", not the legacy REST v0 "XXBTZUSD"). The
//! `book` channel sends a `snapshot` message once per symbol on subscribe,
//! then `update` deltas; a quantity of 0 in an update means "remove this
//! price level."

use std::collections::HashMap;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;

use crate::config::{ExchangeConfig, SymbolConfig};
use crate::orderbook::{OrderBook, SharedBooks};
use crate::proto::pb::market_data_event::Event;
use crate::proto::pb::{
    Decimal as PbDecimal, MarketDataEvent, OrderBookLevel, OrderBookUpdate,
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
    let subscribe_msg = serde_json::json!({
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": native_symbols,
            "depth": BOOK_DEPTH,
        }
    });
    write.send(Message::Text(subscribe_msg.to_string())).await?;
    tracing::info!(symbols = ?native_symbols, "sent book subscription");

    // native symbol ("BTC/USD") -> normalized symbol ("BTC-USD"). Actual book
    // state lives in the shared `books` map, keyed by normalized symbol, so
    // the risk engine can read current prices without going through the
    // broadcast channel.
    let native_to_normalized: HashMap<String, String> = symbols
        .iter()
        .map(|s| (s.exchange_native_symbol.clone(), s.symbol.clone()))
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

        // Non-book messages (subscribe ack, heartbeat, status) are expected
        // and not errors — just nothing to publish.
        if value.get("channel").and_then(|c| c.as_str()) != Some("book") {
            continue;
        }
        let msg_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let Some(entries) = value.get("data").and_then(|d| d.as_array()) else {
            continue;
        };

        for entry in entries {
            let Some(native_symbol) = entry.get("symbol").and_then(|s| s.as_str()) else {
                continue;
            };
            let Some(normalized_symbol) = native_to_normalized.get(native_symbol) else {
                tracing::warn!(%native_symbol, "book update for a symbol we didn't subscribe to");
                continue;
            };

            let bids = parse_levels(entry.get("bids"));
            let asks = parse_levels(entry.get("asks"));

            // Hold the lock only long enough to apply the delta and read
            // back the levels we're about to publish — never across an
            // await point, so a slow subscriber can't stall ingestion.
            let (published_bids, published_asks) = {
                let mut guard = books.lock().await;
                let book = guard
                    .entry(normalized_symbol.clone())
                    .or_insert_with(|| OrderBook::new(normalized_symbol.clone()));

                match msg_type {
                    "snapshot" => book.apply_snapshot(bids, asks),
                    "update" => book.apply_update(bids, asks),
                    other => {
                        tracing::debug!(msg_type = other, "unhandled book message type");
                        continue;
                    }
                }

                (book.bid_levels(PUBLISHED_LEVELS), book.ask_levels(PUBLISHED_LEVELS))
            };

            let event = MarketDataEvent {
                event: Some(Event::OrderBookUpdate(OrderBookUpdate {
                    symbol: normalized_symbol.clone(),
                    exchange: exchange.name.clone(),
                    exchange_timestamp_ns: 0, // Kraken's timestamp string isn't parsed yet; not needed for the pipe to work
                    received_timestamp_ns: now_ns(),
                    bids: to_levels(published_bids),
                    asks: to_levels(published_asks),
                    sequence: 0, // Kraken v2's checksum serves this role; not wired in yet
                })),
            };

            // No subscribers yet is not an error — the ingestion loop keeps
            // the book warm regardless of whether anyone's listening.
            let _ = tx.send(event);
        }
    }

    Ok(())
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
