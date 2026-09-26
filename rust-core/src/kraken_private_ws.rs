//! Kraken's private (authenticated) WebSocket v2 feed: the `executions`
//! channel, which streams real order status transitions and fills for the
//! authenticated account. This is what lets `PortfolioManager` on the
//! Python side (and eventually the risk engine's position tracking) see
//! actual fills instead of always reading zero — see portfolio.py's
//! docstring for the gap this closes.
//!
//! Entirely separate connection from kraken.rs's public market-data feed:
//! different host, requires an authentication token (fetched via REST —
//! see `KrakenRestClient::get_websockets_token`), and scoped to the
//! account's own orders rather than any symbol's public book.
//!
//! **What is and isn't verified here.** Confirmed live, with deliberately
//! fake API credentials: `KrakenRestClient::get_websockets_token`'s
//! signed request reaches Kraken's real `/0/private/GetWebSocketsToken`
//! endpoint and gets back a real, correctly-parsed `EAPI:Invalid key`
//! rejection — the same category of evidence as `AddOrder`'s signing
//! test. The reconnect loop was also confirmed live: it backs off and
//! retries indefinitely without crashing when the token fetch fails.
//!
//! NOT verified, because a real token requires a real account: the
//! private WS connection itself was never reached (token fetch fails
//! first), so the connection URL and the `executions` channel's message
//! schema remain unconfirmed. The message schema below (field names,
//! `order_status`/`exec_type` values) comes from Kraken's public
//! WebSocket v2 documentation, not from an observed execution report. The
//! private WS URL (`wss://ws-auth.kraken.com/v2`) is inferred by
//! combining Kraken's confirmed private-feed host
//! (`wss://ws-auth.kraken.com`, documented for the older v1 API) with the
//! same `/v2` suffix the public feed uses — a reasonable inference, not
//! something this project has connected to and confirmed. Before trusting
//! this for real position tracking, get a real token from a real account
//! and confirm both the URL and the schema against an actual execution
//! report.

use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;

use crate::config::SymbolConfig;
use crate::kraken_rest::KrakenRestClient;
use crate::proto::pb::{Decimal as PbDecimal, OrderSide, OrderStatus, OrderUpdate};
use crate::risk::RiskEngine;

const RECONNECT_DELAY: Duration = Duration::from_secs(3);
const PRIVATE_WS_URL: &str = "wss://ws-auth.kraken.com/v2";
/// How many recent exec_id/trade_id values to remember for de-duplication.
/// Bounded so a long-lived connection can't grow this unboundedly; a
/// redelivery arriving further back than this many executions ago would
/// slip through, which is an acceptable trade-off for a guard against the
/// common case (a reconnect or at-least-once redelivery replaying the last
/// few events), not a full exactly-once guarantee.
const DEDUP_WINDOW: usize = 512;

/// Runs forever, reconnecting on any error — fetching a fresh token on
/// every (re)connect rather than trying to track token expiry. A dropped
/// connection here must never take down order submission or market data;
/// it only means fills stop being observed until it reconnects.
///
/// `risk` is fed real fills (see `apply_fill` in risk.rs) as they're
/// observed on this feed, so `RiskEngine`'s position tracking and the
/// daily kill switch reflect what actually happened at the exchange
/// instead of staying at zero forever.
pub async fn run(
    symbols: Vec<SymbolConfig>,
    rest_client: KrakenRestClient,
    tx: broadcast::Sender<OrderUpdate>,
    risk: Arc<RiskEngine>,
) {
    if symbols.is_empty() {
        tracing::warn!("no symbols configured, not starting private execution feed");
        return;
    }

    let native_to_normalized: HashMap<String, String> = symbols
        .iter()
        .map(|s| (s.exchange_native_symbol.clone(), s.symbol.clone()))
        .collect();

    loop {
        tracing::info!(url = PRIVATE_WS_URL, "connecting to Kraken private execution feed");
        match connect_and_stream(&rest_client, &native_to_normalized, &tx, &risk).await {
            Ok(()) => tracing::warn!("private execution feed stream ended, reconnecting"),
            Err(e) => tracing::error!(error = %e, "private execution feed error, reconnecting"),
        }
        tokio::time::sleep(RECONNECT_DELAY).await;
    }
}

async fn connect_and_stream(
    rest_client: &KrakenRestClient,
    native_to_normalized: &HashMap<String, String>,
    tx: &broadcast::Sender<OrderUpdate>,
    risk: &Arc<RiskEngine>,
) -> anyhow::Result<()> {
    let token = rest_client
        .get_websockets_token()
        .await
        .map_err(|e| anyhow::anyhow!("failed to get private WS token: {e}"))?
        .token;

    let (ws_stream, _) = tokio_tungstenite::connect_async(PRIVATE_WS_URL).await?;
    let (mut write, mut read) = ws_stream.split();

    let subscribe_msg = serde_json::json!({
        "method": "subscribe",
        "params": {
            "channel": "executions",
            "token": token,
            "snap_orders": false,
            "snap_trades": false,
            "order_status": true,
        }
    });
    write.send(Message::Text(subscribe_msg.to_string())).await?;
    tracing::info!("sent executions subscription");

    // Bounded recently-seen exec_id/trade_id set, scoped to this connection
    // (a fresh connection starts with a clean slate — an event redelivered
    // across a reconnect is, at worst, applied once more, which is the
    // same trade-off DEDUP_WINDOW itself makes).
    let mut seen_exec_ids: VecDeque<String> = VecDeque::with_capacity(DEDUP_WINDOW);

    while let Some(msg) = read.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Ping(payload) => {
                write.send(Message::Pong(payload)).await?;
                continue;
            }
            Message::Close(frame) => {
                tracing::warn!(?frame, "kraken closed the private execution feed connection");
                return Ok(());
            }
            _ => continue,
        };

        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, raw = %text, "failed to parse kraken private feed message, skipping");
                continue;
            }
        };

        // A subscribe ack/error, heartbeat, or other non-executions
        // message is expected and not an error — but a failed subscribe
        // ack is worth surfacing loudly rather than silently doing
        // nothing forever.
        if value.get("method").and_then(|m| m.as_str()) == Some("subscribe") {
            let success = value.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
            if !success {
                tracing::error!(raw = %text, "kraken rejected the executions subscription");
            }
            continue;
        }
        if value.get("channel").and_then(|c| c.as_str()) != Some("executions") {
            continue;
        }
        let Some(entries) = value.get("data").and_then(|d| d.as_array()) else {
            continue;
        };

        for entry in entries {
            match build_order_update(entry, native_to_normalized) {
                Some(update) => {
                    // No subscriber yet is not an error.
                    let _ = tx.send(update);
                }
                None => {
                    tracing::debug!(raw = %entry, "execution report missing required fields, skipping");
                }
            }

            if let Some(fill) = build_fill_event(entry, native_to_normalized) {
                if let Some(exec_id) = &fill.exec_id {
                    if seen_exec_ids.contains(exec_id) {
                        tracing::debug!(exec_id, "duplicate execution report, skipping apply_fill");
                        continue;
                    }
                    if seen_exec_ids.len() >= DEDUP_WINDOW {
                        seen_exec_ids.pop_front();
                    }
                    seen_exec_ids.push_back(exec_id.clone());
                }
                risk.apply_fill(&fill.symbol, fill.side, fill.qty, fill.price).await;
            }
        }
    }

    Ok(())
}

/// Builds an `OrderUpdate` from one entry of the executions channel's
/// `data` array. Returns None for an entry missing fields this project
/// needs (order_id, symbol, order_status) — logged by the caller, not
/// treated as fatal, since a status-only or otherwise-shaped message we
/// don't yet handle shouldn't take the whole feed down.
fn build_order_update(entry: &serde_json::Value, native_to_normalized: &HashMap<String, String>) -> Option<OrderUpdate> {
    let exchange_order_id = entry.get("order_id")?.as_str()?.to_string();
    let native_symbol = entry.get("symbol")?.as_str()?;
    let normalized_symbol = native_to_normalized.get(native_symbol)?.clone();

    let kraken_status = entry.get("order_status").and_then(|s| s.as_str()).unwrap_or("");
    let status = map_order_status(kraken_status);

    // client_order_id is how order.rs correlates this back to a strategy
    // and a side (see OrderUpdate's proto docs — it carries no side of
    // its own). Kraken only echoes cl_ord_id if AddOrder was called with
    // one, which kraken_rest.rs::add_order always does — but treat its
    // absence as "unattributable" rather than guessing, same principle
    // as engine.py's handling of an update with no matching pending
    // order.
    let client_order_id = entry.get("cl_ord_id").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let cum_qty = decimal_field(entry, "cum_qty");
    let order_qty = decimal_field(entry, "order_qty");
    let remaining_quantity = match (order_qty, cum_qty) {
        (Some(total), Some(filled)) => Some((total - filled).max(Decimal::ZERO)),
        _ => None,
    };

    Some(OrderUpdate {
        client_order_id,
        exchange_order_id,
        symbol: normalized_symbol,
        status: status as i32,
        reject_reason: entry.get("reason").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        filled_quantity: cum_qty.map(|d| PbDecimal { value: d.to_string() }),
        remaining_quantity: remaining_quantity.map(|d| PbDecimal { value: d.to_string() }),
        avg_fill_price: decimal_field(entry, "avg_price").map(|d| PbDecimal { value: d.to_string() }),
        // Kraken's own `timestamp` field (an ISO-8601 string) isn't parsed
        // here — this is receipt time, not exchange time, the same
        // documented simplification kraken.rs makes for market data's
        // exchange_timestamp_ns.
        timestamp_ns: now_ns(),
    })
}

/// A single real execution (fill) event, extracted from an `executions`
/// channel entry for feeding into `RiskEngine::apply_fill`. Distinct from
/// `OrderUpdate` (which is Python/gRPC-facing and reports order status,
/// not per-event deltas) — this uses `last_qty`/`last_price`, the size and
/// price of *this specific* execution, rather than `cum_qty`/`avg_price`,
/// which are cumulative and would require locally-persisted delta-tracking
/// state to turn into a per-fill quantity (state that breaks across a
/// reconnect — see this module's top-level docs).
struct FillEvent {
    symbol: String,
    side: OrderSide,
    qty: Decimal,
    price: Decimal,
    exec_id: Option<String>,
}

/// Extracts a `FillEvent` from an executions-channel entry, but only for an
/// entry that actually represents a trade: `exec_type == "trade"` is
/// Kraken's documented marker for "this event carries a real execution",
/// as opposed to a pure status transition (new/canceled/amended/etc.) that
/// carries no fill to apply. Returns None for anything else, or for a
/// trade entry missing the fields needed to apply it.
fn build_fill_event(entry: &serde_json::Value, native_to_normalized: &HashMap<String, String>) -> Option<FillEvent> {
    if entry.get("exec_type").and_then(|v| v.as_str()) != Some("trade") {
        return None;
    }

    let native_symbol = entry.get("symbol")?.as_str()?;
    let symbol = native_to_normalized.get(native_symbol)?.clone();

    let side = match entry.get("side").and_then(|v| v.as_str()) {
        Some("buy") => OrderSide::Buy,
        Some("sell") => OrderSide::Sell,
        other => {
            tracing::warn!(?other, "trade execution missing/unrecognized side, skipping apply_fill");
            return None;
        }
    };

    let qty = decimal_field(entry, "last_qty")?;
    let price = decimal_field(entry, "last_price")?;
    if qty <= Decimal::ZERO {
        return None;
    }

    let exec_id = entry
        .get("exec_id")
        .or_else(|| entry.get("trade_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(FillEvent { symbol, side, qty, price, exec_id })
}

/// Maps Kraken v2's `order_status` values onto our proto's OrderStatus.
/// Kraken has no "rejected" status in this enum (a rejection normally
/// never reaches this channel — see kraken_rest.rs's synchronous
/// AddOrder response for that path instead); `expired` is mapped to
/// Canceled as the closest fit, not a perfect semantic match.
fn map_order_status(kraken_status: &str) -> OrderStatus {
    match kraken_status {
        "pending_new" | "new" => OrderStatus::Accepted,
        "partially_filled" => OrderStatus::PartiallyFilled,
        "filled" => OrderStatus::Filled,
        "canceled" | "expired" => OrderStatus::Canceled,
        other => {
            tracing::warn!(order_status = other, "unrecognized Kraken order_status, treating as Unspecified");
            OrderStatus::Unspecified
        }
    }
}

/// Kraken sends numeric fields as JSON numbers; `arbitrary_precision` on
/// serde_json preserves their exact text, parsed straight into Decimal —
/// same approach as kraken.rs's order book parsing.
fn decimal_field(entry: &serde_json::Value, key: &str) -> Option<Decimal> {
    let value = entry.get(key)?;
    Decimal::from_str(&value.to_string()).ok()
}

fn now_ns() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These test the parsing/mapping logic against synthetic JSON shaped
    // like Kraken's documented executions-channel schema — they do NOT
    // confirm that schema is what a real account actually receives (see
    // this module's top-level docs).

    fn native_map() -> HashMap<String, String> {
        HashMap::from([("BTC/USD".to_string(), "BTC-USD".to_string())])
    }

    #[test]
    fn maps_a_partial_fill() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{
                "order_id": "OK4GJX-KSTLS-7DZZO5",
                "cl_ord_id": "test-123",
                "symbol": "BTC/USD",
                "order_status": "partially_filled",
                "order_qty": 0.01,
                "cum_qty": 0.004,
                "avg_price": 26599.9
            }"#,
        )
        .unwrap();

        let update = build_order_update(&entry, &native_map()).expect("should parse");
        assert_eq!(update.client_order_id, "test-123");
        assert_eq!(update.exchange_order_id, "OK4GJX-KSTLS-7DZZO5");
        assert_eq!(update.symbol, "BTC-USD");
        assert_eq!(update.status, OrderStatus::PartiallyFilled as i32);
        assert_eq!(update.filled_quantity.unwrap().value, "0.004");
        // remaining = 0.01 - 0.004 = 0.006
        assert_eq!(update.remaining_quantity.unwrap().value, "0.006");
        assert_eq!(update.avg_fill_price.unwrap().value, "26599.9");
    }

    #[test]
    fn maps_a_full_fill() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{
                "order_id": "OK4GJX-KSTLS-7DZZO6",
                "cl_ord_id": "test-124",
                "symbol": "BTC/USD",
                "order_status": "filled",
                "order_qty": 0.01,
                "cum_qty": 0.01,
                "avg_price": 26600.0
            }"#,
        )
        .unwrap();

        let update = build_order_update(&entry, &native_map()).expect("should parse");
        assert_eq!(update.status, OrderStatus::Filled as i32);
        assert_eq!(update.remaining_quantity.unwrap().value, "0.00");
    }

    #[test]
    fn missing_order_id_is_skipped_not_panicked() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{"symbol": "BTC/USD", "order_status": "new"}"#,
        )
        .unwrap();
        assert!(build_order_update(&entry, &native_map()).is_none());
    }

    #[test]
    fn unmapped_symbol_is_skipped_not_guessed() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{"order_id": "X", "symbol": "DOGE/USD", "order_status": "new"}"#,
        )
        .unwrap();
        assert!(build_order_update(&entry, &native_map()).is_none());
    }

    #[test]
    fn missing_client_order_id_yields_empty_string_not_a_skip() {
        // An execution report with no cl_ord_id (e.g. an order placed by
        // some other means) should still be parsed — just unattributable
        // to a strategy downstream, which order.rs's registry handles.
        let entry: serde_json::Value = serde_json::from_str(
            r#"{"order_id": "X", "symbol": "BTC/USD", "order_status": "new"}"#,
        )
        .unwrap();
        let update = build_order_update(&entry, &native_map()).expect("should still parse");
        assert_eq!(update.client_order_id, "");
    }

    #[test]
    fn build_fill_event_extracts_a_trade_execution() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{
                "order_id": "OK4GJX-KSTLS-7DZZO5",
                "symbol": "BTC/USD",
                "exec_type": "trade",
                "side": "buy",
                "last_qty": 0.004,
                "last_price": 26599.9,
                "exec_id": "EXEC-1"
            }"#,
        )
        .unwrap();

        let fill = build_fill_event(&entry, &native_map()).expect("should parse a trade");
        assert_eq!(fill.symbol, "BTC-USD");
        assert_eq!(fill.side, OrderSide::Buy);
        assert_eq!(fill.qty, Decimal::from_str("0.004").unwrap());
        assert_eq!(fill.price, Decimal::from_str("26599.9").unwrap());
        assert_eq!(fill.exec_id.as_deref(), Some("EXEC-1"));
    }

    #[test]
    fn build_fill_event_ignores_non_trade_exec_types() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{
                "order_id": "OK4GJX-KSTLS-7DZZO5",
                "symbol": "BTC/USD",
                "exec_type": "new",
                "order_status": "new"
            }"#,
        )
        .unwrap();
        assert!(build_fill_event(&entry, &native_map()).is_none());
    }

    #[test]
    fn build_fill_event_falls_back_to_trade_id_when_exec_id_absent() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{
                "symbol": "BTC/USD",
                "exec_type": "trade",
                "side": "sell",
                "last_qty": 0.01,
                "last_price": 30000,
                "trade_id": "TRADE-9"
            }"#,
        )
        .unwrap();
        let fill = build_fill_event(&entry, &native_map()).expect("should parse");
        assert_eq!(fill.exec_id.as_deref(), Some("TRADE-9"));
    }

    #[test]
    fn build_fill_event_skips_a_trade_missing_last_qty() {
        let entry: serde_json::Value = serde_json::from_str(
            r#"{"symbol": "BTC/USD", "exec_type": "trade", "side": "buy", "last_price": 30000}"#,
        )
        .unwrap();
        assert!(build_fill_event(&entry, &native_map()).is_none());
    }

    #[test]
    fn status_mapping() {
        assert_eq!(map_order_status("pending_new"), OrderStatus::Accepted);
        assert_eq!(map_order_status("new"), OrderStatus::Accepted);
        assert_eq!(map_order_status("partially_filled"), OrderStatus::PartiallyFilled);
        assert_eq!(map_order_status("filled"), OrderStatus::Filled);
        assert_eq!(map_order_status("canceled"), OrderStatus::Canceled);
        assert_eq!(map_order_status("expired"), OrderStatus::Canceled);
        assert_eq!(map_order_status("something_kraken_hasn't_documented_yet"), OrderStatus::Unspecified);
    }
}
