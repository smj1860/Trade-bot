//! Startup reconciliation: cross-checks this process's locally persisted
//! order state against Kraken's own record of what's actually open, so a
//! stale or incomplete local picture — from a missed private-feed event, a
//! crash mid-order, or simply the process being down when something
//! happened — is caught and logged rather than silently trusted forever.
//! This is the gap the persistence-layer README section flagged: a
//! restart previously trusted whatever `orders`/`positions` said with no
//! check against reality.
//!
//! Deliberately conservative about what it *does*, not just what it
//! *checks*: it updates a local order's status once Kraken's own
//! `QueryOrders` confirms it's closed/canceled/expired, but it never
//! mutates `RiskEngine`'s position or realized PnL here — those are only
//! ever changed by `RiskEngine::apply_fill`, driven by the live executions
//! feed, so running this can never introduce a double-counted fill. When a
//! since-closed order shows real executed quantity (`vol_exec > 0`) with
//! no matching row in the `fills` table, that's surfaced as a loud
//! `tracing::error!` for a human to check rather than silently backfilled
//! — see the module-level docs on `possible_missed_fills` for why.
//!
//! **NOT verified against a real account**, for the same reason as every
//! other private-endpoint code in this project (kraken_rest.rs,
//! kraken_private_ws.rs): the response schemas come from Kraken's public
//! REST documentation, and only the auth-failure path (deliberately
//! invalid credentials reaching the real endpoint) has been exercised
//! live. The pure decision logic below (`diff_local_vs_kraken_open`,
//! `resolve_closed_status`, `kraken_open_with_no_local_record`) is unit
//! tested against synthetic data independent of that, the same split
//! `kraken_rest.rs` uses for its signing logic.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rust_decimal::Decimal;

use crate::kraken_rest::{KrakenOrderInfo, KrakenRestClient};
use crate::persistence::{order_status_label, OrderRecord, Store};
use crate::proto::pb::OrderStatus;

/// Counts from one reconciliation pass, logged by the caller as a single
/// summary line so an operator sees the whole picture at a glance instead
/// of piecing it together from scattered log lines.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReconciliationSummary {
    pub locally_open_checked: usize,
    pub skipped_no_exchange_order_id: usize,
    pub confirmed_still_open: usize,
    pub closed_since_last_seen: usize,
    /// An order closed with executed quantity but no locally recorded
    /// fill — see the module docs. Always worth a human looking, since it
    /// means `RiskEngine`'s position for that symbol may be understated.
    pub possible_missed_fills: usize,
    /// An order Kraken considers open that this process has no local
    /// record of at all (placed manually, from another process, or from
    /// before this database existed). Logged, never adopted — this
    /// process doesn't know its strategy_id or original intent.
    pub kraken_open_with_no_local_record: usize,
}

/// Splits locally-open orders into (still open on Kraken, no longer open
/// on Kraken) by exchange_order_id membership in `kraken_open_txids`. An
/// order with no `exchange_order_id` at all (nothing was ever sent to
/// Kraken for it, or a plumbing failure before one was assigned) is
/// dropped from both — there's nothing on Kraken's side to check it
/// against. Pure and synchronous so it's testable without a network call.
fn diff_local_vs_kraken_open<'a>(
    local_open: &'a [OrderRecord],
    kraken_open_txids: &HashSet<String>,
) -> (Vec<&'a OrderRecord>, Vec<&'a OrderRecord>) {
    local_open
        .iter()
        .filter(|o| !o.exchange_order_id.is_empty())
        .partition(|o| kraken_open_txids.contains(&o.exchange_order_id))
}

/// Kraken open orders (by txid) that have no corresponding local record at
/// all.
fn kraken_open_with_no_local_record(
    kraken_open: &HashMap<String, KrakenOrderInfo>,
    local_exchange_order_ids: &HashSet<String>,
) -> Vec<String> {
    kraken_open.keys().filter(|txid| !local_exchange_order_ids.contains(*txid)).cloned().collect()
}

/// Maps a Kraken `QueryOrders` result (for an order no longer open) onto
/// the status label this project persists. `"closed"` with executed
/// quantity is a fill; `"closed"` with none, or `"canceled"`/`"expired"`,
/// is a cancellation. Anything else (Kraken still reporting it open, or an
/// unrecognized status) is left as `None` — reconciliation only downgrades
/// an order out of "open", it never invents a new open state.
fn resolve_closed_status(kraken_status: &str, vol_exec: Decimal) -> Option<&'static str> {
    match kraken_status {
        "closed" if vol_exec > Decimal::ZERO => Some(order_status_label(OrderStatus::Filled)),
        "closed" => Some(order_status_label(OrderStatus::Canceled)),
        "canceled" | "expired" => Some(order_status_label(OrderStatus::Canceled)),
        _ => None,
    }
}

/// Runs once at startup for one exchange's execution client. Never returns
/// an error for a reason that should stop the process from starting — a
/// network/auth failure talking to Kraken here just means reconciliation
/// didn't happen this run, which the caller logs loudly and moves on from,
/// the same way a failed persistence-store open is handled.
pub async fn reconcile_startup_state(
    store: &Arc<Store>,
    client: &KrakenRestClient,
    exchange_name: &str,
) -> anyhow::Result<ReconciliationSummary> {
    let mut summary = ReconciliationSummary::default();

    let local_open_for_exchange: Vec<OrderRecord> =
        store.load_open_orders()?.into_iter().filter(|o| o.exchange == exchange_name).collect();
    summary.locally_open_checked = local_open_for_exchange.len();
    summary.skipped_no_exchange_order_id =
        local_open_for_exchange.iter().filter(|o| o.exchange_order_id.is_empty()).count();

    let kraken_open = client.get_open_orders().await?;
    let kraken_open_txids: HashSet<String> = kraken_open.keys().cloned().collect();

    let (still_open, no_longer_open) = diff_local_vs_kraken_open(&local_open_for_exchange, &kraken_open_txids);
    summary.confirmed_still_open = still_open.len();

    for chunk in no_longer_open.chunks(50) {
        let txids: Vec<String> = chunk.iter().map(|o| o.exchange_order_id.clone()).collect();
        let queried = client.query_orders(&txids).await?;

        for order in chunk {
            let Some(info) = queried.get(&order.exchange_order_id) else {
                tracing::warn!(
                    client_order_id = %order.client_order_id,
                    exchange_order_id = %order.exchange_order_id,
                    "order is no longer in Kraken's open list but QueryOrders didn't return it either, leaving as-is"
                );
                continue;
            };
            let vol_exec = match Decimal::from_str(&info.vol_exec) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        exchange_order_id = %order.exchange_order_id,
                        raw = %info.vol_exec,
                        error = %e,
                        "failed to parse vol_exec from Kraken, treating as zero for this reconciliation pass"
                    );
                    Decimal::ZERO
                }
            };
            let Some(new_status) = resolve_closed_status(&info.status, vol_exec) else {
                continue;
            };

            tracing::warn!(
                client_order_id = %order.client_order_id,
                exchange_order_id = %order.exchange_order_id,
                kraken_status = %info.status,
                %vol_exec,
                new_status,
                "order closed on Kraken since this process last saw it, updating local record"
            );
            store.update_order_status(&order.client_order_id, &order.exchange_order_id, new_status, "", now_ns())?;
            summary.closed_since_last_seen += 1;

            if vol_exec > Decimal::ZERO && !store.has_fill_for_order(&order.client_order_id)? {
                summary.possible_missed_fills += 1;
                tracing::error!(
                    client_order_id = %order.client_order_id,
                    exchange_order_id = %order.exchange_order_id,
                    %vol_exec,
                    "order executed on Kraken but no fill was ever recorded locally — likely missed while \
                     this process was down. RiskEngine's position/PnL for this symbol may be understated. \
                     This is NOT backfilled automatically; check manually against Kraken's trade history."
                );
            }
        }
    }

    let local_exchange_order_ids: HashSet<String> = local_open_for_exchange
        .iter()
        .map(|o| o.exchange_order_id.clone())
        .filter(|id| !id.is_empty())
        .collect();
    let unknown = kraken_open_with_no_local_record(&kraken_open, &local_exchange_order_ids);
    summary.kraken_open_with_no_local_record = unknown.len();
    for txid in &unknown {
        tracing::warn!(exchange_order_id = %txid, "Kraken has an open order this process has no local record of");
    }

    Ok(summary)
}

fn now_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kraken_rest::KrakenOrderDescr;

    fn order(client_order_id: &str, exchange_order_id: &str, exchange: &str) -> OrderRecord {
        OrderRecord {
            client_order_id: client_order_id.to_string(),
            exchange_order_id: exchange_order_id.to_string(),
            symbol: "BTC-USD".to_string(),
            exchange: exchange.to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            quantity: "0.01".to_string(),
            limit_price: Some("30000".to_string()),
            strategy_id: "test-strategy".to_string(),
            status: "ACCEPTED".to_string(),
            reject_reason: String::new(),
            created_at_ns: 1,
            updated_at_ns: 1,
        }
    }

    #[test]
    fn diff_splits_by_kraken_open_membership() {
        let local = vec![order("co-1", "TX-1", "kraken"), order("co-2", "TX-2", "kraken")];
        let mut kraken_open = HashSet::new();
        kraken_open.insert("TX-1".to_string());

        let (still_open, closed) = diff_local_vs_kraken_open(&local, &kraken_open);
        assert_eq!(still_open.len(), 1);
        assert_eq!(still_open[0].client_order_id, "co-1");
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].client_order_id, "co-2");
    }

    #[test]
    fn diff_skips_orders_with_no_exchange_order_id() {
        let local = vec![order("co-1", "", "kraken")];
        let kraken_open = HashSet::new();
        let (still_open, closed) = diff_local_vs_kraken_open(&local, &kraken_open);
        assert!(still_open.is_empty());
        assert!(closed.is_empty());
    }

    #[test]
    fn resolve_closed_status_maps_a_fill() {
        assert_eq!(resolve_closed_status("closed", Decimal::from_str("0.01").unwrap()), Some("FILLED"));
    }

    #[test]
    fn resolve_closed_status_maps_a_zero_exec_close_to_canceled() {
        assert_eq!(resolve_closed_status("closed", Decimal::ZERO), Some("CANCELED"));
    }

    #[test]
    fn resolve_closed_status_maps_canceled_and_expired() {
        assert_eq!(resolve_closed_status("canceled", Decimal::ZERO), Some("CANCELED"));
        assert_eq!(resolve_closed_status("expired", Decimal::ZERO), Some("CANCELED"));
    }

    #[test]
    fn resolve_closed_status_ignores_still_open_or_unknown_statuses() {
        assert_eq!(resolve_closed_status("open", Decimal::ZERO), None);
        assert_eq!(resolve_closed_status("pending", Decimal::ZERO), None);
    }

    #[test]
    fn kraken_open_with_no_local_record_finds_the_gap() {
        let mut kraken_open = HashMap::new();
        kraken_open.insert(
            "TX-1".to_string(),
            KrakenOrderInfo {
                cl_ord_id: None,
                status: "open".to_string(),
                descr: KrakenOrderDescr { pair: "XBTUSD".to_string(), side: "buy".to_string() },
                vol: "1".to_string(),
                vol_exec: "0".to_string(),
            },
        );
        let local_ids: HashSet<String> = HashSet::new();
        let unknown = kraken_open_with_no_local_record(&kraken_open, &local_ids);
        assert_eq!(unknown, vec!["TX-1".to_string()]);
    }

    #[test]
    fn kraken_open_with_no_local_record_is_empty_when_everything_matches() {
        let mut kraken_open = HashMap::new();
        kraken_open.insert(
            "TX-1".to_string(),
            KrakenOrderInfo {
                cl_ord_id: Some("co-1".to_string()),
                status: "open".to_string(),
                descr: KrakenOrderDescr { pair: "XBTUSD".to_string(), side: "buy".to_string() },
                vol: "1".to_string(),
                vol_exec: "0".to_string(),
            },
        );
        let mut local_ids = HashSet::new();
        local_ids.insert("TX-1".to_string());
        assert!(kraken_open_with_no_local_record(&kraken_open, &local_ids).is_empty());
    }

    #[tokio::test]
    async fn reconcile_startup_state_updates_orders_confirmed_closed_on_kraken() {
        // No live Kraken here (that requires a funded account — see this
        // module's docs) — this exercises the store side of
        // reconcile_startup_state's logic directly via its pure helpers,
        // which is what's actually unit-testable without a network call.
        let store = Arc::new(Store::open_in_memory().unwrap());
        store.upsert_order(&order("co-1", "TX-1", "kraken")).unwrap();
        assert_eq!(store.load_open_orders().unwrap().len(), 1);

        // Simulate what reconcile_startup_state does once it learns TX-1
        // is closed with executed quantity, without needing a real
        // KrakenRestClient to produce that QueryOrders result.
        let vol_exec = Decimal::from_str("0.01").unwrap();
        let new_status = resolve_closed_status("closed", vol_exec).unwrap();
        store.update_order_status("co-1", "TX-1", new_status, "", 2).unwrap();

        assert!(store.load_open_orders().unwrap().is_empty(), "order should no longer show as open");
        assert!(
            !store.has_fill_for_order("co-1").unwrap(),
            "no fill was ever recorded — this is exactly the possible_missed_fills case"
        );
    }
}
