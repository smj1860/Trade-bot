//! Dead-man's switch: institutional audit Phase 1.1. Tracks the last time
//! each strategy_id sent a heartbeat (via `OrderService::SendHeartbeat`,
//! see trading.proto), so a background watchdog task (wired in `main.rs`)
//! can tell when a strategy process has gone dark — crashed, lost its
//! gRPC connection, or the machine it runs on went offline — and act on
//! it, rather than leaving an open position or a resting order with
//! nobody watching it.
//!
//! Deliberately split into two pieces: `HeartbeatMonitor` here is pure
//! bookkeeping (record a heartbeat, ask who's gone stale), fully testable
//! without any Kraken/gRPC involvement. What actually *happens* to a
//! stale strategy (canceling orders, optionally flattening positions,
//! alerting) lives in `main.rs`'s watchdog task, which is what actually
//! needs the execution client, the persistence store, and the risk
//! engine — this module knows about none of them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rust_decimal::Decimal;

use crate::alerting::AlertSink;
use crate::config::Config;
use crate::kraken_rest::{CancelOrderOutcome, KrakenRestClient};
use crate::order::OrderServiceImpl;
use crate::persistence::Store;
use crate::proto::pb::{Decimal as PbDecimal, OrderRequest, OrderSide, OrderType};
use crate::risk::RiskEngine;

/// The strategy_id attached to a synthetic reduce-to-flat order the
/// watchdog itself submits — never something a real Python strategy would
/// send — so it's obvious in logs/persistence which orders came from the
/// dead-man's switch rather than a live decision.
const WATCHDOG_STRATEGY_ID: &str = "dead-man-switch";

#[derive(Default)]
pub struct HeartbeatMonitor {
    last_seen: Mutex<HashMap<String, Instant>>,
}

impl HeartbeatMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a heartbeat for `strategy_id` at the current time. Called
    /// from the `SendHeartbeat` RPC handler on every call, regardless of
    /// whether this strategy_id has been seen before.
    pub fn record(&self, strategy_id: &str) {
        self.last_seen.lock().unwrap().insert(strategy_id.to_string(), Instant::now());
    }

    /// Every strategy_id that has ever sent a heartbeat whose most recent
    /// one is now older than `timeout`. A strategy_id that has *never*
    /// sent a heartbeat is not included — this module has no opinion on
    /// whether a strategy is required to exist at all, only on strategies
    /// that were once alive and have since gone quiet. Order of the
    /// returned list is unspecified.
    pub fn stale_strategies(&self, timeout: Duration) -> Vec<String> {
        let now = Instant::now();
        self.last_seen
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, &last)| now.duration_since(last) > timeout)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Removes a strategy_id from tracking entirely — used once the
    /// watchdog has acted on a stale strategy (canceled its orders,
    /// alerted, etc.) so it isn't re-flagged and re-acted-on every watchdog
    /// tick until it sends a fresh heartbeat. A later heartbeat from the
    /// same strategy_id simply re-adds it via `record`.
    pub fn clear(&self, strategy_id: &str) {
        self.last_seen.lock().unwrap().remove(strategy_id);
    }
}

/// Builds the alert text for a strategy the watchdog just declared dead.
/// Standalone so its wording is testable without spinning up the whole
/// watchdog loop, same reasoning as `alerting::possible_missed_fills_message`.
pub fn dead_man_switch_message(strategy_id: &str, timeout_secs: u64, auto_cancel: bool, auto_flatten: bool) -> String {
    format!(
        ":skull: Dead-man's switch tripped: strategy `{strategy_id}` has not sent a heartbeat in over \
         {timeout_secs}s and is being treated as dead. auto_cancel_orders={auto_cancel}, \
         auto_flatten_positions={auto_flatten}. See heartbeat.rs's dead-man's-switch docs."
    )
}

/// Cancels every locally-open order this process knows about for
/// `strategy_id`, one exchange call per order. Best-effort: a failure to
/// cancel any individual order is logged and does not stop the rest from
/// being attempted. Without a persistence store (it failed to open at
/// startup), there's no local record of which orders belong to which
/// strategy, so this is a no-op — logged, not silent.
async fn cancel_open_orders_for_strategy(
    strategy_id: &str,
    store: &Option<Arc<Store>>,
    execution_clients: &HashMap<String, KrakenRestClient>,
) {
    let Some(store) = store else {
        tracing::warn!(
            strategy_id,
            "dead-man's switch: no persistence store configured, cannot look up this strategy's open orders to cancel"
        );
        return;
    };

    let open_orders = match store.load_open_orders() {
        Ok(orders) => orders,
        Err(e) => {
            tracing::error!(strategy_id, error = %e, "dead-man's switch: failed to load open orders for cancellation");
            return;
        }
    };

    for order in open_orders.into_iter().filter(|o| o.strategy_id == strategy_id) {
        if order.exchange_order_id.is_empty() {
            continue; // nothing was ever sent to the exchange for this one
        }
        let Some(client) = execution_clients.get(&order.exchange) else {
            tracing::error!(
                strategy_id,
                exchange = %order.exchange,
                client_order_id = %order.client_order_id,
                "dead-man's switch: no execution client configured for this order's exchange, cannot cancel"
            );
            continue;
        };

        match client.cancel_order(&order.exchange_order_id).await {
            Ok(CancelOrderOutcome::Canceled) => {
                tracing::warn!(
                    strategy_id,
                    client_order_id = %order.client_order_id,
                    exchange_order_id = %order.exchange_order_id,
                    "dead-man's switch: canceled a resting order for a dead strategy"
                );
                if let Err(e) = store.update_order_status(&order.client_order_id, &order.exchange_order_id, "CANCELED", "", now_ns()) {
                    tracing::error!(client_order_id = %order.client_order_id, error = %e, "failed to persist canceled status");
                }
            }
            Ok(CancelOrderOutcome::AlreadyClosed) => {
                tracing::info!(
                    strategy_id,
                    client_order_id = %order.client_order_id,
                    "dead-man's switch: order was already closed on the exchange, nothing to cancel"
                );
            }
            Ok(CancelOrderOutcome::KrakenRejected { messages }) => {
                tracing::error!(
                    strategy_id,
                    client_order_id = %order.client_order_id,
                    reason = %messages.join("; "),
                    "dead-man's switch: Kraken rejected the cancel request"
                );
            }
            Err(e) => {
                tracing::error!(strategy_id, client_order_id = %order.client_order_id, error = %e, "dead-man's switch: cancel request failed");
            }
        }
    }
}

/// Submits a market order for every symbol with a nonzero tracked
/// position, sized and sided to bring that position back to flat. Uses
/// `RiskEngine::position`, which reflects only fills this process has
/// actually seen since it started (see that method's own docs) — not a
/// substitute for checking the exchange's actual holdings, but the same
/// source of truth every other risk check in this codebase already
/// relies on. Goes through `OrderServiceImpl::submit_internal`, i.e. the
/// real risk-evaluation path — a flatten order can still be rejected
/// (e.g. it would somehow exceed a size limit), which is logged, not
/// silently overridden.
async fn flatten_all_positions(config: &Config, order_service: &OrderServiceImpl, risk: &RiskEngine) {
    for symbol in &config.symbols {
        let position = risk.position(&symbol.symbol).await;
        if position == Decimal::ZERO {
            continue;
        }
        let side = if position > Decimal::ZERO { OrderSide::Sell } else { OrderSide::Buy };
        let quantity = position.abs();

        let order = OrderRequest {
            client_order_id: format!("{WATCHDOG_STRATEGY_ID}-{}-{}", symbol.symbol, now_ns()),
            symbol: symbol.symbol.clone(),
            exchange: symbol.exchange.clone(),
            side: side as i32,
            r#type: OrderType::Market as i32,
            quantity: Some(PbDecimal { value: quantity.to_string() }),
            limit_price: None,
            strategy_id: WATCHDOG_STRATEGY_ID.to_string(),
        };

        tracing::warn!(
            symbol = %symbol.symbol,
            %position,
            side = ?side,
            "dead-man's switch: submitting a flatten order for a dead strategy's open position"
        );
        match order_service.submit_internal(order).await {
            Ok(update) => {
                tracing::info!(symbol = %symbol.symbol, status = update.status, "dead-man's switch: flatten order submitted");
            }
            Err(status) => {
                tracing::error!(symbol = %symbol.symbol, error = %status, "dead-man's switch: flatten order failed to submit");
            }
        }
    }
}

/// Runs forever, polling for strategies that have gone silent past their
/// configured heartbeat timeout and reacting per `DeadManSwitchConfig`.
/// Spawned once from `main.rs`; a no-op loop (never fires) if
/// `dead_man_switch.enabled` is false.
pub async fn run_watchdog(
    monitor: Arc<HeartbeatMonitor>,
    config: Arc<Config>,
    order_service: OrderServiceImpl,
    execution_clients: Arc<HashMap<String, KrakenRestClient>>,
    store: Option<Arc<Store>>,
    risk: Arc<RiskEngine>,
    alert_sink: AlertSink,
) {
    let dms = &config.dead_man_switch;
    if !dms.enabled {
        tracing::info!("dead-man's switch disabled by config — watchdog task not active");
        return;
    }

    let timeout = Duration::from_secs(dms.heartbeat_timeout_secs);
    let check_interval = Duration::from_secs(dms.check_interval_secs.max(1));
    tracing::info!(
        heartbeat_timeout_secs = dms.heartbeat_timeout_secs,
        check_interval_secs = dms.check_interval_secs,
        auto_cancel_orders = dms.auto_cancel_orders,
        auto_flatten_positions = dms.auto_flatten_positions,
        "dead-man's switch watchdog started"
    );

    loop {
        tokio::time::sleep(check_interval).await;

        for strategy_id in monitor.stale_strategies(timeout) {
            tracing::error!(
                strategy_id,
                timeout_secs = dms.heartbeat_timeout_secs,
                "dead-man's switch: no heartbeat received within the configured timeout, treating strategy as dead"
            );
            alert_sink
                .send(&dead_man_switch_message(
                    &strategy_id,
                    dms.heartbeat_timeout_secs,
                    dms.auto_cancel_orders,
                    dms.auto_flatten_positions,
                ))
                .await;

            if dms.auto_cancel_orders {
                cancel_open_orders_for_strategy(&strategy_id, &store, &execution_clients).await;
            }
            if dms.auto_flatten_positions {
                flatten_all_positions(&config, &order_service, &risk).await;
            }

            // Don't re-trip on the same silence every check_interval —
            // a fresh heartbeat (if the strategy comes back) re-adds it.
            monitor.clear(&strategy_id);
        }
    }
}

fn now_ns() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strategy_that_never_heartbeat_is_never_stale() {
        let monitor = HeartbeatMonitor::new();
        assert!(monitor.stale_strategies(Duration::from_millis(0)).is_empty());
    }

    #[test]
    fn a_freshly_recorded_heartbeat_is_not_stale() {
        let monitor = HeartbeatMonitor::new();
        monitor.record("strat-1");
        assert!(monitor.stale_strategies(Duration::from_secs(60)).is_empty());
    }

    #[test]
    fn a_heartbeat_older_than_the_timeout_is_stale() {
        let monitor = HeartbeatMonitor::new();
        monitor.record("strat-1");
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(monitor.stale_strategies(Duration::from_millis(10)), vec!["strat-1".to_string()]);
    }

    #[test]
    fn only_the_stale_strategy_is_reported_among_several() {
        let monitor = HeartbeatMonitor::new();
        monitor.record("stale-one");
        std::thread::sleep(Duration::from_millis(20));
        monitor.record("fresh-one");
        let stale = monitor.stale_strategies(Duration::from_millis(10));
        assert_eq!(stale, vec!["stale-one".to_string()]);
    }

    #[test]
    fn a_fresh_heartbeat_resets_staleness() {
        let monitor = HeartbeatMonitor::new();
        monitor.record("strat-1");
        std::thread::sleep(Duration::from_millis(20));
        assert!(!monitor.stale_strategies(Duration::from_millis(10)).is_empty());

        monitor.record("strat-1");
        assert!(monitor.stale_strategies(Duration::from_millis(10)).is_empty());
    }

    #[test]
    fn clear_removes_a_strategy_from_tracking() {
        let monitor = HeartbeatMonitor::new();
        monitor.record("strat-1");
        std::thread::sleep(Duration::from_millis(20));
        monitor.clear("strat-1");
        assert!(monitor.stale_strategies(Duration::from_millis(10)).is_empty());
    }

    #[test]
    fn dead_man_switch_message_includes_strategy_and_settings() {
        let msg = dead_man_switch_message("strat-1", 30, true, false);
        assert!(msg.contains("strat-1"));
        assert!(msg.contains("30"));
        assert!(msg.contains("auto_cancel_orders=true"));
        assert!(msg.contains("auto_flatten_positions=false"));
    }
}
