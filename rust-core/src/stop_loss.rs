//! Per-position stop-loss / auto-reduce: institutional audit Phase 2.3.
//!
//! This is deliberately a third, independent line of defense alongside
//! two that already exist elsewhere in this codebase, each of which
//! covers a different failure mode:
//! - The daily kill switch (risk.rs) reacts to *realized* PnL — it only
//!   notices a loss once a closing fill actually books it, so a large
//!   open position that has moved hard against a strategy but hasn't
//!   been closed yet is invisible to it.
//! - The dead-man's switch (heartbeat.rs) reacts to strategy *liveness* —
//!   a strategy that is very much alive and heartbeating normally, but
//!   simply riding a bad trade, never trips it.
//!
//! This module watches each open position's live *unrealized* PnL
//! (risk.rs's `unrealized_pnl_pct`, marked against the current book) and,
//! once a position's loss crosses `stop_loss.max_loss_pct`, submits a
//! reduce-to-flat market order for that symbol alone — independent of
//! whether the owning strategy is alive, silent, or has already been
//! flattened by the dead-man's switch for an unrelated reason.
//!
//! Structurally this mirrors heartbeat.rs's watchdog: a pure-bookkeeping
//! piece (`StopLossState`, just a debounce set) plus a polling loop
//! (`run_stop_loss_monitor`) that submits its flatten order through
//! `OrderServiceImpl::submit_internal` — the same real risk-evaluation
//! path every other order takes, including heartbeat.rs's own flatten
//! orders. A reduce-to-exactly-flat order always passes the volatility
//! breaker's reduce-only check (see risk.rs's `check_volatility_breaker`
//! docs), so a stop-loss flatten fires even while that breaker is
//! tripped — which is exactly when it's most likely to be needed.

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rust_decimal::Decimal;
use tokio::sync::Mutex;

use crate::alerting::AlertSink;
use crate::config::{Config, SymbolConfig};
use crate::order::OrderServiceImpl;
use crate::proto::pb::{Decimal as PbDecimal, OrderRequest, OrderSide, OrderType};
use crate::risk::RiskEngine;

/// The strategy_id attached to a synthetic stop-loss flatten order — never
/// something a real Python strategy would send — so it's obvious in
/// logs/persistence which orders came from the stop-loss monitor rather
/// than a live strategy decision. Distinct from heartbeat.rs's
/// `WATCHDOG_STRATEGY_ID` so the two auto-reduce sources stay
/// distinguishable in the fills audit trail.
const STOP_LOSS_STRATEGY_ID: &str = "stop-loss";

/// Tracks, per symbol, that a stop-loss flatten order has already been
/// submitted for the *current* breach, so the monitor doesn't resubmit
/// another flatten order on every `check_interval_secs` tick while the
/// first one's fill is still in flight. Cleared once the symbol's
/// position returns to flat, so a fresh position that later breaches its
/// own stop again is treated as a new event, not a duplicate of the last
/// one.
#[derive(Default)]
pub struct StopLossState {
    triggered: Mutex<HashSet<String>>,
}

impl StopLossState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Builds the alert text for a position the monitor just decided to
/// flatten. Standalone so its wording is testable without spinning up the
/// whole monitor loop, same reasoning as `heartbeat::dead_man_switch_message`.
pub fn stop_loss_message(symbol: &str, unrealized_pnl_pct: Decimal, max_loss_pct: Decimal) -> String {
    format!(
        ":rotating_light: Stop-loss triggered for `{symbol}`: unrealized PnL {unrealized_pnl_pct:.4} \
         (≈{:.2}%) has breached the configured max loss of {max_loss_pct:.4} (≈{:.2}%) — submitting a \
         reduce-to-flat order. See stop_loss.rs's docs.",
        unrealized_pnl_pct * Decimal::from(100),
        max_loss_pct * Decimal::from(100),
    )
}

fn now_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

/// Submits a single reduce-to-flat market order for `symbol_cfg`, sized
/// and sided to bring `position` back to exactly zero. Same
/// submit_internal path (real risk evaluation, not a bypass) that
/// heartbeat.rs's `flatten_all_positions` uses for the same reason:
/// a flatten order can still be rejected by an unrelated guardrail, which
/// is logged rather than silently overridden.
async fn flatten_one_symbol(order_service: &OrderServiceImpl, symbol_cfg: &SymbolConfig, position: Decimal) {
    let side = if position > Decimal::ZERO { OrderSide::Sell } else { OrderSide::Buy };
    let quantity = position.abs();

    let order = OrderRequest {
        client_order_id: format!("{STOP_LOSS_STRATEGY_ID}-{}-{}", symbol_cfg.symbol, now_ns()),
        symbol: symbol_cfg.symbol.clone(),
        exchange: symbol_cfg.exchange.clone(),
        side: side as i32,
        r#type: OrderType::Market as i32,
        quantity: Some(PbDecimal { value: quantity.to_string() }),
        limit_price: None,
        strategy_id: STOP_LOSS_STRATEGY_ID.to_string(),
        // A stop-loss flatten needs to actually execute now, not rest —
        // post-only is meaningless (and would be ignored/rejected by
        // Kraken) on a MARKET order anyway. Same reasoning as
        // heartbeat.rs's flatten orders.
        post_only: false,
    };

    tracing::warn!(
        symbol = %symbol_cfg.symbol,
        %position,
        side = ?side,
        "stop-loss: submitting a reduce-to-flat order for a breached position"
    );
    match order_service.submit_internal(order).await {
        Ok(update) => {
            tracing::info!(symbol = %symbol_cfg.symbol, status = update.status, "stop-loss: flatten order submitted");
        }
        Err(status) => {
            tracing::error!(symbol = %symbol_cfg.symbol, error = %status, "stop-loss: flatten order failed to submit");
        }
    }
}

/// Runs forever, polling every configured symbol's unrealized PnL and
/// flattening any position whose loss has breached `stop_loss.max_loss_pct`.
/// Spawned once from `main.rs`; a no-op loop (never fires) if
/// `stop_loss.enabled` is false or `max_loss_pct` fails to parse.
pub async fn run_stop_loss_monitor(
    state: Arc<StopLossState>,
    config: Arc<Config>,
    order_service: OrderServiceImpl,
    risk: Arc<RiskEngine>,
    alert_sink: AlertSink,
) {
    let sl = &config.stop_loss;
    if !sl.enabled {
        tracing::info!("stop-loss monitor disabled by config — task not active");
        return;
    }
    let Ok(max_loss_pct) = Decimal::from_str(&sl.max_loss_pct) else {
        tracing::error!(max_loss_pct = %sl.max_loss_pct, "stop-loss: invalid max_loss_pct in config, monitor not active");
        return;
    };
    let check_interval = Duration::from_secs(sl.check_interval_secs.max(1));
    tracing::info!(
        max_loss_pct = %sl.max_loss_pct,
        check_interval_secs = sl.check_interval_secs,
        "stop-loss monitor started"
    );

    loop {
        tokio::time::sleep(check_interval).await;

        for symbol_cfg in &config.symbols {
            let symbol = &symbol_cfg.symbol;
            let position = risk.position(symbol).await;

            if position.is_zero() {
                // Flat again (closed normally, or a previous stop-loss/
                // dead-man's-switch flatten already landed) — clear any
                // stale debounce entry so a future breach on a fresh
                // position is treated as new.
                state.triggered.lock().await.remove(symbol);
                continue;
            }

            let Some(pnl_pct) = risk.unrealized_pnl_pct(symbol).await else {
                continue; // no live book to mark against yet
            };
            if pnl_pct > -max_loss_pct {
                continue; // within tolerance (or in profit)
            }

            {
                let mut triggered = state.triggered.lock().await;
                if !triggered.insert(symbol.clone()) {
                    continue; // already acted on this breach, awaiting the flatten fill
                }
            }

            tracing::error!(
                symbol,
                %pnl_pct,
                %max_loss_pct,
                "stop-loss: unrealized loss has breached the configured max, flattening position"
            );
            alert_sink.send(&stop_loss_message(symbol, pnl_pct, -max_loss_pct)).await;
            flatten_one_symbol(&order_service, symbol_cfg, position).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_loss_message_includes_symbol_and_both_percentages() {
        let msg = stop_loss_message("BTC-USD", Decimal::from_str("-0.045").unwrap(), Decimal::from_str("-0.03").unwrap());
        assert!(msg.contains("BTC-USD"));
        assert!(msg.contains("-0.0450"));
        assert!(msg.contains("-4.50"));
        assert!(msg.contains("-0.0300"));
        assert!(msg.contains("-3.00"));
    }

    #[tokio::test]
    async fn a_symbol_with_no_prior_breach_can_be_marked_triggered() {
        let state = StopLossState::new();
        assert!(!state.triggered.lock().await.contains("BTC-USD"));
        state.triggered.lock().await.insert("BTC-USD".to_string());
        assert!(state.triggered.lock().await.contains("BTC-USD"));
    }

    #[tokio::test]
    async fn clearing_a_symbol_allows_it_to_be_re_triggered() {
        let state = StopLossState::new();
        state.triggered.lock().await.insert("BTC-USD".to_string());
        state.triggered.lock().await.remove("BTC-USD");
        assert!(!state.triggered.lock().await.contains("BTC-USD"));
    }
}
