//! Real observability: institutional audit Phase 2.4.
//!
//! Two things, deliberately built on infrastructure this codebase already
//! has rather than standing up a new dashboarding stack:
//!
//! 1. Alerting (via the same `AlertSink` Phase 1.5 wired up for
//!    reconciliation) when the fraction of orders rejected over a rolling
//!    window gets too high. An elevated rejection rate is a leading
//!    indicator that something upstream has gone wrong — a strategy's
//!    sizing drifted out of the configured limits, a config got stale, a
//!    guardrail is now too tight for real market conditions — often well
//!    before it shows up as a PnL problem.
//! 2. A periodic structured log line summarizing realized PnL and total
//!    exposure (`persistence::Store::realized_pnl_since` /
//!    `risk::RiskEngine::total_exposure_usd`), so an operator can feed
//!    this process's existing structured logs into whatever log-based
//!    dashboard (Grafana Loki, CloudWatch Logs Insights, etc.) they
//!    already run, rather than this project inventing its own dashboard.
//!    "Real" observability here means real numbers an operator can
//!    actually act on, not a promise of a UI this codebase doesn't have
//!    the infrastructure to host yet.
//! 3. A periodic Sharpe/Sortino/Calmar/max-drawdown summary
//!    (`performance.rs`, institutional audit Phase 3.4) computed from the
//!    same `fills` history the PnL summary above reads — judges the
//!    live risk engine's own realized returns the same way
//!    scripts/train_model.py's offline evaluation judges a candidate
//!    model, without needing to export the fills table and run that
//!    script by hand.
//!
//! Structurally this mirrors heartbeat.rs's watchdog and stop_loss.rs's
//! monitor: a small polling loop, spawned once from `main.rs`, sharing
//! the same `AlertSink` and `RiskEngine` those already use.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::alerting::AlertSink;
use crate::config::Config;
use crate::performance;
use crate::persistence::Store;
use crate::risk::RiskEngine;

fn now_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

/// Debounces the rejection-rate alert so it fires once per sustained
/// breach rather than once per `check_interval_secs` tick for as long as
/// the rate stays elevated. Set once an alert fires, cleared once a later
/// check finds the rate back under the threshold — a later breach after
/// that is treated as a new event and alerts again.
#[derive(Default)]
pub struct ObservabilityState {
    rejection_alert_active: AtomicBool,
}

impl ObservabilityState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Builds the alert text for an elevated order-rejection rate. Standalone
/// so its wording is testable without a database or the monitor loop,
/// same reasoning as `alerting::possible_missed_fills_message` and
/// `stop_loss::stop_loss_message`.
pub fn elevated_rejection_rate_message(rejected: u64, total: u64, rate: f64, max_rate: f64, window_secs: u64) -> String {
    format!(
        ":warning: Order rejection rate elevated: {rejected}/{total} orders ({:.1}%) rejected over the last \
         {window_secs}s, above the configured max of {:.1}%. Check recent reject_reason values in the `orders` \
         table for what's actually being rejected. See observability.rs's docs.",
        rate * 100.0,
        max_rate * 100.0,
    )
}

/// Runs forever, periodically checking the rolling order-rejection rate
/// (alerting once per sustained breach) and logging a PnL/exposure
/// summary. Spawned once from `main.rs`; a no-op loop (never fires) if
/// `observability.enabled` is false or no persistence store is configured
/// (there's nothing to query stats from without one — same optional-
/// infrastructure posture as every other store-dependent feature here).
pub async fn run_observability_monitor(
    state: Arc<ObservabilityState>,
    config: Arc<Config>,
    store: Option<Arc<Store>>,
    risk: Arc<RiskEngine>,
    alert_sink: AlertSink,
) {
    let obs = &config.observability;
    if !obs.enabled {
        tracing::info!("observability monitor disabled by config — task not active");
        return;
    }
    let Some(store) = store else {
        tracing::warn!(
            "observability monitor: no persistence store configured, cannot compute order/PnL stats — task not active"
        );
        return;
    };

    let check_interval = Duration::from_secs(obs.check_interval_secs.max(1));
    let window_ns = Duration::from_secs(obs.window_secs.max(1)).as_nanos() as i64;
    let performance_window_ns = Duration::from_secs(obs.performance_window_days.max(1) * 86_400).as_nanos() as i64;
    tracing::info!(
        check_interval_secs = obs.check_interval_secs,
        window_secs = obs.window_secs,
        min_sample_size = obs.min_sample_size,
        max_rejection_rate = obs.max_rejection_rate,
        performance_window_days = obs.performance_window_days,
        "observability monitor started"
    );

    loop {
        tokio::time::sleep(check_interval).await;
        let since_ns = now_ns() - window_ns;

        match store.order_stats_since(since_ns) {
            Ok(stats) => {
                let rate = stats.rejection_rate();
                let breached = stats.total >= obs.min_sample_size && rate > obs.max_rejection_rate;
                if breached {
                    if !state.rejection_alert_active.swap(true, Ordering::SeqCst) {
                        tracing::error!(
                            rejected = stats.rejected,
                            total = stats.total,
                            rate,
                            max_rate = obs.max_rejection_rate,
                            "observability: elevated order-rejection rate"
                        );
                        alert_sink
                            .send(&elevated_rejection_rate_message(
                                stats.rejected,
                                stats.total,
                                rate,
                                obs.max_rejection_rate,
                                obs.window_secs,
                            ))
                            .await;
                    }
                } else {
                    state.rejection_alert_active.store(false, Ordering::SeqCst);
                }
            }
            Err(e) => tracing::error!(error = %e, "observability: failed to query order stats"),
        }

        match store.realized_pnl_since(since_ns) {
            Ok(realized_pnl_usd) => {
                let total_exposure_usd = risk.total_exposure_usd().await;
                tracing::info!(
                    window_secs = obs.window_secs,
                    %realized_pnl_usd,
                    %total_exposure_usd,
                    "observability: PnL/exposure summary"
                );
            }
            Err(e) => tracing::error!(error = %e, "observability: failed to query realized PnL"),
        }

        let performance_since_ns = now_ns() - performance_window_ns;
        match store.fills_since(performance_since_ns) {
            Ok(fills) => {
                let metrics = performance::compute_metrics(&fills, now_ns());
                tracing::info!(
                    performance_window_days = obs.performance_window_days,
                    days_of_history = metrics.days,
                    sharpe = ?metrics.sharpe,
                    sortino = ?metrics.sortino,
                    calmar = ?metrics.calmar,
                    max_drawdown_usd = metrics.max_drawdown_usd,
                    "observability: performance summary (Sharpe/Sortino/Calmar/max-drawdown)"
                );
            }
            Err(e) => tracing::error!(error = %e, "observability: failed to query fills for performance summary"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevated_rejection_rate_message_includes_counts_and_percentages() {
        let msg = elevated_rejection_rate_message(6, 20, 0.30, 0.25, 900);
        assert!(msg.contains("6/20"));
        assert!(msg.contains("30.0%"));
        assert!(msg.contains("25.0%"));
        assert!(msg.contains("900s"));
    }

    #[test]
    fn a_fresh_state_has_no_active_alert() {
        let state = ObservabilityState::new();
        assert!(!state.rejection_alert_active.load(Ordering::SeqCst));
    }

    #[test]
    fn the_alert_flag_can_be_set_and_cleared() {
        let state = ObservabilityState::new();
        assert!(!state.rejection_alert_active.swap(true, Ordering::SeqCst));
        assert!(state.rejection_alert_active.load(Ordering::SeqCst));
        state.rejection_alert_active.store(false, Ordering::SeqCst);
        assert!(!state.rejection_alert_active.load(Ordering::SeqCst));
    }
}
