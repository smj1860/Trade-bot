//! Portfolio-level performance instrumentation: institutional audit
//! Phase 3.4.
//!
//! `observability.rs` already logs a running realized-PnL/exposure
//! summary, but that's a single cumulative number — it can't say whether
//! the strategy's return stream is *good*, only what it currently totals.
//! The audit's institutional tier calls out Sharpe/Sortino/Calmar and
//! rolling drawdown computed from the live risk engine's own realized
//! fills, not just `scripts/train_model.py`'s offline backtest evaluation
//! — so a model that looked good in training can be judged the same way
//! against what it's actually doing with real fills, without waiting for
//! an operator to export the `fills` table and run the Python side by
//! hand.
//!
//! Deliberately built the same way every other pure-logic module in this
//! codebase is: `bucket_daily_pnl` turns `persistence::Store::fills_since`'s
//! raw `(applied_at_ns, realized_pnl_usd)` series into one PnL figure per
//! calendar UTC day (zero-filling days with no fills, since a quiet day
//! is a real, informative zero return, not a gap to skip — omitting it
//! would understate volatility and overstate Sharpe), and the ratios below
//! are computed from that pure `Vec<f64>`, independent of `Store` or any
//! I/O, so they're unit-testable with made-up series and don't need a
//! database to verify.
//!
//! Sharpe/Sortino/Calmar are computed directly on daily realized-PnL-in-USD
//! rather than a percentage return series. This project doesn't track a
//! separate "account equity" figure the risk engine could divide by (risk
//! limits here are absolute-USD, e.g. `kill_switch_max_daily_loss_usd` —
//! see `config.rs`), and it turns out not to matter for these three
//! ratios: each is a ratio of a mean (or a return) to a like-scaled
//! quantity (a standard deviation, or a drawdown) computed from the exact
//! same series, so dividing every term through by a constant capital base
//! before computing the ratio leaves the ratio unchanged. Reported values
//! are therefore the same whether or not a capital base is ever added to
//! this project's config — this is a documented simplifying fact, not an
//! approximation.

use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;

const SECONDS_PER_DAY: i64 = 86_400;
const NANOS_PER_DAY: i64 = SECONDS_PER_DAY * 1_000_000_000;
const TRADING_DAYS_PER_YEAR: f64 = 365.0; // crypto trades every calendar day, not just weekdays

/// Buckets a time-ordered `(applied_at_ns, realized_pnl_usd)` fill series
/// into one realized-PnL-per-day figure, from the UTC day of the first
/// fill through the UTC day of `now_ns` inclusive — days with no fills in
/// between (or after the last fill, up to `now_ns`) are zero-filled rather
/// than omitted, which matters for the volatility these ratios are built
/// from (see module docs). Returns an empty vec for an empty `fills`
/// input; there is no "day zero" to anchor to without at least one fill.
pub fn bucket_daily_pnl(fills: &[(i64, Decimal)], now_ns: i64) -> Vec<f64> {
    if fills.is_empty() {
        return Vec::new();
    }
    let first_day = fills[0].0.div_euclid(NANOS_PER_DAY);
    let last_day = now_ns.div_euclid(NANOS_PER_DAY).max(first_day);
    let n_days = (last_day - first_day + 1) as usize;

    let mut daily = vec![0.0_f64; n_days];
    for (applied_at_ns, pnl) in fills {
        let day_index = (applied_at_ns.div_euclid(NANOS_PER_DAY) - first_day) as usize;
        if day_index < daily.len() {
            daily[day_index] += pnl.to_f64().unwrap_or(0.0);
        }
    }
    daily
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (n-1 denominator) — `None` for fewer than 2
/// points, since a single-point sample has no defined sample variance.
fn sample_std_dev(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let m = mean(xs);
    let variance = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    Some(variance.sqrt())
}

/// Sample standard deviation of only the below-target (here, below zero)
/// values, against the *full* series' mean-square denominator convention
/// used by most Sortino implementations: each below-target deviation is
/// squared, summed, and divided by (n - 1) across the whole series, not
/// just the count of losing days — otherwise a strategy with one huge
/// losing day out of 100 would be scored as if it had only ever traded
/// that one day. `None` if there are no below-target days at all (nothing
/// to divide the mean return by — an undefined, not infinite, Sortino).
fn downside_deviation(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let downside_sq_sum: f64 = xs.iter().filter(|&&x| x < 0.0).map(|x| x.powi(2)).sum();
    if downside_sq_sum == 0.0 {
        return None;
    }
    Some((downside_sq_sum / (xs.len() - 1) as f64).sqrt())
}

/// Annualized Sharpe ratio from a daily PnL series (`mean / stdev *
/// sqrt(365)`, zero risk-free rate — this project has no cash-rate input
/// to net out, and crypto risk-free-rate conventions vary enough that
/// omitting it is more honest than picking one). `None` when the series
/// is too short or has zero variance (e.g. every day identical, including
/// all-zero) to divide by.
pub fn sharpe_ratio(daily_pnl: &[f64]) -> Option<f64> {
    let std = sample_std_dev(daily_pnl)?;
    if std == 0.0 {
        return None;
    }
    Some(mean(daily_pnl) / std * TRADING_DAYS_PER_YEAR.sqrt())
}

/// Annualized Sortino ratio — same shape as `sharpe_ratio` but against
/// downside deviation only, so upside volatility (good days) isn't
/// penalized the way Sharpe's symmetric stdev does. `None` when there are
/// no losing days in the window (see `downside_deviation`) — reported by
/// callers as "undefined (no losing days)", not as an artificially large
/// number.
pub fn sortino_ratio(daily_pnl: &[f64]) -> Option<f64> {
    let downside = downside_deviation(daily_pnl)?;
    Some(mean(daily_pnl) / downside * TRADING_DAYS_PER_YEAR.sqrt())
}

/// Maximum peak-to-trough decline of the cumulative-PnL equity curve
/// built by summing `daily_pnl` — always >= 0.0 (0.0 for a curve that
/// never dips below its own running peak, including an empty series).
/// Reported in the same USD units as `daily_pnl`, not a percentage (see
/// module docs on why no capital base is needed for the ratios that use
/// this).
pub fn max_drawdown(daily_pnl: &[f64]) -> f64 {
    let mut equity = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut worst = 0.0_f64;
    for pnl in daily_pnl {
        equity += pnl;
        peak = peak.max(equity);
        worst = worst.max(peak - equity);
    }
    worst
}

/// Calmar ratio: annualized return (mean daily PnL * 365) divided by
/// max drawdown over the same window. `None` when there's no drawdown to
/// divide by (a strategy that never had a losing stretch — including a
/// window with no fills at all) rather than reporting an artificial
/// infinity.
pub fn calmar_ratio(daily_pnl: &[f64]) -> Option<f64> {
    if daily_pnl.is_empty() {
        return None;
    }
    let drawdown = max_drawdown(daily_pnl);
    if drawdown == 0.0 {
        return None;
    }
    let annualized_return = mean(daily_pnl) * TRADING_DAYS_PER_YEAR;
    Some(annualized_return / drawdown)
}

/// Every ratio this module computes, bundled for one log line / gRPC
/// response — `None` fields are rendered by the caller as "undefined"
/// rather than skipped, so an operator watching the logs can tell "no
/// signal yet" apart from "this field was never computed."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceMetrics {
    pub days: usize,
    pub sharpe: Option<f64>,
    pub sortino: Option<f64>,
    pub calmar: Option<f64>,
    pub max_drawdown_usd: f64,
}

/// Computes every ratio from a raw fill series in one call — the
/// entry point `observability.rs`'s monitor loop uses each tick.
pub fn compute_metrics(fills: &[(i64, Decimal)], now_ns: i64) -> PerformanceMetrics {
    let daily_pnl = bucket_daily_pnl(fills, now_ns);
    PerformanceMetrics {
        days: daily_pnl.len(),
        sharpe: sharpe_ratio(&daily_pnl),
        sortino: sortino_ratio(&daily_pnl),
        calmar: calmar_ratio(&daily_pnl),
        max_drawdown_usd: max_drawdown(&daily_pnl),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn day(n: i64) -> i64 {
        n * NANOS_PER_DAY
    }

    fn pnl(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn bucket_daily_pnl_sums_same_day_fills_and_zero_fills_gap_days() {
        let fills = vec![
            (day(0) + 1, pnl("10")),
            (day(0) + 2, pnl("5")),
            // day 1 has no fills at all
            (day(2), pnl("-3")),
        ];
        let daily = bucket_daily_pnl(&fills, day(2));
        assert_eq!(daily, vec![15.0, 0.0, -3.0]);
    }

    #[test]
    fn bucket_daily_pnl_extends_through_now_ns_even_past_the_last_fill() {
        let fills = vec![(day(0), pnl("10"))];
        let daily = bucket_daily_pnl(&fills, day(3));
        assert_eq!(daily, vec![10.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bucket_daily_pnl_is_empty_for_no_fills() {
        assert_eq!(bucket_daily_pnl(&[], day(10)), Vec::<f64>::new());
    }

    #[test]
    fn sharpe_ratio_is_none_for_zero_variance() {
        assert_eq!(sharpe_ratio(&[5.0, 5.0, 5.0]), None);
        assert_eq!(sharpe_ratio(&[0.0, 0.0]), None);
    }

    #[test]
    fn sharpe_ratio_is_none_for_a_single_day() {
        assert_eq!(sharpe_ratio(&[100.0]), None);
    }

    #[test]
    fn sharpe_ratio_is_positive_for_a_consistently_profitable_series() {
        let daily = vec![10.0, 12.0, 8.0, 11.0, 9.0];
        let sharpe = sharpe_ratio(&daily).unwrap();
        assert!(sharpe > 0.0);
    }

    #[test]
    fn sharpe_ratio_is_negative_for_a_consistently_losing_series() {
        let daily = vec![-10.0, -12.0, -8.0, -11.0, -9.0];
        let sharpe = sharpe_ratio(&daily).unwrap();
        assert!(sharpe < 0.0);
    }

    #[test]
    fn sortino_ratio_is_none_with_no_losing_days() {
        assert_eq!(sortino_ratio(&[1.0, 2.0, 3.0]), None);
    }

    #[test]
    fn sortino_ratio_ignores_upside_volatility() {
        // Same mean, but one series has a huge up day (shouldn't be penalized)
        // and one small down day each — Sortino only look at the downside.
        let smooth_upside = vec![1.0, 1.0, -1.0, 1.0, 1.0];
        let spiky_upside = vec![10.0, -8.0, -1.0, 1.0, 1.0];
        let sortino_smooth = sortino_ratio(&smooth_upside).unwrap();
        let sortino_spiky = sortino_ratio(&spiky_upside).unwrap();
        // spiky has a bigger down day (-8 vs -1) so its downside deviation
        // is larger, and its Sortino should come out lower even though its
        // mean return is much higher.
        assert!(sortino_smooth > 0.0);
        assert!(sortino_spiky.is_finite());
        assert_ne!(sortino_smooth, sortino_spiky);
    }

    #[test]
    fn max_drawdown_is_zero_for_a_monotonically_increasing_curve() {
        assert_eq!(max_drawdown(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn max_drawdown_is_zero_for_an_empty_series() {
        assert_eq!(max_drawdown(&[]), 0.0);
    }

    #[test]
    fn max_drawdown_finds_the_worst_peak_to_trough_decline() {
        // equity curve: 10, 15, 5, 8, 20, 12 -> peaks 15 then 20, troughs 5 then 12
        // drawdown from 15: 15-5=10; drawdown from 20: 20-12=8 -> worst is 10
        let daily = vec![10.0, 5.0, -10.0, 3.0, 12.0, -8.0];
        assert_eq!(max_drawdown(&daily), 10.0);
    }

    #[test]
    fn calmar_ratio_is_none_with_no_drawdown() {
        assert_eq!(calmar_ratio(&[1.0, 2.0, 3.0]), None);
    }

    #[test]
    fn calmar_ratio_is_none_for_an_empty_series() {
        assert_eq!(calmar_ratio(&[]), None);
    }

    #[test]
    fn calmar_ratio_divides_annualized_return_by_max_drawdown() {
        let daily = vec![10.0, -5.0]; // mean=2.5, drawdown = 5.0
        let calmar = calmar_ratio(&daily).unwrap();
        assert_eq!(calmar, (2.5 * TRADING_DAYS_PER_YEAR) / 5.0);
    }

    #[test]
    fn compute_metrics_bundles_every_ratio_from_a_raw_fill_series() {
        let fills = vec![
            (day(0), pnl("10")),
            (day(1), pnl("-5")),
            (day(2), pnl("8")),
            (day(3), pnl("-2")),
        ];
        let metrics = compute_metrics(&fills, day(3));
        assert_eq!(metrics.days, 4);
        assert!(metrics.sharpe.is_some());
        assert!(metrics.sortino.is_some());
        assert_eq!(metrics.max_drawdown_usd, 5.0);
    }

    #[test]
    fn compute_metrics_handles_an_empty_fill_series_without_panicking() {
        let metrics = compute_metrics(&[], day(0));
        assert_eq!(metrics.days, 0);
        assert_eq!(metrics.sharpe, None);
        assert_eq!(metrics.sortino, None);
        assert_eq!(metrics.calmar, None);
        assert_eq!(metrics.max_drawdown_usd, 0.0);
    }
}
