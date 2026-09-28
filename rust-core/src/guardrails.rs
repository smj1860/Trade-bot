//! Pre-trade circuit breakers and slippage/spread guardrails — a layer of
//! defense between an order that has already passed risk.rs's plain
//! size/notional/position caps and the exchange, aimed specifically at
//! failure modes those caps don't address at all: a liquidity vacuum that
//! makes the *actual* fill price much worse than the top-of-book price a
//! naive check would use, a spread that has blown out well past its own
//! recent normal, or a sudden volatility spike consistent with a flash
//! crash or an exchange disruption.
//!
//! Every function here is a pure computation over data the caller already
//! has (order book levels, a book's own rolling `PriceSample` history —
//! see orderbook.rs) rather than a stateful tracker of its own: the book's
//! history buffer is the only piece of state this needs, and it's already
//! fed continuously by ordinary market-data ingestion (every
//! `apply_snapshot`/`apply_update` call), so there's no separate wiring
//! required to keep it warm. `risk.rs` is the caller that threads these
//! into `RiskEngine::evaluate`, including the volatility breaker's
//! freeze/cooldown state (which *does* need to persist across calls, and
//! lives in `RiskEngine` for that reason, not here).

use std::time::{Duration, Instant};

use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;

use crate::orderbook::PriceSample;

/// Walks `levels` (best price first — exactly what
/// `OrderBook::bid_levels`/`ask_levels` already return) consuming `qty`,
/// and returns the size-weighted average fill price a market order of
/// this size would actually get. This is deliberately not just the
/// top-of-book price: an order larger than the best level's own quantity
/// pays a worse blended price for the rest, and that's exactly the
/// liquidity-vacuum cost this guardrail exists to catch.
///
/// Returns `None` if the given levels don't carry enough total quantity
/// to fill `qty` at all — the caller should treat that as "the book is
/// too thin to safely estimate this order's cost" (reject), not as "zero
/// samples, so no problem."
pub fn simulate_fill_price(levels: &[(Decimal, Decimal)], qty: Decimal) -> Option<Decimal> {
    if qty <= Decimal::ZERO {
        return None;
    }
    let mut remaining = qty;
    let mut notional = Decimal::ZERO;
    for &(price, level_qty) in levels {
        if remaining <= Decimal::ZERO {
            break;
        }
        let take = remaining.min(level_qty);
        notional += take * price;
        remaining -= take;
    }
    if remaining > Decimal::ZERO {
        return None;
    }
    Some(notional / qty)
}

/// `|fill_price - mid| / mid`, as an unsigned fraction — how much worse
/// than the current midpoint a simulated fill price is. Unsigned because
/// the guardrail cares about magnitude regardless of side: a buy's fill
/// price is expected to sit at/above mid, a sell's at/below, and either
/// one being far from mid is equally a sign of a thin/dislocated book.
pub fn slippage_fraction(mid: Decimal, fill_price: Decimal) -> Decimal {
    if mid <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    ((fill_price - mid) / mid).abs()
}

/// Current bid-ask spread as a fraction of the midpoint.
pub fn spread_fraction(best_bid: Decimal, best_ask: Decimal, mid: Decimal) -> Decimal {
    if mid <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    (best_ask - best_bid) / mid
}

/// Mean of a rolling window of past `spread_pct` samples — the baseline
/// the *current* spread gets compared against (see
/// `RiskEngine::check_spread_guard`, which multiplies this by a
/// configured factor for the actual dynamic threshold). Returns `None`
/// if there are fewer than `min_samples`: a baseline built from a
/// handful of ticks right after startup is noise, not signal, and the
/// caller should read "not enough history yet" as "the dynamic check
/// doesn't apply yet," never as "baseline is zero, so anything is too
/// wide."
pub fn rolling_average_spread(samples: &[PriceSample], min_samples: usize) -> Option<Decimal> {
    if samples.len() < min_samples {
        return None;
    }
    let sum: Decimal = samples.iter().map(|s| s.spread_pct).sum();
    Some(sum / Decimal::from(samples.len() as u64))
}

/// Parkinson's high-low range volatility estimate over a single window:
/// `sqrt(ln(high/low)^2 / (4 * ln 2))`, using this window's own highest
/// and lowest mid price — the same formula
/// `strategy/indicators.py::parkinson_vol` uses on the Python side, but
/// computed from live book mid-price ticks rather than real traded
/// high/low, since this crate has no OHLC bar aggregation of its own
/// (that lives in the Python strategy layer). A coarser proxy than a
/// true trade-derived Parkinson estimate, but scale-free the same way,
/// and it's only ever compared against its *own* recent history's
/// distribution (see `volatility_zscore`) rather than an absolute
/// threshold — so the approximation only needs to be internally
/// consistent with itself, not calibrated against real trade prints.
///
/// Computed in `f64`: this is a statistical threshold check, not a money
/// calculation, and `rust_decimal` has no `sqrt`/`ln` without pulling in
/// its `maths` feature for a single guardrail computation.
pub fn window_parkinson_vol(mid_prices: &[Decimal]) -> Option<f64> {
    if mid_prices.len() < 2 {
        return None;
    }
    let high = mid_prices.iter().copied().max()?.to_f64()?;
    let low = mid_prices.iter().copied().min()?.to_f64()?;
    if high <= 0.0 || low <= 0.0 {
        return None;
    }
    let ln_ratio = (high / low).ln();
    Some(((ln_ratio * ln_ratio) / (4.0 * std::f64::consts::LN_2)).sqrt())
}

/// A single reading from `assess_volatility`: the current short-window
/// vol, the baseline distribution it's being judged against, and the
/// resulting Z-score. Kept as a struct (rather than just a bare f64)
/// so `RiskEngine` can log the full picture on a trip, not just the
/// verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolatilityReading {
    pub short_window_vol: f64,
    pub baseline_mean: f64,
    pub baseline_stddev: f64,
    pub zscore: f64,
}

/// The minimum number of non-degenerate baseline buckets required before
/// the volatility breaker will render a verdict at all — same
/// not-enough-history-yet philosophy as `rolling_average_spread`'s
/// `min_samples`. Below this, `assess_volatility` returns `None` and the
/// caller must treat that as "can't judge yet," not "never trips."
const MIN_BASELINE_BUCKETS: usize = 5;

/// Buckets `samples` (`PriceSample`s already filtered to the baseline
/// window by the caller — see `OrderBook::recent_samples`) into
/// consecutive `bucket` -sized slices by age-from-`now`, computes each
/// bucket's `window_parkinson_vol`, and compares the *most recent*
/// `short_window`'s reading against the mean/stddev of every *older*
/// bucket's reading (the current short window is deliberately excluded
/// from its own baseline — including a live spike in the distribution
/// it's being compared against would blunt exactly the signal this
/// exists to catch).
///
/// Returns `None` if the current short window doesn't have at least 2
/// samples (can't compute a range at all) or fewer than
/// `MIN_BASELINE_BUCKETS` older buckets have at least 2 samples each
/// (not enough baseline history yet).
pub fn assess_volatility(
    samples: &[PriceSample],
    now: Instant,
    short_window: Duration,
    bucket: Duration,
) -> Option<VolatilityReading> {
    let short_prices: Vec<Decimal> = samples
        .iter()
        .filter(|s| now.duration_since(s.at) <= short_window)
        .map(|s| s.mid)
        .collect();
    let short_window_vol = window_parkinson_vol(&short_prices)?;

    let bucket_secs = bucket.as_secs_f64().max(1.0);
    let mut buckets: std::collections::HashMap<u64, Vec<Decimal>> = std::collections::HashMap::new();
    for s in samples {
        let age = now.duration_since(s.at);
        if age <= short_window {
            continue; // excluded from its own baseline, see docs above
        }
        let bucket_index = (age.as_secs_f64() / bucket_secs) as u64;
        buckets.entry(bucket_index).or_default().push(s.mid);
    }

    let baseline_readings: Vec<f64> =
        buckets.values().filter_map(|prices| window_parkinson_vol(prices)).collect();
    if baseline_readings.len() < MIN_BASELINE_BUCKETS {
        return None;
    }

    let n = baseline_readings.len() as f64;
    let mean = baseline_readings.iter().sum::<f64>() / n;
    let variance = baseline_readings.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    let zscore = if stddev > 0.0 { (short_window_vol - mean) / stddev } else { 0.0 };

    Some(VolatilityReading { short_window_vol, baseline_mean: mean, baseline_stddev: stddev, zscore })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn simulate_fill_price_within_top_level() {
        let levels = vec![(d("100"), d("2")), (d("101"), d("5"))];
        assert_eq!(simulate_fill_price(&levels, d("1")), Some(d("100")));
    }

    #[test]
    fn simulate_fill_price_blends_across_levels() {
        let levels = vec![(d("100"), d("1")), (d("102"), d("1"))];
        // 1 @ 100 + 1 @ 102, qty=2 -> vwap = 101
        assert_eq!(simulate_fill_price(&levels, d("2")), Some(d("101")));
    }

    #[test]
    fn simulate_fill_price_none_when_book_too_thin() {
        let levels = vec![(d("100"), d("1"))];
        assert_eq!(simulate_fill_price(&levels, d("5")), None);
    }

    #[test]
    fn simulate_fill_price_none_for_non_positive_qty() {
        let levels = vec![(d("100"), d("1"))];
        assert_eq!(simulate_fill_price(&levels, d("0")), None);
    }

    #[test]
    fn slippage_fraction_computes_magnitude_regardless_of_direction() {
        assert_eq!(slippage_fraction(d("100"), d("101")), d("0.01"));
        assert_eq!(slippage_fraction(d("100"), d("99")), d("0.01"));
    }

    #[test]
    fn spread_fraction_basic() {
        // (101 - 100) / 100.5, sanity check it's small and positive
        let f = spread_fraction(d("100"), d("101"), d("100.5"));
        assert!(f > Decimal::ZERO && f < d("0.02"));
    }

    fn sample(at: Instant, mid: &str, spread_pct: &str) -> PriceSample {
        PriceSample { at, mid: d(mid), spread_pct: d(spread_pct) }
    }

    #[test]
    fn rolling_average_spread_none_below_min_samples() {
        let now = Instant::now();
        let samples = vec![sample(now, "100", "0.01"), sample(now, "100", "0.02")];
        assert_eq!(rolling_average_spread(&samples, 5), None);
    }

    #[test]
    fn rolling_average_spread_computes_mean() {
        let now = Instant::now();
        let samples = vec![
            sample(now, "100", "0.01"),
            sample(now, "100", "0.02"),
            sample(now, "100", "0.03"),
        ];
        assert_eq!(rolling_average_spread(&samples, 3), Some(d("0.02")));
    }

    #[test]
    fn window_parkinson_vol_none_with_fewer_than_two_prices() {
        assert_eq!(window_parkinson_vol(&[d("100")]), None);
        assert_eq!(window_parkinson_vol(&[]), None);
    }

    #[test]
    fn window_parkinson_vol_zero_when_flat() {
        let prices = vec![d("100"), d("100"), d("100")];
        assert_eq!(window_parkinson_vol(&prices), Some(0.0));
    }

    #[test]
    fn window_parkinson_vol_positive_when_range_exists() {
        let prices = vec![d("99"), d("100"), d("101")];
        let vol = window_parkinson_vol(&prices).unwrap();
        assert!(vol > 0.0);
    }

    #[test]
    fn window_parkinson_vol_wider_range_reads_higher() {
        let narrow = vec![d("99.5"), d("100"), d("100.5")];
        let wide = vec![d("95"), d("100"), d("105")];
        assert!(window_parkinson_vol(&wide).unwrap() > window_parkinson_vol(&narrow).unwrap());
    }

    /// Builds a history: `n_baseline_buckets` buckets of calm, low-range
    /// prices further in the past than `short_window`, then a spiky
    /// high-range set of prices within the most recent `short_window` —
    /// simulating a genuine volatility surge relative to recent normal.
    fn spiky_history(now: Instant, short_window: Duration, bucket: Duration) -> Vec<PriceSample> {
        let mut samples = Vec::new();
        // Calm baseline: 6 buckets, each with a narrow 3-sample range
        // (small but non-zero, so the baseline itself has a real,
        // non-degenerate mean/stddev to compare the spike against).
        for bucket_i in 1..=6u32 {
            let base_age = short_window + bucket * bucket_i;
            for (offset_secs, price) in [(0u64, "100.0"), (1, "100.05"), (2, "100.1")] {
                let at = now - base_age - Duration::from_secs(offset_secs);
                samples.push(sample(at, price, "0.001"));
            }
        }
        // Current short window: a wide range, well outside the calm
        // baseline's distribution.
        for (offset_secs, price) in [(0u64, "90"), (1, "100"), (2, "110")] {
            let at = now - Duration::from_secs(offset_secs);
            samples.push(sample(at, price, "0.02"));
        }
        samples
    }

    #[test]
    fn assess_volatility_none_without_enough_baseline_history() {
        let now = Instant::now();
        let short_window = Duration::from_secs(60);
        // Only the current short window, no older buckets at all.
        let samples = vec![sample(now, "99", "0.01"), sample(now, "101", "0.01")];
        assert_eq!(assess_volatility(&samples, now, short_window, Duration::from_secs(60)), None);
    }

    #[test]
    fn assess_volatility_flags_a_genuine_spike() {
        let now = Instant::now();
        let short_window = Duration::from_secs(60);
        let bucket = Duration::from_secs(60);
        let samples = spiky_history(now, short_window, bucket);

        let reading = assess_volatility(&samples, now, short_window, bucket).expect("enough history");
        assert!(reading.short_window_vol > reading.baseline_mean);
        assert!(reading.zscore > 3.0, "expected a large positive zscore, got {}", reading.zscore);
    }

    #[test]
    fn assess_volatility_calm_market_has_a_small_zscore() {
        let now = Instant::now();
        let short_window = Duration::from_secs(60);
        let bucket = Duration::from_secs(60);
        let mut samples = Vec::new();
        // Baseline buckets and the current window are all equally calm.
        for bucket_i in 0..=6u32 {
            for offset_secs in [0u64, 1, 2] {
                let at = now - bucket * bucket_i - Duration::from_secs(offset_secs);
                samples.push(sample(at, "100.05", "0.001"));
            }
        }
        let reading = assess_volatility(&samples, now, short_window, bucket).expect("enough history");
        assert!(reading.zscore.abs() < 3.0, "expected a small zscore for a calm market, got {}", reading.zscore);
    }
}
