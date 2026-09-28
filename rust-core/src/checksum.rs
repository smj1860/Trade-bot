//! Kraken WebSocket v2 order-book checksum validation (institutional audit
//! Phase 1.2). Every `book` channel message (snapshot and update alike)
//! carries a `checksum` field computed over the top 10 bid/ask levels;
//! recomputing the same value locally after applying a message is the
//! cheapest way to detect a desynced local book — the single most common
//! way an order-book maintainer silently goes wrong, and something every
//! other guardrail in this codebase (slippage, spread, the volatility
//! breaker) implicitly trusts is correct, since none of them can tell a
//! good book from a subtly corrupted one on their own.
//!
//! Algorithm, per Kraken's own docs
//! (<https://docs.kraken.com/api/docs/guides/spot-ws-book-v2/>):
//!   1. Take the top 10 ask levels (lowest price first) and the top 10 bid
//!      levels (highest price first).
//!   2. For each level, format price and quantity by removing the decimal
//!      point, then stripping leading zeros (trailing zeros stay — they're
//!      just digits once the decimal point is gone).
//!   3. Concatenate: all formatted ask levels in order (price then qty,
//!      level after level), then all formatted bid levels the same way,
//!      into one string.
//!   4. Standard CRC-32 (the IEEE/zlib variant — what `crc32fast`
//!      implements) of that string's ASCII bytes, as an unsigned 32-bit
//!      integer.
//!
//! Correctness here depends on one property of `rust_decimal::Decimal`
//! that isn't obvious from its name: parsing a string preserves its
//! original scale (trailing zeros included) rather than normalizing them
//! away — `Decimal::from_str("0.00100000").to_string()` stays
//! `"0.00100000"`, not `"0.001"`. `kraken.rs`'s `parse_levels` already
//! depends on this (its own docs note serde_json's `arbitrary_precision`
//! feature preserves the wire text into `Decimal::from_str` rather than
//! round-tripping through `f64`) — but `preserves_trailing_zeros_through_round_trip`
//! below exists specifically to catch a regression (a rust_decimal
//! upgrade, or new arithmetic on a stored level that rescales it) that
//! would change formatted output for reasons that have nothing to do with
//! an actually-desynced book.
//!
//! **Fixed-precision padding (critical, found via live testing against
//! Kraken's real feed — see git history for the incident this fixed):**
//! Kraken's own JSON serializer does NOT send every level at the trading
//! pair's full fixed precision — it strips trailing zeros per-message, so
//! the same book can show `"0.0931"` on one level and `"0.0931001"` on
//! another in the very same message. Kraken's checksum is nonetheless
//! computed as though every value were first padded out to the pair's
//! fixed `pair_decimals` (price) / `lot_decimals` (quantity) precision —
//! the same precision this codebase already tracks per-symbol as
//! `SymbolConfig.tick_size` / `lot_size` (see `decimals_for_config_value`
//! in config.rs) — and only THEN had its decimal point and leading zeros
//! stripped. Skipping this padding step produces checksums that mismatch
//! real, perfectly-in-sync Kraken order books essentially immediately;
//! it does not show up against Kraken's own published worked example
//! because that example's data happens to already be presented at full
//! fixed precision for every level.

use rust_decimal::Decimal;

/// Formats one price or quantity component per Kraken's checksum
/// algorithm: pad the value to `decimals` fractional digits (Kraken's own
/// wire messages may show fewer — see this module's top-level docs), then
/// strip the decimal point, then strip leading zeros. A value that
/// normalizes to nothing (all zeros — never a real resting level, but
/// defensive) becomes `"0"` rather than an empty string, since an empty
/// component would silently shift the rest of the concatenation.
fn format_component(value: Decimal, decimals: u32) -> String {
    let padded = format!("{:.prec$}", value, prec = decimals as usize);
    let without_point: String = padded.chars().filter(|c| *c != '.').collect();
    let stripped = without_point.trim_start_matches('0');
    if stripped.is_empty() {
        "0".to_string()
    } else {
        stripped.to_string()
    }
}

fn format_levels(levels: &[(Decimal, Decimal)], price_decimals: u32, qty_decimals: u32) -> String {
    let mut s = String::new();
    for (price, qty) in levels {
        s.push_str(&format_component(*price, price_decimals));
        s.push_str(&format_component(*qty, qty_decimals));
    }
    s
}

/// Computes Kraken's book checksum over the given ask/bid levels. `asks`
/// must already be sorted lowest-price-first and `bids` highest-price-first,
/// each truncated to (at most) the top 10 — exactly what
/// `OrderBook::ask_levels(10)` / `OrderBook::bid_levels(10)` return. A book
/// with fewer than 10 levels on a side is used as-is; Kraken's own
/// algorithm has no special padding case for a thin book.
///
/// `price_decimals`/`qty_decimals` are the trading pair's fixed precision
/// (Kraken's `pair_decimals`/`lot_decimals`) — see this module's top-level
/// docs for why every value must be padded to these before formatting.
/// Callers derive these from `SymbolConfig.tick_size`/`lot_size` via
/// `config::decimal_places`.
pub fn compute_book_checksum(
    asks: &[(Decimal, Decimal)],
    bids: &[(Decimal, Decimal)],
    price_decimals: u32,
    qty_decimals: u32,
) -> u32 {
    let mut input = format_levels(asks, price_decimals, qty_decimals);
    input.push_str(&format_levels(bids, price_decimals, qty_decimals));
    crc32fast::hash(input.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn preserves_trailing_zeros_through_round_trip() {
        // See this module's top-level docs: the whole checksum depends on
        // this. If a future rust_decimal upgrade ever changes this
        // behavior, this test fails loudly here rather than the checksum
        // just quietly starting to mismatch real Kraken data.
        assert_eq!(Decimal::from_str("0.00100000").unwrap().to_string(), "0.00100000");
        assert_eq!(Decimal::from_str("45296.1").unwrap().to_string(), "45296.1");
    }

    #[test]
    fn format_component_strips_point_and_leading_zeros() {
        assert_eq!(format_component(d("45285.2"), 1), "452852");
        assert_eq!(format_component(d("0.00100000"), 8), "100000");
    }

    #[test]
    fn format_component_of_a_whole_number_is_unchanged() {
        assert_eq!(format_component(d("5"), 0), "5");
    }

    #[test]
    fn format_component_of_zero_is_the_literal_zero() {
        assert_eq!(format_component(d("0"), 0), "0");
        assert_eq!(format_component(d("0.0"), 1), "0");
    }

    #[test]
    fn format_component_pads_to_the_pairs_fixed_precision() {
        // The critical fix: Kraken's own wire messages don't always show a
        // value at the pair's full fixed precision (see this module's
        // top-level docs), so the same logical price must format
        // identically regardless of how many digits happened to arrive.
        assert_eq!(format_component(d("0.0931"), 7), format_component(d("0.0931000"), 7));
        assert_eq!(format_component(d("0.0931"), 7), "931000");
    }

    #[test]
    fn padding_makes_the_checksum_independent_of_kraken_trailing_zero_stripping() {
        // Reproduces the root cause of a real production incident found via
        // live testing against Kraken's actual feed: its JSON serializer
        // strips trailing zeros inconsistently per-level, even within one
        // message (a live DOGE/USD book showed "0.0931" next to
        // "0.0931001" in the same update). Without padding every value out
        // to the pair's fixed `tick_size`/`lot_size` precision first, this
        // produced checksum mismatches against a perfectly in-sync book —
        // the same price arriving with fewer digits than usual looked like
        // a different, wrong book. With padding, both wire
        // representations of the same price/qty hash identically.
        let price_decimals = 7;
        let qty_decimals = 8;
        let stripped = vec![(d("0.0931"), d("1265.625"))];
        let full_precision = vec![(d("0.0931000"), d("1265.62500000"))];
        let bids: Vec<(Decimal, Decimal)> = vec![];

        assert_eq!(
            compute_book_checksum(&stripped, &bids, price_decimals, qty_decimals),
            compute_book_checksum(&full_precision, &bids, price_decimals, qty_decimals),
        );
    }

    /// Kraken's own worked example
    /// (<https://docs.kraken.com/api/docs/guides/spot-ws-book-v2/>):
    /// 10 ask levels low-to-high, 10 bid levels high-to-low, expected
    /// checksum 3310070434. This is the one test in this module that
    /// would catch a subtly wrong concatenation order or CRC variant that
    /// every other test here, being self-referential, cannot.
    #[test]
    fn matches_krakens_published_worked_example() {
        let asks = vec![
            (d("45285.2"), d("0.00100000")),
            (d("45286.4"), d("1.54571953")),
            (d("45286.6"), d("1.54571109")),
            (d("45289.6"), d("1.54560911")),
            (d("45290.2"), d("0.15890660")),
            (d("45291.8"), d("1.54553491")),
            (d("45294.7"), d("0.04454749")),
            (d("45296.1"), d("0.35380000")),
            (d("45297.5"), d("0.09945542")),
            (d("45299.5"), d("0.18772827")),
        ];
        let bids = vec![
            (d("45283.5"), d("0.10000000")),
            (d("45283.4"), d("1.54582015")),
            (d("45282.1"), d("0.10000000")),
            (d("45281.0"), d("0.10000000")),
            (d("45280.3"), d("1.54592586")),
            (d("45279.0"), d("0.07990000")),
            (d("45277.6"), d("0.03310103")),
            (d("45277.5"), d("0.30000000")),
            (d("45277.3"), d("1.54602737")),
            (d("45276.6"), d("0.15445238")),
        ];

        // This example's data happens to already be presented at full fixed
        // precision on every level (price to 1 decimal, qty to 8), so the
        // padding fix is a no-op here and this keeps passing unchanged.
        assert_eq!(compute_book_checksum(&asks, &bids, 1, 8), 3_310_070_434);
    }

    #[test]
    fn a_single_changed_level_changes_the_checksum() {
        let asks = vec![(d("100.0"), d("1.0"))];
        let bids = vec![(d("99.0"), d("1.0"))];
        let baseline = compute_book_checksum(&asks, &bids, 1, 1);

        let moved_asks = vec![(d("100.1"), d("1.0"))];
        assert_ne!(compute_book_checksum(&moved_asks, &bids, 1, 1), baseline);
    }

    #[test]
    fn a_thinner_than_ten_deep_book_is_used_as_is() {
        // No special-casing needed — just confirms this doesn't panic on
        // a book with fewer than 10 levels on a side, which is the normal
        // state right after a fresh subscription with limited liquidity.
        let asks = vec![(d("100.0"), d("1.0"))];
        let bids = vec![];
        let _ = compute_book_checksum(&asks, &bids, 1, 1);
    }
}
