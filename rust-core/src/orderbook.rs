//! In-memory order book, keyed by price level. Currency-agnostic: it knows
//! nothing about which asset it's tracking beyond the `symbol` label.

use rust_decimal::Decimal;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// How long a book keeps its own rolling mid-price/spread history for —
/// see `guardrails.rs`, which reads this history to compute a rolling
/// average spread and a volatility baseline. An hour comfortably covers
/// both (the spread guard's "3x 1-hour rolling average" and the
/// volatility breaker's baseline window), with one shared history buffer
/// rather than two separately-sampled ones that could drift apart.
const HISTORY_MAX_AGE: Duration = Duration::from_secs(3600);
/// Hard cap on retained samples regardless of update rate, so a
/// pathologically chatty feed can't grow this without bound before the
/// age-based prune below ever gets a chance to run.
const HISTORY_MAX_SAMPLES: usize = 20_000;

/// One point of a book's own rolling mid-price/spread history, recorded
/// automatically whenever both sides of the book are populated after a
/// snapshot or update. `at` is process-local (`Instant`), which is all a
/// same-process rolling-window guardrail needs — never persisted or
/// compared across a restart.
#[derive(Debug, Clone, Copy)]
pub struct PriceSample {
    pub at: Instant,
    pub mid: Decimal,
    pub spread_pct: Decimal,
}

/// Shared, live order book state, keyed by normalized symbol. The Kraken
/// ingestion task writes into this; the risk engine reads current prices
/// out of it when evaluating a market order or a position limit. It's the
/// single source of truth for "what does the book look like right now,"
/// separate from the `MarketDataEvent` stream (which is a broadcast of
/// changes, not a queryable snapshot).
pub type SharedBooks = Arc<Mutex<HashMap<String, OrderBook>>>;

pub fn new_shared_books() -> SharedBooks {
    Arc::new(Mutex::new(HashMap::new()))
}

#[derive(Debug, Default, Clone)]
pub struct OrderBook {
    pub symbol: String,
    /// price -> quantity. BTreeMap keeps levels sorted; best bid is the
    /// highest price (iterate from the back), best ask is the lowest
    /// (iterate from the front).
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    /// Rolling mid-price/spread history, oldest first — see `PriceSample`
    /// and `guardrails.rs`. Recorded on every snapshot/update that leaves
    /// both sides of the book populated, so it's fed continuously by
    /// ordinary market-data ingestion (kraken.rs) with no separate wiring
    /// needed — a guardrail check at order-evaluation time just reads
    /// whatever history has already accumulated.
    history: VecDeque<PriceSample>,
    /// The book's subscribed depth (Kraken's `book` channel `depth`
    /// parameter), or `None` for an unbounded book (used by tests that
    /// don't care about depth maintenance). When set, `apply_snapshot` and
    /// `apply_update` trim each side back down to this many levels after
    /// every change — see their docs for why this is required, not
    /// optional, for a depth-limited Kraken subscription.
    depth: Option<usize>,
}

impl OrderBook {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            history: VecDeque::new(),
            depth: None,
        }
    }

    /// Like `new`, but maintains the book at a fixed depth on every
    /// snapshot/update — what `kraken.rs` uses for a real Kraken
    /// subscription (see `apply_update`'s docs).
    pub fn with_depth(symbol: impl Into<String>, depth: usize) -> Self {
        Self { depth: Some(depth), ..Self::new(symbol) }
    }

    /// Replaces the book entirely, as a Kraken "snapshot" message does.
    pub fn apply_snapshot(&mut self, bids: Vec<(Decimal, Decimal)>, asks: Vec<(Decimal, Decimal)>) {
        self.bids.clear();
        self.asks.clear();
        for (price, qty) in bids {
            self.bids.insert(price, qty);
        }
        for (price, qty) in asks {
            self.asks.insert(price, qty);
        }
        self.trim_to_depth();
        self.record_history_sample();
    }

    /// Applies a delta, as a Kraken "update" message does. A zero quantity
    /// at a price level means that level is removed.
    ///
    /// For a depth-limited subscription (Kraken's `book` channel `depth`
    /// parameter — this codebase always subscribes at `BOOK_DEPTH`), Kraken
    /// does not reliably pair every new level that enters the window with
    /// an explicit deletion of the level it displaces: per Kraken's own
    /// docs, the *client* is responsible for trimming each side back down
    /// to the subscribed depth after applying a delta, dropping the
    /// worst-priced excess entries (lowest-price bids, highest-price
    /// asks). Skipping this step is exactly the kind of bug that looks
    /// harmless on paper (the book "still holds valid data") but produces
    /// a checksum computed over the wrong top-10 window — confirmed via
    /// live testing against Kraken's real feed, where every symbol
    /// mismatched on its very first post-snapshot update despite the
    /// applied deltas themselves being entirely correct.
    pub fn apply_update(&mut self, bids: Vec<(Decimal, Decimal)>, asks: Vec<(Decimal, Decimal)>) {
        for (price, qty) in bids {
            if qty.is_zero() {
                self.bids.remove(&price);
            } else {
                self.bids.insert(price, qty);
            }
        }
        for (price, qty) in asks {
            if qty.is_zero() {
                self.asks.remove(&price);
            } else {
                self.asks.insert(price, qty);
            }
        }
        self.trim_to_depth();
        self.record_history_sample();
    }

    /// Drops the worst-priced excess levels on each side down to `depth`
    /// (a no-op when `depth` is `None`, or already within it). Bids are
    /// sorted ascending in the BTreeMap, so the worst bid is the lowest
    /// price (`pop_first`); asks are sorted ascending too, so the worst
    /// ask is the highest price (`pop_last`).
    fn trim_to_depth(&mut self) {
        let Some(depth) = self.depth else { return };
        while self.bids.len() > depth {
            self.bids.pop_first();
        }
        while self.asks.len() > depth {
            self.asks.pop_last();
        }
    }

    /// Appends a `PriceSample` from the book's current best bid/ask (a
    /// no-op if either side is empty — a one-sided book has no meaningful
    /// mid/spread), then prunes anything older than `HISTORY_MAX_AGE` and
    /// enforces `HISTORY_MAX_SAMPLES` as a hard backstop.
    fn record_history_sample(&mut self) {
        let (Some((bid, _)), Some((ask, _))) = (self.best_bid(), self.best_ask()) else {
            return;
        };
        if ask <= bid {
            // A crossed/locked book mid-update is a transient, not a real
            // tradeable state — skip recording rather than let a
            // momentarily-negative spread pollute the guardrails' history.
            return;
        }
        let mid = (bid + ask) / Decimal::TWO;
        if mid <= Decimal::ZERO {
            return;
        }
        let spread_pct = (ask - bid) / mid;
        let now = Instant::now();
        self.history.push_back(PriceSample { at: now, mid, spread_pct });

        while let Some(front) = self.history.front() {
            if now.duration_since(front.at) > HISTORY_MAX_AGE {
                self.history.pop_front();
            } else {
                break;
            }
        }
        while self.history.len() > HISTORY_MAX_SAMPLES {
            self.history.pop_front();
        }
    }

    /// This book's own rolling history, oldest first, filtered to samples
    /// no older than `max_age` — see `guardrails.rs` for how it's used
    /// (rolling average spread, volatility baseline).
    pub fn recent_samples(&self, max_age: Duration) -> Vec<PriceSample> {
        let now = Instant::now();
        self.history.iter().filter(|s| now.duration_since(s.at) <= max_age).copied().collect()
    }

    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.iter().next_back().map(|(p, q)| (*p, *q))
    }

    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.iter().next().map(|(p, q)| (*p, *q))
    }

    /// Bid levels, best (highest price) first.
    pub fn bid_levels(&self, depth: usize) -> Vec<(Decimal, Decimal)> {
        self.bids.iter().rev().take(depth).map(|(p, q)| (*p, *q)).collect()
    }

    /// Ask levels, best (lowest price) first.
    pub fn ask_levels(&self, depth: usize) -> Vec<(Decimal, Decimal)> {
        self.asks.iter().take(depth).map(|(p, q)| (*p, *q)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn snapshot_replaces_book() {
        let mut book = OrderBook::new("BTC-USD");
        book.apply_snapshot(
            vec![(d("100.0"), d("1.0")), (d("99.0"), d("2.0"))],
            vec![(d("101.0"), d("1.5")), (d("102.0"), d("3.0"))],
        );
        assert_eq!(book.best_bid(), Some((d("100.0"), d("1.0"))));
        assert_eq!(book.best_ask(), Some((d("101.0"), d("1.5"))));
        assert_eq!(book.bid_levels(10), vec![(d("100.0"), d("1.0")), (d("99.0"), d("2.0"))]);
        assert_eq!(book.ask_levels(10), vec![(d("101.0"), d("1.5")), (d("102.0"), d("3.0"))]);
    }

    #[test]
    fn update_adds_and_changes_levels() {
        let mut book = OrderBook::new("BTC-USD");
        book.apply_snapshot(vec![(d("100.0"), d("1.0"))], vec![(d("101.0"), d("1.0"))]);
        book.apply_update(vec![(d("100.5"), d("0.5"))], vec![]);
        assert_eq!(book.best_bid(), Some((d("100.5"), d("0.5"))));
    }

    #[test]
    fn zero_quantity_update_removes_level() {
        let mut book = OrderBook::new("BTC-USD");
        book.apply_snapshot(
            vec![(d("100.0"), d("1.0")), (d("99.0"), d("2.0"))],
            vec![],
        );
        book.apply_update(vec![(d("100.0"), d("0"))], vec![]);
        assert_eq!(book.best_bid(), Some((d("99.0"), d("2.0"))));
    }

    #[test]
    fn unbounded_book_never_trims_when_no_depth_is_set() {
        // OrderBook::new (no depth) is what most of this module's other
        // tests use — confirms that path is unaffected by the trimming
        // added for with_depth.
        let mut book = OrderBook::new("BTC-USD");
        book.apply_snapshot(
            vec![(d("100.0"), d("1.0")), (d("99.0"), d("1.0")), (d("98.0"), d("1.0"))],
            vec![],
        );
        assert_eq!(book.bid_levels(10).len(), 3);
    }

    #[test]
    fn with_depth_trims_a_snapshot_that_arrives_oversized() {
        let mut book = OrderBook::with_depth("BTC-USD", 2);
        book.apply_snapshot(
            vec![(d("100.0"), d("1.0")), (d("99.0"), d("1.0")), (d("98.0"), d("1.0"))],
            vec![(d("101.0"), d("1.0")), (d("102.0"), d("1.0")), (d("103.0"), d("1.0"))],
        );
        // Worst bid (lowest price, 98.0) and worst ask (highest price,
        // 103.0) are the ones dropped to get back to depth 2.
        assert_eq!(book.bid_levels(10), vec![(d("100.0"), d("1.0")), (d("99.0"), d("1.0"))]);
        assert_eq!(book.ask_levels(10), vec![(d("101.0"), d("1.0")), (d("102.0"), d("1.0"))]);
    }

    #[test]
    fn with_depth_trims_the_worst_level_when_a_new_one_pushes_the_book_over_depth() {
        // Reproduces the real bug this trimming fixes: a depth-limited
        // Kraken subscription doesn't always pair a new level entering
        // the top-N with an explicit deletion of the level it displaces —
        // per Kraken's own docs, the client must trim back down to depth
        // itself. Without this, live testing showed every symbol's book
        // checksum mismatching on its very first post-snapshot update.
        let mut book = OrderBook::with_depth("BTC-USD", 2);
        book.apply_snapshot(
            vec![],
            vec![(d("100.0"), d("1.0")), (d("101.0"), d("1.0"))],
        );
        // A new, worse ask arrives with no accompanying deletion.
        book.apply_update(vec![], vec![(d("99.5"), d("1.0"))]);

        assert_eq!(
            book.ask_levels(10),
            vec![(d("99.5"), d("1.0")), (d("100.0"), d("1.0"))],
            "the book should have trimmed the new worst level (101.0) to stay at depth 2, \
             matching what a depth-limited Kraken subscription itself would show"
        );
    }

    #[test]
    fn best_bid_and_ask_never_cross_after_realistic_updates() {
        let mut book = OrderBook::new("BTC-USD");
        book.apply_snapshot(
            vec![(d("100.0"), d("1.0"))],
            vec![(d("101.0"), d("1.0"))],
        );
        // A tighter bid arrives, still must stay below best ask in a sane feed.
        book.apply_update(vec![(d("100.9"), d("0.2"))], vec![]);
        let (bid, _) = book.best_bid().unwrap();
        let (ask, _) = book.best_ask().unwrap();
        assert!(bid < ask);
    }
}
