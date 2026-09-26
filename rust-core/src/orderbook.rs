//! In-memory order book, keyed by price level. Currency-agnostic: it knows
//! nothing about which asset it's tracking beyond the `symbol` label.

use rust_decimal::Decimal;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

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
}

impl OrderBook {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
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
    }

    /// Applies a delta, as a Kraken "update" message does. A zero quantity
    /// at a price level means that level is removed.
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
