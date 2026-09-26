//! Risk guardrail enforcement. This is the last line of defense before an
//! order request reaches an exchange: it runs regardless of what a
//! strategy or model asked for, and it fails closed — any missing data,
//! bad config, or unrecognized symbol is a rejection, never a pass.
//!
//! Enforced, in order:
//! 1. Order rate limit (global orders/minute)
//! 2. Symbol must be configured and enabled for the requesting exchange
//! 3. Order size <= per-symbol max_order_size
//! 4. Order notional (qty * price) <= per-symbol max_order_notional_usd
//! 5. Projected position after this order <= per-symbol max_position_usd
//! 6. Projected combined portfolio exposure <= global max_total_position_usd

use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rust_decimal::Decimal;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::orderbook::SharedBooks;
use crate::persistence::{FillRecord, Store};
use crate::proto::pb::{OrderRequest, OrderSide};

const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);
const SECONDS_PER_DAY: u64 = 86_400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskVerdict {
    Approved,
    Rejected(String),
}

/// Net position in a single symbol, tracked on an average-cost basis. A
/// fill in the same direction as `qty` extends the position and rolls
/// `avg_entry_price` forward as a size-weighted average; a fill in the
/// opposite direction closes some or all of the position and realizes PnL
/// against `avg_entry_price` for the overlapping quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionState {
    qty: Decimal,
    avg_entry_price: Decimal,
}

impl PositionState {
    const ZERO: PositionState = PositionState {
        qty: Decimal::ZERO,
        avg_entry_price: Decimal::ZERO,
    };
}

/// UTC day index (days since the Unix epoch) used to decide when the daily
/// kill-switch counter should reset. Deliberately avoids a `chrono`
/// dependency — integer division of Unix seconds by seconds-per-day is a
/// correct UTC day boundary on its own.
fn current_day_index() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / SECONDS_PER_DAY)
        .unwrap_or(0)
}

/// Sign of a `Decimal` as -1/0/1, via comparison rather than relying on a
/// `Decimal::signum()` API this codebase hasn't otherwise exercised.
fn sign_of(d: Decimal) -> i32 {
    if d > Decimal::ZERO {
        1
    } else if d < Decimal::ZERO {
        -1
    } else {
        0
    }
}

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

pub struct RiskEngine {
    config: Arc<Config>,
    books: SharedBooks,
    /// Net position per normalized symbol, average-cost basis. Updated only
    /// on confirmed fills coming from `apply_fill` (driven by Kraken's
    /// private execution feed — see kraken_private_ws.rs). Starts empty, so
    /// every symbol reads as flat until a real fill has been applied.
    positions: Mutex<HashMap<String, PositionState>>,
    recent_order_times: Mutex<VecDeque<Instant>>,
    /// Realized PnL (USD) accumulated since the start of the current UTC
    /// day, per the kill switch. Reset to zero whenever `kill_switch_day`
    /// rolls over.
    realized_pnl_usd: Mutex<Decimal>,
    kill_switch_day: Mutex<u64>,
    /// Set via `attach_store` once a `Store` is available. `None` means
    /// running with in-memory-only state (every unit test in this module,
    /// and a real deployment only if the database failed to open at
    /// startup) — `apply_fill` still works correctly, it just has nothing
    /// to survive a restart with.
    store: Mutex<Option<Arc<Store>>>,
}

impl RiskEngine {
    pub fn new(config: Arc<Config>, books: SharedBooks) -> Self {
        Self {
            config,
            books,
            positions: Mutex::new(HashMap::new()),
            recent_order_times: Mutex::new(VecDeque::new()),
            realized_pnl_usd: Mutex::new(Decimal::ZERO),
            kill_switch_day: Mutex::new(current_day_index()),
            store: Mutex::new(None),
        }
    }

    /// Hydrates in-memory state from `store` (positions and the kill
    /// switch's daily counter, if any was saved) and keeps `store` for
    /// `apply_fill` to write through to from here on. Called once at
    /// startup, after `RiskEngine::new` — kept separate from `new` itself
    /// so every existing unit test can keep constructing a plain
    /// in-memory engine without touching a database.
    ///
    /// A kill-switch counter persisted from a previous UTC day is treated
    /// as stale and NOT loaded — restoring yesterday's loss as today's
    /// would either trip the switch for no reason or (if it was a gain)
    /// mask today's actual losses under a leftover cushion. Positions are
    /// loaded regardless of age, since a position doesn't expire at
    /// midnight the way a daily loss counter does.
    pub async fn attach_store(&self, store: Arc<Store>) -> anyhow::Result<()> {
        let persisted_positions = store.load_positions()?;
        {
            let mut positions = self.positions.lock().await;
            for p in persisted_positions {
                positions.insert(p.symbol, PositionState { qty: p.qty, avg_entry_price: p.avg_entry_price });
            }
        }

        let today = current_day_index();
        if let Some(state) = store.load_kill_switch_state()? {
            if state.day_index == today {
                *self.kill_switch_day.lock().await = state.day_index;
                *self.realized_pnl_usd.lock().await = state.realized_pnl_usd;
                tracing::info!(realized_today = %state.realized_pnl_usd, "restored today's kill-switch counter from disk");
            } else {
                tracing::info!(
                    persisted_day = state.day_index,
                    today,
                    "persisted kill-switch counter is from a previous day, starting today at zero"
                );
            }
        }

        *self.store.lock().await = Some(store);
        Ok(())
    }

    /// Applies a confirmed fill from the exchange to this symbol's tracked
    /// position, realizing PnL against the existing average entry price for
    /// any quantity that closes or flips the position. `qty` is always
    /// positive (the size of this specific fill); `side` says which
    /// direction it moved the position. Returns `false` without changing
    /// any state if this fill was already applied (see `exec_id` below).
    ///
    /// Deliberately driven by Kraken's per-execution `last_qty`/`last_price`
    /// fields (see kraken_private_ws.rs), not by diffing `cum_qty` across
    /// messages — a `cum_qty` delta requires locally-persisted state to
    /// interpret, and that state would be lost/reset across a WebSocket
    /// reconnect, silently double- or under-counting fills. A per-event
    /// delta needs no such state.
    ///
    /// `exec_id` (Kraken's `exec_id`/`trade_id`, when present) is checked
    /// against the fills already recorded in `store` before anything else
    /// happens — this is the durable half of fill de-duplication;
    /// kraken_private_ws.rs's in-memory window catches a redelivery within
    /// the same connection cheaply, this catches one across a reconnect or
    /// a full process restart, which the in-memory window cannot.
    /// `client_order_id` is carried through only for the fills audit
    /// trail; it plays no role in the position/PnL math.
    pub async fn apply_fill(
        &self,
        symbol: &str,
        side: OrderSide,
        qty: Decimal,
        price: Decimal,
        exec_id: Option<&str>,
        client_order_id: Option<&str>,
    ) -> bool {
        if qty <= Decimal::ZERO {
            tracing::warn!(symbol, ?side, %qty, "apply_fill called with non-positive quantity, ignoring");
            return false;
        }

        let store = self.store.lock().await.clone();

        if let Some(exec_id) = exec_id {
            match &store {
                Some(store) => match store.fill_exists(exec_id) {
                    Ok(true) => {
                        tracing::debug!(exec_id, "fill already recorded, skipping apply_fill");
                        return false;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        tracing::error!(exec_id, error = %e, "failed to check fill dedup store, applying anyway");
                    }
                },
                None => {}
            }
        }

        self.maybe_roll_over_day().await;

        let signed_qty = match side {
            OrderSide::Sell => -qty,
            _ => qty,
        };

        let mut realized_delta = Decimal::ZERO;
        let new_state;
        {
            let mut positions = self.positions.lock().await;
            let current = positions.get(symbol).copied().unwrap_or(PositionState::ZERO);

            new_state = if current.qty.is_zero() || sign_of(current.qty) == sign_of(signed_qty) {
                // Opening or extending a position in the same direction:
                // roll the average entry price forward, size-weighted.
                let new_qty = current.qty + signed_qty;
                let new_avg = if new_qty.is_zero() {
                    Decimal::ZERO
                } else {
                    ((current.qty.abs() * current.avg_entry_price) + (qty * price)) / new_qty.abs()
                };
                PositionState { qty: new_qty, avg_entry_price: new_avg }
            } else {
                // Opposite direction: this fill closes some or all of the
                // existing position, realizing PnL on the closed quantity
                // against the existing average entry price.
                let closing_qty = qty.min(current.qty.abs());
                let pnl_per_unit = if current.qty > Decimal::ZERO {
                    price - current.avg_entry_price
                } else {
                    current.avg_entry_price - price
                };
                realized_delta = pnl_per_unit * closing_qty;

                let leftover = qty - closing_qty;
                let new_qty = current.qty + signed_qty;
                if leftover > Decimal::ZERO {
                    // The fill was larger than the existing position: it
                    // fully closes it and flips to a fresh position on the
                    // remaining quantity, with a new average-entry basis.
                    PositionState { qty: new_qty, avg_entry_price: price }
                } else {
                    // Partial or exact close: position shrinks (or hits
                    // zero) but the average entry price for whatever
                    // remains is unchanged.
                    PositionState {
                        qty: new_qty,
                        avg_entry_price: if new_qty.is_zero() { Decimal::ZERO } else { current.avg_entry_price },
                    }
                }
            };
            positions.insert(symbol.to_string(), new_state);
        }

        let mut realized_after = None;
        if !realized_delta.is_zero() {
            let mut realized = self.realized_pnl_usd.lock().await;
            *realized += realized_delta;
            realized_after = Some(*realized);
            tracing::info!(symbol, %realized_delta, total_today = %*realized, "realized PnL updated from fill");
        }

        if let Some(store) = &store {
            let now = now_ns();
            if let Err(e) = store.upsert_position(symbol, new_state.qty, new_state.avg_entry_price, now) {
                tracing::error!(symbol, error = %e, "failed to persist updated position");
            }
            if let Some(realized_after) = realized_after {
                let day = *self.kill_switch_day.lock().await;
                if let Err(e) = store.save_kill_switch_state(day, realized_after) {
                    tracing::error!(error = %e, "failed to persist kill-switch state");
                }
            }
            let fill = FillRecord {
                exec_id: exec_id.map(|s| s.to_string()),
                client_order_id: client_order_id.map(|s| s.to_string()),
                symbol: symbol.to_string(),
                side: format!("{side:?}").to_uppercase(),
                qty,
                price,
                realized_pnl_usd: realized_delta,
                applied_at_ns: now,
            };
            if let Err(e) = store.record_fill(&fill) {
                tracing::error!(symbol, error = %e, "failed to record fill in persistence store");
            }
        }

        true
    }

    /// Current net position for a symbol, in base-asset units. Zero for any
    /// symbol with no fills applied yet.
    pub async fn position(&self, symbol: &str) -> Decimal {
        self.positions.lock().await.get(symbol).map(|p| p.qty).unwrap_or(Decimal::ZERO)
    }

    /// Realized PnL (USD) accumulated since the start of the current UTC
    /// day.
    pub async fn realized_pnl_today(&self) -> Decimal {
        *self.realized_pnl_usd.lock().await
    }

    /// Resets the daily realized-PnL counter when the UTC day has rolled
    /// over since it was last checked, persisting the reset immediately so
    /// a restart moments later doesn't resurrect yesterday's total.
    async fn maybe_roll_over_day(&self) {
        let today = current_day_index();
        let mut day = self.kill_switch_day.lock().await;
        if *day != today {
            *day = today;
            let mut realized = self.realized_pnl_usd.lock().await;
            tracing::info!(previous_total = %*realized, "kill switch day rolled over, resetting realized PnL");
            *realized = Decimal::ZERO;

            if let Some(store) = self.store.lock().await.as_ref() {
                if let Err(e) = store.save_kill_switch_state(today, Decimal::ZERO) {
                    tracing::error!(error = %e, "failed to persist kill-switch day rollover");
                }
            }
        }
    }

    /// Fails closed on a bad config, consistent with the rest of this
    /// module: an unparseable daily-loss limit rejects every order rather
    /// than silently disabling the kill switch.
    async fn check_kill_switch(&self) -> Option<String> {
        self.maybe_roll_over_day().await;

        let Ok(max_daily_loss) = Decimal::from_str(&self.config.risk.global.kill_switch_max_daily_loss_usd) else {
            return Some("invalid config: kill_switch_max_daily_loss_usd".to_string());
        };

        let realized = *self.realized_pnl_usd.lock().await;
        if realized <= -max_daily_loss {
            return Some(format!(
                "daily kill switch triggered: realized loss {realized} has reached/exceeded the \
                 max daily loss of {max_daily_loss} — no further orders will be approved today"
            ));
        }
        None
    }

    pub async fn evaluate(&self, order: &OrderRequest) -> RiskVerdict {
        if let Some(reason) = self.check_kill_switch().await {
            return RiskVerdict::Rejected(reason);
        }

        if let Some(reason) = self.check_rate_limit().await {
            return RiskVerdict::Rejected(reason);
        }

        let Some(symbol_cfg) = self
            .config
            .symbols
            .iter()
            .find(|s| s.enabled && s.symbol == order.symbol && s.exchange == order.exchange)
        else {
            return RiskVerdict::Rejected(format!(
                "symbol {} is not configured/enabled for exchange {}",
                order.symbol, order.exchange
            ));
        };

        let Some(qty) = order.quantity.as_ref().and_then(|q| Decimal::from_str(&q.value).ok()) else {
            return RiskVerdict::Rejected("missing or unparseable quantity".to_string());
        };
        if qty <= Decimal::ZERO {
            return RiskVerdict::Rejected("quantity must be positive".to_string());
        }

        let Ok(max_order_size) = Decimal::from_str(&symbol_cfg.risk.max_order_size) else {
            return RiskVerdict::Rejected("invalid config: max_order_size".to_string());
        };
        if qty > max_order_size {
            return RiskVerdict::Rejected(format!(
                "order size {qty} exceeds max_order_size {max_order_size} for {}",
                order.symbol
            ));
        }

        let side = match OrderSide::try_from(order.side) {
            Ok(OrderSide::Buy) => OrderSide::Buy,
            Ok(OrderSide::Sell) => OrderSide::Sell,
            _ => return RiskVerdict::Rejected("order side must be BUY or SELL".to_string()),
        };

        let Some(price) = self.effective_price(order, side).await else {
            return RiskVerdict::Rejected(format!(
                "no price available to evaluate order for {} (no limit price and no live order book)",
                order.symbol
            ));
        };

        let notional = qty * price;
        let Ok(max_notional) = Decimal::from_str(&symbol_cfg.risk.max_order_notional_usd) else {
            return RiskVerdict::Rejected("invalid config: max_order_notional_usd".to_string());
        };
        if notional > max_notional {
            return RiskVerdict::Rejected(format!(
                "order notional {notional} exceeds max_order_notional_usd {max_notional} for {}",
                order.symbol
            ));
        }

        let Ok(max_position_usd) = Decimal::from_str(&symbol_cfg.risk.max_position_usd) else {
            return RiskVerdict::Rejected("invalid config: max_position_usd".to_string());
        };
        let signed_qty = if side == OrderSide::Buy { qty } else { -qty };
        let current_position = {
            let positions = self.positions.lock().await;
            positions.get(&order.symbol).map(|p| p.qty).unwrap_or(Decimal::ZERO)
        };
        let projected_notional = ((current_position + signed_qty) * price).abs();
        if projected_notional > max_position_usd {
            return RiskVerdict::Rejected(format!(
                "projected position {projected_notional} would exceed max_position_usd {max_position_usd} for {}",
                order.symbol
            ));
        }

        let Ok(max_total_position_usd) = Decimal::from_str(&self.config.risk.global.max_total_position_usd) else {
            return RiskVerdict::Rejected("invalid config: max_total_position_usd".to_string());
        };
        let combined_exposure = self.combined_exposure_excluding(&order.symbol).await + projected_notional;
        if combined_exposure > max_total_position_usd {
            return RiskVerdict::Rejected(format!(
                "combined portfolio exposure {combined_exposure} would exceed max_total_position_usd {max_total_position_usd}"
            ));
        }

        RiskVerdict::Approved
    }

    /// Rate limiting applies to every request regardless of verdict, so a
    /// client can't dodge it by sending orders that get rejected anyway.
    async fn check_rate_limit(&self) -> Option<String> {
        let mut recent = self.recent_order_times.lock().await;
        let now = Instant::now();
        while let Some(&front) = recent.front() {
            if now.duration_since(front) > RATE_LIMIT_WINDOW {
                recent.pop_front();
            } else {
                break;
            }
        }
        let limit = self.config.risk.global.max_orders_per_minute;
        if recent.len() as u32 >= limit {
            return Some(format!("global order rate limit exceeded ({limit} orders/minute)"));
        }
        recent.push_back(now);
        None
    }

    /// Limit price if given; otherwise the current best ask (buy) / best
    /// bid (sell) from the live book, since a market order's cost depends
    /// on where the book actually is right now.
    async fn effective_price(&self, order: &OrderRequest, side: OrderSide) -> Option<Decimal> {
        if let Some(p) = order.limit_price.as_ref().and_then(|p| Decimal::from_str(&p.value).ok()) {
            return Some(p);
        }
        let books = self.books.lock().await;
        let book = books.get(&order.symbol)?;
        match side {
            OrderSide::Buy => book.best_ask().map(|(p, _)| p),
            OrderSide::Sell => book.best_bid().map(|(p, _)| p),
            _ => None,
        }
    }

    /// Sum of |position * current best price| across every symbol except
    /// `exclude_symbol` (the caller adds that symbol's own projected
    /// exposure separately, since it's evaluating a hypothetical order,
    /// not the current position). Uses best bid as a price proxy for
    /// other symbols — a deliberate approximation for a guardrail, not
    /// meant to be a precise mark-to-market.
    async fn combined_exposure_excluding(&self, exclude_symbol: &str) -> Decimal {
        let positions = self.positions.lock().await;
        let books = self.books.lock().await;
        let mut total = Decimal::ZERO;
        for (symbol, position) in positions.iter() {
            if symbol == exclude_symbol || position.qty.is_zero() {
                continue;
            }
            if let Some(book) = books.get(symbol) {
                if let Some((price, _)) = book.best_bid() {
                    total += (position.qty * price).abs();
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExchangeConfig, GeneralConfig, GlobalRisk, RiskSection, SymbolConfig, SymbolRisk};
    use crate::orderbook::{new_shared_books, OrderBook};
    use crate::proto::pb::{Decimal as PbDecimal, OrderRequest, OrderType};

    fn test_config() -> Config {
        Config {
            general: GeneralConfig { log_level: "info".into() },
            exchanges: vec![ExchangeConfig {
                name: "kraken".into(),
                ws_url: "wss://ws.kraken.com/v2".into(),
                rest_url: "https://api.kraken.com".into(),
                enabled: true,
            }],
            symbols: vec![SymbolConfig {
                symbol: "BTC-USD".into(),
                exchange: "kraken".into(),
                exchange_native_symbol: "BTC/USD".into(),
                rest_native_symbol: "XBTUSD".into(),
                tick_size: "0.1".into(),
                lot_size: "0.00000001".into(),
                enabled: true,
                risk: SymbolRisk {
                    max_position_usd: "5000".into(),
                    max_order_size: "0.05".into(),
                    max_order_notional_usd: "2000".into(),
                },
            }],
            risk: RiskSection {
                global: GlobalRisk {
                    max_total_position_usd: "10000".into(),
                    max_orders_per_minute: 20,
                    kill_switch_max_daily_loss_usd: "500".into(),
                },
            },
            execution: crate::config::ExecutionConfig { dry_run: true },
            persistence: crate::config::PersistenceConfig { database_path: ":memory:".to_string() },
        }
    }

    fn buy_order(symbol: &str, qty: &str, limit_price: Option<&str>) -> OrderRequest {
        OrderRequest {
            client_order_id: "test-1".into(),
            symbol: symbol.into(),
            exchange: "kraken".into(),
            side: OrderSide::Buy as i32,
            r#type: if limit_price.is_some() { OrderType::Limit as i32 } else { OrderType::Market as i32 },
            quantity: Some(PbDecimal { value: qty.into() }),
            limit_price: limit_price.map(|p| PbDecimal { value: p.into() }),
            strategy_id: "test-strategy".into(),
        }
    }

    #[tokio::test]
    async fn rejects_unconfigured_symbol() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        let order = buy_order("DOGE-USD", "1.0", Some("0.1"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn approves_order_within_all_limits() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        // 0.01 BTC @ $30,000 = $300 notional, well under every limit.
        let order = buy_order("BTC-USD", "0.01", Some("30000"));
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    #[tokio::test]
    async fn rejects_order_size_over_symbol_limit() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        // max_order_size is 0.05
        let order = buy_order("BTC-USD", "0.06", Some("30000"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn rejects_order_notional_over_limit() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        // 0.04 BTC @ $60,000 = $2,400 notional > max_order_notional_usd (2000)
        let order = buy_order("BTC-USD", "0.04", Some("60000"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn rejects_market_order_with_no_live_book() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        let order = buy_order("BTC-USD", "0.01", None); // no limit price, no book populated
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn approves_market_order_using_live_book_price() {
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            book.apply_snapshot(
                vec![(Decimal::from_str("29999").unwrap(), Decimal::from_str("1").unwrap())],
                vec![(Decimal::from_str("30001").unwrap(), Decimal::from_str("1").unwrap())],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(test_config()), books);
        let order = buy_order("BTC-USD", "0.01", None); // market order, buys at best ask
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    #[tokio::test]
    async fn rate_limit_rejects_after_threshold() {
        let mut config = test_config();
        config.risk.global.max_orders_per_minute = 2;
        let engine = RiskEngine::new(Arc::new(config), new_shared_books());
        let order = buy_order("BTC-USD", "0.01", Some("30000"));

        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
        // third order within the window exceeds the limit of 2/minute
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn apply_fill_opens_and_tracks_a_long_position() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("0.01").unwrap());
        assert_eq!(engine.realized_pnl_today().await, Decimal::ZERO);
    }

    #[tokio::test]
    async fn apply_fill_extends_a_position_with_a_weighted_average_entry_price() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("32000").unwrap(), None, None)
            .await;
        // 0.02 total, avg entry (30000 + 32000) / 2 = 31000
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("0.02").unwrap());

        // Closing the full 0.02 @ 33000 realizes (33000 - 31000) * 0.02 = 40
        engine
            .apply_fill("BTC-USD", OrderSide::Sell, Decimal::from_str("0.02").unwrap(), Decimal::from_str("33000").unwrap(), None, None)
            .await;
        assert_eq!(engine.position("BTC-USD").await, Decimal::ZERO);
        assert_eq!(engine.realized_pnl_today().await, Decimal::from_str("40").unwrap());
    }

    #[tokio::test]
    async fn apply_fill_realizes_a_loss_and_flips_the_position_on_overshoot() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;

        // Sell 0.02 @ 29000: closes the 0.01 long at a loss of (29000-30000)*0.01 = -10,
        // then flips to a fresh 0.01 short at an entry price of 29000.
        engine
            .apply_fill("BTC-USD", OrderSide::Sell, Decimal::from_str("0.02").unwrap(), Decimal::from_str("29000").unwrap(), None, None)
            .await;
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("-0.01").unwrap());
        assert_eq!(engine.realized_pnl_today().await, Decimal::from_str("-10").unwrap());
    }

    #[tokio::test]
    async fn kill_switch_rejects_orders_after_daily_loss_limit_hit() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        // max daily loss in test_config is 500. Realize a 600 loss: go long
        // 1.0 @ 30000, then close it at 29400 (loss of 600).
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("1.0").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;
        engine
            .apply_fill("BTC-USD", OrderSide::Sell, Decimal::from_str("1.0").unwrap(), Decimal::from_str("29400").unwrap(), None, None)
            .await;
        assert_eq!(engine.realized_pnl_today().await, Decimal::from_str("-600").unwrap());

        let order = buy_order("BTC-USD", "0.01", Some("30000"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn kill_switch_does_not_trip_on_gains() {
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;
        engine
            .apply_fill("BTC-USD", OrderSide::Sell, Decimal::from_str("0.01").unwrap(), Decimal::from_str("31000").unwrap(), None, None)
            .await;
        assert_eq!(engine.realized_pnl_today().await, Decimal::from_str("10").unwrap());

        let order = buy_order("BTC-USD", "0.01", Some("30000"));
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    #[tokio::test]
    async fn apply_fill_persists_position_and_survives_a_fresh_engine() {
        let store = Arc::new(crate::persistence::Store::open_in_memory().unwrap());
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine.attach_store(store.clone()).await.unwrap();

        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("30000").unwrap(), None, None)
            .await;
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("0.01").unwrap());

        // A brand new engine attached to the same store picks up the
        // persisted position without ever seeing the fill directly.
        let restarted = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        restarted.attach_store(store).await.unwrap();
        assert_eq!(restarted.position("BTC-USD").await, Decimal::from_str("0.01").unwrap());
    }

    #[tokio::test]
    async fn apply_fill_deduplicates_by_exec_id_via_the_store() {
        let store = Arc::new(crate::persistence::Store::open_in_memory().unwrap());
        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine.attach_store(store).await.unwrap();

        let applied = engine
            .apply_fill(
                "BTC-USD",
                OrderSide::Buy,
                Decimal::from_str("0.01").unwrap(),
                Decimal::from_str("30000").unwrap(),
                Some("EXEC-1"),
                Some("co-1"),
            )
            .await;
        assert!(applied);
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("0.01").unwrap());

        // Same exec_id redelivered (e.g. after a reconnect) must not be
        // applied a second time.
        let applied_again = engine
            .apply_fill(
                "BTC-USD",
                OrderSide::Buy,
                Decimal::from_str("0.01").unwrap(),
                Decimal::from_str("30000").unwrap(),
                Some("EXEC-1"),
                Some("co-1"),
            )
            .await;
        assert!(!applied_again);
        assert_eq!(engine.position("BTC-USD").await, Decimal::from_str("0.01").unwrap());
    }

    #[tokio::test]
    async fn attach_store_ignores_a_kill_switch_counter_from_a_previous_day() {
        let store = crate::persistence::Store::open_in_memory().unwrap();
        // A day index that's certainly not today.
        store.save_kill_switch_state(1, Decimal::from_str("-999").unwrap()).unwrap();

        let engine = RiskEngine::new(Arc::new(test_config()), new_shared_books());
        engine.attach_store(Arc::new(store)).await.unwrap();

        assert_eq!(engine.realized_pnl_today().await, Decimal::ZERO);
    }
}
