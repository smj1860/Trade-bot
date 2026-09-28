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
//! 5. Dynamic slippage guard: simulated fill price vs. mid, walking real
//!    book depth (see guardrails::simulate_fill_price)
//! 6. Dynamic spread guard: current spread vs. its own rolling average
//!    (see guardrails::rolling_average_spread)
//! 7. Volatility circuit breaker: if tripped, only reduce-only/flat orders
//!    are approved (see guardrails::assess_volatility and
//!    `check_volatility_breaker` below)
//! 8. Projected position after this order <= per-symbol max_position_usd
//! 9. Projected combined portfolio exposure <= global max_total_position_usd

use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rust_decimal::Decimal;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::guardrails;
use crate::orderbook::SharedBooks;
use crate::persistence::{FillRecord, Store};
use crate::proto::pb::{OrderRequest, OrderSide, OrderType};

/// Baseline window used for both the spread guard's rolling average and
/// the volatility breaker's baseline distribution — matches
/// `orderbook.rs::HISTORY_MAX_AGE`, the longest either guardrail could
/// possibly look back regardless of what's requested.
const GUARDRAIL_HISTORY_WINDOW: Duration = Duration::from_secs(3600);

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
    /// Per-symbol volatility-breaker freeze deadline: while `Instant::now()
    /// < deadline`, only reduce-only/flat orders for that symbol are
    /// approved (see `check_volatility_breaker`). Deliberately in-memory
    /// only, not persisted — a freeze is a short-lived reaction to a live
    /// market condition, not state that should outlive a restart the way
    /// positions and the kill switch counter do.
    vol_freeze_until: Mutex<HashMap<String, Instant>>,
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
            vol_freeze_until: Mutex::new(HashMap::new()),
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

        let signed_qty = if side == OrderSide::Buy { qty } else { -qty };
        let current_position = {
            let positions = self.positions.lock().await;
            positions.get(&order.symbol).map(|p| p.qty).unwrap_or(Decimal::ZERO)
        };

        if let Some(reason) = self.check_slippage_and_spread(order, side, qty, price).await {
            return RiskVerdict::Rejected(reason);
        }

        if let Some(reason) = self.check_volatility_breaker(order, current_position, signed_qty).await {
            return RiskVerdict::Rejected(reason);
        }

        let Ok(max_position_usd) = Decimal::from_str(&symbol_cfg.risk.max_position_usd) else {
            return RiskVerdict::Rejected("invalid config: max_position_usd".to_string());
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

    /// Dynamic slippage + spread guardrails (see guardrails.rs). Both are
    /// no-ops (approve) when there's no live book for this symbol yet, or
    /// not enough rolling history for the spread check specifically —
    /// "can't judge yet" is not the same as "reject," and every other
    /// check in `evaluate` already has its own independent protection
    /// (order size, notional, position caps) that doesn't depend on this
    /// history existing.
    ///
    /// Institutional audit Phase 1.3: a LIMIT order doesn't take liquidity
    /// at submission time the way a MARKET order does — it rests on the
    /// book, waiting to be someone else's counterparty. Walking the book
    /// with `simulate_fill_price` and rejecting on "expected slippage" for
    /// a resting order was evaluating a fill that was never going to
    /// happen at submission time; the fix isn't to loosen the check, it's
    /// to not run a market-fill simulation against an order that isn't a
    /// market fill. What a LIMIT order needs instead is a guarantee it
    /// won't silently become a taker: `check_post_only_would_not_cross`
    /// below rejects a limit price that would immediately match the
    /// opposite side, independent of (and before) Kraken's own
    /// `oflags=post` doing the same thing exchange-side (see
    /// kraken_rest.rs's AddOrderRequest::post_only) — belt and suspenders,
    /// since this check runs before an order is ever sent and gives a
    /// specific, attributable rejection reason rather than a bare Kraken
    /// error. The spread guard still applies to both order types: a
    /// dislocated book is a reason for caution about resting an order in
    /// it too, not just about paying more to cross it.
    async fn check_slippage_and_spread(
        &self,
        order: &OrderRequest,
        side: OrderSide,
        qty: Decimal,
        mid_or_limit_price: Decimal,
    ) -> Option<String> {
        let books = self.books.lock().await;
        let Some(book) = books.get(&order.symbol) else {
            return None;
        };
        let (Some((best_bid, _)), Some((best_ask, _))) = (book.best_bid(), book.best_ask()) else {
            return None;
        };
        if best_ask <= best_bid {
            return None; // crossed/locked book — nothing sane to check against
        }
        let mid = (best_bid + best_ask) / Decimal::TWO;
        let is_limit_order = matches!(OrderType::try_from(order.r#type), Ok(OrderType::Limit));

        if is_limit_order {
            if let Some(reason) =
                check_post_only_would_not_cross(&order.symbol, side, mid_or_limit_price, best_bid, best_ask)
            {
                return Some(reason);
            }
        } else {
            // Slippage: walk the side of the book this order would
            // actually consume (asks for a buy, bids for a sell) — the
            // top-of-book price alone understates cost for anything
            // bigger than the best level's own quantity. Only meaningful
            // for a MARKET order, which really does consume this
            // liquidity right now — see this method's docs.
            let levels = match side {
                OrderSide::Buy => book.ask_levels(50),
                OrderSide::Sell => book.bid_levels(50),
                _ => Vec::new(),
            };
            if let Some(fill_price) = guardrails::simulate_fill_price(&levels, qty) {
                let Ok(max_slippage) = Decimal::from_str(&self.config.risk.global.max_slippage_pct) else {
                    return Some("invalid config: max_slippage_pct".to_string());
                };
                let slippage = guardrails::slippage_fraction(mid, fill_price);
                if slippage > max_slippage {
                    return Some(format!(
                        "expected slippage {slippage} (fill price {fill_price} vs mid {mid}) exceeds \
                         max_slippage_pct {max_slippage} for {} — book is too thin for this order size",
                        order.symbol
                    ));
                }
            } else {
                // Not enough depth on the relevant side to fill this order
                // at all — a market order this large has no honest
                // "expected fill price" to check, which is itself the
                // liquidity-vacuum condition this guardrail exists to
                // catch.
                return Some(format!(
                    "order book for {} does not have enough depth on the {side:?} side to fill a {qty} order \
                     — refusing to estimate slippage against a book this thin",
                    order.symbol
                ));
            }
        }

        // Spread: current spread vs. this book's own rolling average,
        // scaled by a configured multiplier — a threshold that adapts to
        // each symbol's normal spread rather than one flat number.
        let current_spread = guardrails::spread_fraction(best_bid, best_ask, mid_or_limit_price.max(mid));
        let history = book.recent_samples(GUARDRAIL_HISTORY_WINDOW);
        if let Some(avg_spread) =
            guardrails::rolling_average_spread(&history, self.config.risk.global.min_spread_samples)
        {
            let Ok(multiplier) = Decimal::from_str(&self.config.risk.global.spread_multiplier) else {
                return Some("invalid config: spread_multiplier".to_string());
            };
            let threshold = avg_spread * multiplier;
            if current_spread > threshold {
                return Some(format!(
                    "current spread {current_spread} exceeds {multiplier}x its rolling average \
                     ({avg_spread}, threshold {threshold}) for {} — book looks dislocated from its own recent normal",
                    order.symbol
                ));
            }
        }

        None
    }

    /// Micro-volatility circuit breaker. If a fresh spike trips it, this
    /// symbol freezes into reduce-only/flat for `vol_circuit_breaker_freeze_secs`
    /// (see `vol_freeze_until`): while frozen, an order is only approved
    /// if it would not increase the position's absolute size and would
    /// not flip its sign — i.e. it can only shrink or close the existing
    /// position, never grow or reverse it. A flat symbol (no position)
    /// approves nothing at all while frozen, matching the "Reduce-Only /
    /// Flat" framing this guardrail was asked for in.
    async fn check_volatility_breaker(
        &self,
        order: &OrderRequest,
        current_position: Decimal,
        signed_qty: Decimal,
    ) -> Option<String> {
        let now = Instant::now();

        {
            let freezes = self.vol_freeze_until.lock().await;
            if let Some(&deadline) = freezes.get(&order.symbol) {
                if now < deadline {
                    return self.reduce_only_verdict(order, current_position, signed_qty, deadline);
                }
            }
        }

        let history = {
            let books = self.books.lock().await;
            let Some(book) = books.get(&order.symbol) else {
                return None;
            };
            book.recent_samples(GUARDRAIL_HISTORY_WINDOW)
        };

        let short_window = Duration::from_secs(self.config.risk.global.vol_short_window_secs);
        let bucket = Duration::from_secs(self.config.risk.global.vol_baseline_bucket_secs);
        let Some(reading) = guardrails::assess_volatility(&history, now, short_window, bucket) else {
            return None; // not enough history to judge yet — approve, don't guess
        };

        let Ok(threshold_stddev) = self.config.risk.global.vol_circuit_breaker_stddev.parse::<f64>() else {
            return Some("invalid config: vol_circuit_breaker_stddev".to_string());
        };

        if reading.zscore <= threshold_stddev {
            return None;
        }

        let freeze_secs = self.config.risk.global.vol_circuit_breaker_freeze_secs;
        let deadline = now + Duration::from_secs(freeze_secs);
        self.vol_freeze_until.lock().await.insert(order.symbol.clone(), deadline);
        tracing::warn!(
            symbol = %order.symbol,
            zscore = reading.zscore,
            short_window_vol = reading.short_window_vol,
            baseline_mean = reading.baseline_mean,
            baseline_stddev = reading.baseline_stddev,
            freeze_secs,
            "volatility circuit breaker tripped — freezing to reduce-only/flat"
        );

        self.reduce_only_verdict(order, current_position, signed_qty, deadline)
    }

    /// Shared reduce-only check for `check_volatility_breaker`, used both
    /// on a freshly-tripped breaker and on one still inside an earlier
    /// freeze window.
    fn reduce_only_verdict(
        &self,
        order: &OrderRequest,
        current_position: Decimal,
        signed_qty: Decimal,
        deadline: Instant,
    ) -> Option<String> {
        let new_qty = current_position + signed_qty;
        let increases_size = new_qty.abs() > current_position.abs();
        let flips_sign = !current_position.is_zero() && !new_qty.is_zero() && sign_of(new_qty) != sign_of(current_position);

        if increases_size || flips_sign {
            let remaining = deadline.saturating_duration_since(Instant::now()).as_secs();
            return Some(format!(
                "volatility circuit breaker is active for {} (reduce-only/flat for another {remaining}s) — \
                 this order would increase or flip the position, which is not allowed while frozen",
                order.symbol
            ));
        }
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

/// Institutional audit Phase 1.3's client-side post-only guarantee: a
/// limit order that would already cross the book at submission time isn't
/// a resting maker order, it's a market order wearing a limit price. A
/// buy crosses once its price reaches the best ask (it would match, not
/// rest, at that price and anything better); a sell crosses once its
/// price reaches the best bid, symmetrically. Pure and synchronous so
/// it's directly unit-testable without spinning up a book/engine — see
/// `check_slippage_and_spread`'s docs for why this replaces the
/// market-fill slippage simulation for a LIMIT order.
fn check_post_only_would_not_cross(
    symbol: &str,
    side: OrderSide,
    limit_price: Decimal,
    best_bid: Decimal,
    best_ask: Decimal,
) -> Option<String> {
    match side {
        OrderSide::Buy if limit_price >= best_ask => Some(format!(
            "limit price {limit_price} would cross the book (best ask {best_ask}) for {symbol} — \
             a post-only/maker order must rest below the best ask, not take it"
        )),
        OrderSide::Sell if limit_price <= best_bid => Some(format!(
            "limit price {limit_price} would cross the book (best bid {best_bid}) for {symbol} — \
             a post-only/maker order must rest above the best bid, not take it"
        )),
        _ => None,
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
                    max_slippage_pct: "0.005".into(),
                    spread_multiplier: "3.0".into(),
                    min_spread_samples: 30,
                    vol_circuit_breaker_stddev: "4.0".into(),
                    vol_short_window_secs: 60,
                    vol_baseline_bucket_secs: 60,
                    vol_circuit_breaker_freeze_secs: 300,
                },
            },
            execution: crate::config::ExecutionConfig {
                dry_run: true,
                rate_limit: crate::config::RateLimitConfig::default(),
            },
            persistence: crate::config::PersistenceConfig { database_path: ":memory:".to_string() },
            dead_man_switch: crate::config::DeadManSwitchConfig::default(),
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
            post_only: false,
        }
    }

    fn sell_order(symbol: &str, qty: &str, limit_price: Option<&str>) -> OrderRequest {
        OrderRequest {
            client_order_id: "test-1".into(),
            symbol: symbol.into(),
            exchange: "kraken".into(),
            side: OrderSide::Sell as i32,
            r#type: if limit_price.is_some() { OrderType::Limit as i32 } else { OrderType::Market as i32 },
            quantity: Some(PbDecimal { value: qty.into() }),
            limit_price: limit_price.map(|p| PbDecimal { value: p.into() }),
            strategy_id: "test-strategy".into(),
            post_only: false,
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

    // --- Slippage / spread / volatility guardrails (guardrails.rs, wired
    // in via check_slippage_and_spread / check_volatility_breaker) ---

    #[tokio::test]
    async fn rejects_order_when_book_too_thin_to_estimate_slippage() {
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            // Only 0.01 available at the best ask — far less than the 1.0
            // this order wants to buy.
            book.apply_snapshot(
                vec![(Decimal::from_str("29999").unwrap(), Decimal::from_str("1").unwrap())],
                vec![(Decimal::from_str("30001").unwrap(), Decimal::from_str("0.01").unwrap())],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(test_config()), books);
        let order = buy_order("BTC-USD", "1.0", None); // market order, wants more depth than exists
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn rejects_order_when_expected_slippage_exceeds_configured_max() {
        let mut config = test_config();
        config.risk.global.max_slippage_pct = "0.0001".to_string(); // very tight, 0.01%
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            // Buying 1.0: 0.5 @ 30001, then 0.5 @ 30500 -> vwap well above
            // mid (~30000), tripping even a modest slippage cap.
            book.apply_snapshot(
                vec![(Decimal::from_str("29999").unwrap(), Decimal::from_str("1").unwrap())],
                vec![
                    (Decimal::from_str("30001").unwrap(), Decimal::from_str("0.5").unwrap()),
                    (Decimal::from_str("30500").unwrap(), Decimal::from_str("0.5").unwrap()),
                ],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(config), books);
        let order = buy_order("BTC-USD", "1.0", None);
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn approves_order_within_configured_slippage_tolerance() {
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
        let order = buy_order("BTC-USD", "0.01", None); // fills entirely at best ask, negligible slippage
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    #[tokio::test]
    async fn rejects_order_when_spread_blows_out_past_its_rolling_average() {
        let mut config = test_config();
        config.risk.global.min_spread_samples = 3; // fast to build up in a test
        config.risk.global.max_slippage_pct = "1.0".to_string(); // isolate the spread guard specifically
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            // A handful of narrow-spread snapshots to build up a tight
            // rolling average (~0.0067% spread each). Each call fully
            // replaces the book (unlike apply_update, which only adds/
            // changes individual levels) so top-of-book actually moves.
            for _ in 0..5 {
                book.apply_snapshot(
                    vec![(Decimal::from_str("29999").unwrap(), Decimal::from_str("1").unwrap())],
                    vec![(Decimal::from_str("30001").unwrap(), Decimal::from_str("1").unwrap())],
                );
            }
            // ...then the book blows out to a much wider spread right
            // before the order is evaluated.
            book.apply_snapshot(
                vec![(Decimal::from_str("29000").unwrap(), Decimal::from_str("1").unwrap())],
                vec![(Decimal::from_str("31000").unwrap(), Decimal::from_str("1").unwrap())],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(config), books);
        let order = buy_order("BTC-USD", "0.01", None);
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn spread_guard_does_not_apply_before_enough_history_accumulates() {
        // Same wide spread as above, but default min_spread_samples (30)
        // means this book's single snapshot hasn't built up a baseline
        // yet — the dynamic spread check should simply not apply, not
        // reject over an assumed baseline of zero. Slippage is also
        // loosened here so this test is isolated to the spread guard.
        let mut config = test_config();
        config.risk.global.max_slippage_pct = "1.0".to_string();
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            book.apply_snapshot(
                vec![(Decimal::from_str("29000").unwrap(), Decimal::from_str("1").unwrap())],
                vec![(Decimal::from_str("31000").unwrap(), Decimal::from_str("1").unwrap())],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(config), books);
        let order = buy_order("BTC-USD", "0.01", None);
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    // --- Institutional audit Phase 1.3: LIMIT orders skip the market-fill
    // slippage simulation and instead get a post-only non-crossing check
    // (check_post_only_would_not_cross) ---

    #[test]
    fn post_only_check_approves_a_buy_that_rests_below_the_best_ask() {
        let d = |s: &str| Decimal::from_str(s).unwrap();
        assert_eq!(
            check_post_only_would_not_cross("BTC-USD", OrderSide::Buy, d("29999"), d("29999"), d("30001")),
            None
        );
    }

    #[test]
    fn post_only_check_rejects_a_buy_priced_at_or_through_the_best_ask() {
        let d = |s: &str| Decimal::from_str(s).unwrap();
        assert!(check_post_only_would_not_cross("BTC-USD", OrderSide::Buy, d("30001"), d("29999"), d("30001"))
            .is_some());
        assert!(check_post_only_would_not_cross("BTC-USD", OrderSide::Buy, d("30500"), d("29999"), d("30001"))
            .is_some());
    }

    #[test]
    fn post_only_check_approves_a_sell_that_rests_above_the_best_bid() {
        let d = |s: &str| Decimal::from_str(s).unwrap();
        assert_eq!(
            check_post_only_would_not_cross("BTC-USD", OrderSide::Sell, d("30001"), d("29999"), d("30001")),
            None
        );
    }

    #[test]
    fn post_only_check_rejects_a_sell_priced_at_or_through_the_best_bid() {
        let d = |s: &str| Decimal::from_str(s).unwrap();
        assert!(check_post_only_would_not_cross("BTC-USD", OrderSide::Sell, d("29999"), d("29999"), d("30001"))
            .is_some());
        assert!(check_post_only_would_not_cross("BTC-USD", OrderSide::Sell, d("29500"), d("29999"), d("30001"))
            .is_some());
    }

    #[tokio::test]
    async fn a_limit_order_far_too_large_for_the_book_to_fill_as_a_market_order_is_still_approved() {
        // The whole point of this phase's fix: a resting LIMIT order isn't
        // taking this liquidity, so the "book too thin to simulate a fill"
        // rejection (see rejects_order_when_book_too_thin_to_estimate_slippage
        // above, which covers the MARKET-order case) must NOT apply here.
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            book.apply_snapshot(
                vec![(Decimal::from_str("29999").unwrap(), Decimal::from_str("1").unwrap())],
                vec![(Decimal::from_str("30001").unwrap(), Decimal::from_str("0.01").unwrap())],
            );
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(test_config()), books);
        // Resting well below the best ask, so it doesn't cross — but wants
        // more size (0.05, test_config's max_order_size) than the 0.01
        // resting at the best ask, which would have failed the old
        // market-fill-simulation check.
        let order = buy_order("BTC-USD", "0.05", Some("29999"));
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }

    #[tokio::test]
    async fn a_limit_order_priced_through_the_book_is_rejected_as_not_actually_maker() {
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
        // A "limit" buy priced above the best ask would just be a
        // disguised market order — the risk engine should catch this even
        // before it reaches Kraken's own oflags=post rejection.
        let order = buy_order("BTC-USD", "0.01", Some("31000"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    /// Builds a book whose history contains a calm baseline (5 separate
    /// 1-second-apart clusters, each with 2 ticks so `window_parkinson_vol`
    /// has a real range to compute — see `MIN_BASELINE_BUCKETS`) older
    /// than `short_window`, then a sharp spike within it. `OrderBook` has
    /// no injectable clock (see guardrails.rs's docs on why), so spacing
    /// the clusters into distinct baseline buckets means really waiting —
    /// this takes ~6.3s of real time, which is why only the tests that
    /// actually need a *tripped* breaker pay for it.
    async fn spiky_vol_book(symbol: &str) -> SharedBooks {
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            guard.insert(symbol.to_string(), OrderBook::new(symbol));
        }

        for cluster_price in ["100.00", "100.05", "100.10", "100.05", "100.00"] {
            {
                let mut guard = books.lock().await;
                let book = guard.get_mut(symbol).unwrap();
                for offset in ["0.00", "0.02"] {
                    let bid = Decimal::from_str(cluster_price).unwrap() + Decimal::from_str(offset).unwrap();
                    // apply_snapshot (not apply_update) so each tick fully
                    // replaces top-of-book rather than just adding another
                    // price level alongside whatever's already there —
                    // otherwise best_bid/best_ask stop moving and the book
                    // can end up crossed once the spike below is applied.
                    book.apply_snapshot(
                        vec![(bid, Decimal::from_str("1").unwrap())],
                        vec![(bid + Decimal::from_str("0.1").unwrap(), Decimal::from_str("1").unwrap())],
                    );
                }
            }
            tokio::time::sleep(Duration::from_millis(1050)).await;
        }

        // A sharp, wide-range spike, recorded well inside the 1s short
        // window relative to the evaluate() call right after this
        // function returns.
        {
            let mut guard = books.lock().await;
            let book = guard.get_mut(symbol).unwrap();
            for price in ["90", "110"] {
                let bid = Decimal::from_str(price).unwrap();
                book.apply_snapshot(
                    vec![(bid, Decimal::from_str("1").unwrap())],
                    vec![(bid + Decimal::from_str("0.1").unwrap(), Decimal::from_str("1").unwrap())],
                );
            }
        }
        books
    }

    fn vol_test_config() -> Config {
        let mut config = test_config();
        config.risk.global.vol_short_window_secs = 1;
        config.risk.global.vol_baseline_bucket_secs = 1;
        config.risk.global.vol_circuit_breaker_stddev = "1.0".to_string();
        config.risk.global.vol_circuit_breaker_freeze_secs = 2;
        // Loosen the other guardrails so this test is isolated to the
        // volatility breaker, not incidentally tripping slippage/spread
        // over the same wide-range spike.
        config.risk.global.max_slippage_pct = "1.0".to_string();
        config.risk.global.min_spread_samples = 1_000_000; // effectively disabled
        config
    }

    #[tokio::test]
    async fn volatility_breaker_trips_on_a_genuine_spike_and_freezes_new_exposure() {
        let books = spiky_vol_book("BTC-USD").await;
        let engine = RiskEngine::new(Arc::new(vol_test_config()), books);

        // Flat symbol, breaker tripped: even a small buy is rejected,
        // matching the "Reduce-Only / Flat" framing (nothing to reduce).
        let order = buy_order("BTC-USD", "0.001", Some("100"));
        assert!(matches!(engine.evaluate(&order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn volatility_breaker_allows_reduce_only_orders_while_frozen() {
        let books = spiky_vol_book("BTC-USD").await;
        let engine = RiskEngine::new(Arc::new(vol_test_config()), books);

        // Open a long position first (before evaluating against the spike
        // — apply_fill doesn't go through the breaker, only evaluate()
        // does).
        engine
            .apply_fill("BTC-USD", OrderSide::Buy, Decimal::from_str("0.01").unwrap(), Decimal::from_str("100").unwrap(), None, None)
            .await;

        // Trip the breaker via one rejected buy (which would grow the
        // position, so it's correctly rejected and also trips/records the
        // freeze for the next check).
        let growing_order = buy_order("BTC-USD", "0.005", Some("100"));
        assert!(matches!(engine.evaluate(&growing_order).await, RiskVerdict::Rejected(_)));

        // A sell that only shrinks the existing long (never flips it) is
        // still approved while frozen. Priced above the book's best bid
        // (110, per spiky_vol_book) so it doesn't trip the post-only
        // non-crossing check (institutional audit Phase 1.3) — this test
        // is about the volatility breaker, not post-only pricing.
        let reducing_order = sell_order("BTC-USD", "0.005", Some("111"));
        assert_eq!(engine.evaluate(&reducing_order).await, RiskVerdict::Approved);

        // But a sell large enough to flip the position to short is not.
        let flipping_order = sell_order("BTC-USD", "0.02", Some("111"));
        assert!(matches!(engine.evaluate(&flipping_order).await, RiskVerdict::Rejected(_)));
    }

    #[tokio::test]
    async fn volatility_breaker_does_not_trip_in_a_calm_market() {
        let books = new_shared_books();
        {
            let mut guard = books.lock().await;
            let mut book = OrderBook::new("BTC-USD");
            for _ in 0..8 {
                book.apply_snapshot(
                    vec![(Decimal::from_str("100.00").unwrap(), Decimal::from_str("1").unwrap())],
                    vec![(Decimal::from_str("100.10").unwrap(), Decimal::from_str("1").unwrap())],
                );
            }
            guard.insert("BTC-USD".to_string(), book);
        }
        let engine = RiskEngine::new(Arc::new(vol_test_config()), books);
        let order = buy_order("BTC-USD", "0.001", Some("100"));
        assert_eq!(engine.evaluate(&order).await, RiskVerdict::Approved);
    }
}
