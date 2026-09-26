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
use std::time::{Duration, Instant};

use rust_decimal::Decimal;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::orderbook::SharedBooks;
use crate::proto::pb::{OrderRequest, OrderSide};

const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskVerdict {
    Approved,
    Rejected(String),
}

pub struct RiskEngine {
    config: Arc<Config>,
    books: SharedBooks,
    /// Net position per normalized symbol, in base-asset units. Positive =
    /// long. Updated only on confirmed fills (not wired up yet — no
    /// exchange execution client exists, so this stays at zero for every
    /// symbol in this build; the position-limit checks below are real and
    /// tested, they just have nothing but zero to start from).
    positions: Mutex<HashMap<String, Decimal>>,
    recent_order_times: Mutex<VecDeque<Instant>>,
}

impl RiskEngine {
    pub fn new(config: Arc<Config>, books: SharedBooks) -> Self {
        Self {
            config,
            books,
            positions: Mutex::new(HashMap::new()),
            recent_order_times: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn evaluate(&self, order: &OrderRequest) -> RiskVerdict {
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
            positions.get(&order.symbol).copied().unwrap_or(Decimal::ZERO)
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
            if symbol == exclude_symbol || position.is_zero() {
                continue;
            }
            if let Some(book) = books.get(symbol) {
                if let Some((price, _)) = book.best_bid() {
                    total += (*position * price).abs();
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
}
