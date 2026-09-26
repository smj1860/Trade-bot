use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::config::Config;
use crate::kraken_rest::{AddOrderOutcome, AddOrderRequest, KrakenRestClient, OrderSide as KrakenSide};
use crate::proto::pb::{
    order_service_server::OrderService, OrderRequest, OrderSide, OrderStatus, OrderType, OrderUpdate,
    StreamOrderUpdatesRequest,
};
use crate::risk::{RiskEngine, RiskVerdict};

/// Receives order requests from Python and runs them through the risk
/// engine, then — if the order is Approved — forwards it to the
/// exchange's REST execution client.
///
/// A rejection (risk or exchange) is a normal, successful RPC response;
/// only a transport/plumbing failure becomes a gRPC-level error. If no
/// execution client is configured for the order's exchange (no API
/// credentials were set at startup), that is reported honestly rather
/// than silently dropped or faked as a success.
pub struct OrderServiceImpl {
    risk: Arc<RiskEngine>,
    config: Arc<Config>,
    execution_clients: Arc<HashMap<String, KrakenRestClient>>,
}

impl OrderServiceImpl {
    pub fn new(
        risk: Arc<RiskEngine>,
        config: Arc<Config>,
        execution_clients: Arc<HashMap<String, KrakenRestClient>>,
    ) -> Self {
        Self {
            risk,
            config,
            execution_clients,
        }
    }

    /// Executes an already-risk-approved order against the exchange it
    /// was requested on. Returns the `OrderUpdate` to send back to
    /// Python, or a gRPC `Status` only for genuine plumbing failures
    /// (no client configured, unmapped symbol, network error) — a
    /// legitimate exchange-side rejection is still an `Ok(OrderUpdate)`.
    async fn execute(&self, order: &OrderRequest) -> Result<OrderUpdate, Status> {
        let Some(client) = self.execution_clients.get(&order.exchange) else {
            return Err(Status::failed_precondition(format!(
                "order passed risk checks, but no execution client is configured for exchange '{}' \
                 (likely missing API credentials at startup) — refusing to claim it was submitted",
                order.exchange
            )));
        };

        let Some(symbol_cfg) = self
            .config
            .symbols
            .iter()
            .find(|s| s.symbol == order.symbol && s.exchange == order.exchange)
        else {
            // Should be unreachable — the risk engine already validated this
            // symbol is configured — but fail closed rather than panic.
            return Err(Status::internal(format!(
                "no config entry for symbol {} on exchange {} despite passing risk checks",
                order.symbol, order.exchange
            )));
        };

        let side = match OrderSide::try_from(order.side) {
            Ok(OrderSide::Buy) => KrakenSide::Buy,
            Ok(OrderSide::Sell) => KrakenSide::Sell,
            _ => {
                return Err(Status::internal(
                    "order side was neither BUY nor SELL despite passing risk checks",
                ))
            }
        };

        let order_type = match OrderType::try_from(order.r#type) {
            Ok(OrderType::Limit) => "limit",
            Ok(OrderType::Market) => "market",
            _ => {
                return Err(Status::invalid_argument(
                    "order type must be LIMIT or MARKET",
                ))
            }
        };

        let Some(quantity) = order.quantity.as_ref() else {
            return Err(Status::internal("order had no quantity despite passing risk checks"));
        };

        let price = if order_type == "limit" {
            let Some(p) = order.limit_price.as_ref() else {
                return Err(Status::invalid_argument("limit order requires a limit_price"));
            };
            Some(p.value.clone())
        } else {
            None
        };

        let add_order = AddOrderRequest {
            pair: symbol_cfg.rest_native_symbol.clone(),
            side,
            order_type,
            volume: quantity.value.clone(),
            price,
            client_order_id: order.client_order_id.clone(),
            validate: self.config.execution.dry_run,
        };

        tracing::info!(
            client_order_id = %order.client_order_id,
            symbol = %order.symbol,
            pair = %add_order.pair,
            dry_run = self.config.execution.dry_run,
            "sending order to Kraken"
        );

        let outcome = client
            .add_order(&add_order)
            .await
            .map_err(|e| Status::unavailable(format!("failed to reach Kraken: {e}")))?;

        match outcome {
            AddOrderOutcome::Accepted { exchange_order_id } => {
                let dry_run_note = if self.config.execution.dry_run {
                    " (validate=true: Kraken accepted the request but did not place it)"
                } else {
                    ""
                };
                tracing::info!(
                    client_order_id = %order.client_order_id,
                    exchange_order_id = ?exchange_order_id,
                    "Kraken accepted order{}",
                    dry_run_note
                );
                Ok(OrderUpdate {
                    client_order_id: order.client_order_id.clone(),
                    exchange_order_id: exchange_order_id.unwrap_or_default(),
                    symbol: order.symbol.clone(),
                    status: OrderStatus::Accepted as i32,
                    reject_reason: String::new(),
                    filled_quantity: None,
                    remaining_quantity: Some(quantity.clone()),
                    avg_fill_price: None,
                    timestamp_ns: now_ns(),
                })
            }
            AddOrderOutcome::KrakenRejected { messages } => {
                let reason = messages.join("; ");
                tracing::warn!(
                    client_order_id = %order.client_order_id,
                    reason = %reason,
                    "Kraken rejected order"
                );
                Ok(OrderUpdate {
                    client_order_id: order.client_order_id.clone(),
                    exchange_order_id: String::new(),
                    symbol: order.symbol.clone(),
                    status: OrderStatus::Rejected as i32,
                    reject_reason: reason,
                    filled_quantity: None,
                    remaining_quantity: None,
                    avg_fill_price: None,
                    timestamp_ns: now_ns(),
                })
            }
        }
    }
}

#[tonic::async_trait]
impl OrderService for OrderServiceImpl {
    type StreamOrderUpdatesStream =
        Pin<Box<dyn Stream<Item = Result<OrderUpdate, Status>> + Send + 'static>>;

    async fn submit_order(
        &self,
        request: Request<OrderRequest>,
    ) -> Result<Response<OrderUpdate>, Status> {
        let order = request.into_inner();

        match self.risk.evaluate(&order).await {
            RiskVerdict::Rejected(reason) => {
                tracing::warn!(
                    client_order_id = %order.client_order_id,
                    symbol = %order.symbol,
                    reason = %reason,
                    "order rejected by risk engine"
                );
                Ok(Response::new(OrderUpdate {
                    client_order_id: order.client_order_id,
                    exchange_order_id: String::new(),
                    symbol: order.symbol,
                    status: OrderStatus::Rejected as i32,
                    reject_reason: reason,
                    filled_quantity: None,
                    remaining_quantity: None,
                    avg_fill_price: None,
                    timestamp_ns: now_ns(),
                }))
            }
            RiskVerdict::Approved => self.execute(&order).await.map(Response::new),
        }
    }

    async fn stream_order_updates(
        &self,
        request: Request<StreamOrderUpdatesRequest>,
    ) -> Result<Response<Self::StreamOrderUpdatesStream>, Status> {
        let strategy_id = request.into_inner().strategy_id;
        tracing::info!(%strategy_id, "order update subscription received");

        let stream = tokio_stream::empty();
        Ok(Response::new(Box::pin(stream)))
    }
}

fn now_ns() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}
