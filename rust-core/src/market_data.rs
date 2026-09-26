use std::collections::HashSet;
use std::pin::Pin;

use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::proto::pb::{
    market_data_service_server::MarketDataService, MarketDataEvent, SubscribeRequest,
};

/// Streams order book updates / metrics to subscribed Python clients.
/// Actual market data is published onto `tx` by the exchange ingestion
/// tasks (see kraken.rs); this service just fans it out, filtered to the
/// symbols each caller asked for.
#[derive(Debug, Clone)]
pub struct MarketDataServiceImpl {
    tx: broadcast::Sender<MarketDataEvent>,
}

impl MarketDataServiceImpl {
    pub fn new(tx: broadcast::Sender<MarketDataEvent>) -> Self {
        Self { tx }
    }
}

#[tonic::async_trait]
impl MarketDataService for MarketDataServiceImpl {
    type SubscribeMarketDataStream =
        Pin<Box<dyn Stream<Item = Result<MarketDataEvent, Status>> + Send + 'static>>;

    async fn subscribe_market_data(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeMarketDataStream>, Status> {
        let symbols: HashSet<String> = request.into_inner().symbols.into_iter().collect();
        tracing::info!(?symbols, "market data subscription received");

        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(move |item| match item {
            Ok(event) => {
                let matches = symbols.is_empty()
                    || event_symbol(&event).map(|s| symbols.contains(s)).unwrap_or(false);
                matches.then_some(Ok(event))
            }
            Err(BroadcastStreamRecvError::Lagged(skipped)) => {
                // A slow subscriber missed some updates. Don't kill the
                // stream over it — order book snapshots self-correct on
                // the next message; just log and carry on.
                tracing::warn!(skipped, "subscriber lagged, dropped events");
                None
            }
        });

        Ok(Response::new(Box::pin(stream)))
    }
}

fn event_symbol(event: &MarketDataEvent) -> Option<&str> {
    use crate::proto::pb::market_data_event::Event;
    match &event.event {
        Some(Event::OrderBookUpdate(u)) => Some(&u.symbol),
        Some(Event::Metric(m)) => Some(&m.symbol),
        None => None,
    }
}
