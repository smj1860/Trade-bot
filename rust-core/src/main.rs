mod config;
mod kraken;
mod kraken_private_ws;
mod kraken_rest;
mod market_data;
mod order;
mod orderbook;
mod proto;
mod risk;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use tonic::transport::Server;

use config::Config;
use kraken_rest::{KrakenCredentials, KrakenRestClient};
use market_data::MarketDataServiceImpl;
use order::OrderServiceImpl;
use orderbook::new_shared_books;
use proto::pb::market_data_service_server::MarketDataServiceServer;
use proto::pb::order_service_server::OrderServiceServer;
use risk::RiskEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // tokio-tungstenite pulls in rustls without picking a crypto backend by
    // default when more than one is reachable in the dependency tree; this
    // has to happen once, before any TLS connection (Kraken's wss:// URL)
    // is attempted, or it panics deep inside the first connect call.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    let config_path =
        std::env::var("CONFIG_PATH").unwrap_or_else(|_| "../config/config.example.toml".to_string());
    let config = Arc::new(Config::load(&config_path)?);
    tracing::info!(
        path = %config_path,
        exchanges = config.exchanges.len(),
        symbols = config.symbols.len(),
        dry_run = config.execution.dry_run,
        "config loaded"
    );

    // Single broadcast channel carries every market data event; the gRPC
    // service filters per-subscriber by symbol. Capacity is generous
    // headroom for a burst across all symbols between subscriber polls.
    let (tx, _rx) = broadcast::channel(4096);

    // Shared, queryable order book state. The risk engine reads current
    // prices from this directly — it's separate from the broadcast stream
    // above, which is a one-way feed of changes, not something you can
    // query "what's the price right now" against.
    let books = new_shared_books();

    for exchange in &config.exchanges {
        if !exchange.enabled {
            continue;
        }
        let symbols = config.symbols_for_exchange(&exchange.name);
        match exchange.name.as_str() {
            "kraken" => {
                let exchange = exchange.clone();
                let tx = tx.clone();
                let books = books.clone();
                tokio::spawn(kraken::run(exchange, symbols, tx, books));
            }
            other => {
                tracing::warn!(exchange = other, "no ingestion adapter implemented for this exchange yet");
            }
        }
    }

    let risk_engine = Arc::new(RiskEngine::new(config.clone(), books));

    // An execution client is only built for an exchange when credentials
    // are actually present. No credentials -> no entry in the map, and a
    // risk-approved order for that exchange comes back as an honest
    // "no execution client configured" rejection instead of a crash or a
    // faked success. Credentials are read from env vars, never config
    // files, so they're never accidentally committed alongside
    // config.toml.
    let mut execution_clients: HashMap<String, KrakenRestClient> = HashMap::new();
    for exchange in &config.exchanges {
        if exchange.name != "kraken" || !exchange.enabled {
            continue;
        }
        match (std::env::var("KRAKEN_API_KEY"), std::env::var("KRAKEN_API_SECRET")) {
            (Ok(api_key), Ok(api_secret)) if !api_key.is_empty() && !api_secret.is_empty() => {
                execution_clients.insert(
                    exchange.name.clone(),
                    KrakenRestClient::new(exchange.rest_url.clone(), KrakenCredentials { api_key, api_secret }),
                );
                tracing::info!(exchange = %exchange.name, "execution client configured");
            }
            _ => {
                tracing::warn!(
                    exchange = %exchange.name,
                    "KRAKEN_API_KEY / KRAKEN_API_SECRET not set — approved orders on this exchange \
                     will be rejected with 'no execution client configured' rather than sent anywhere"
                );
            }
        }
    }
    // Real fills/status changes come from Kraken's private (authenticated)
    // WebSocket feed — entirely separate from the public market-data feed
    // above. One task per exchange that actually has an execution client
    // configured (no credentials -> no client -> nothing to authenticate
    // as, so no point starting this). All updates funnel onto one
    // broadcast channel that OrderServiceImpl.StreamOrderUpdates fans out
    // from, filtered per-subscriber by strategy_id via `strategy_registry`
    // (populated in order.rs when an order is actually sent to Kraken).
    let (order_updates_tx, _rx) = broadcast::channel(4096);
    let strategy_registry: order::StrategyRegistry = Arc::new(Mutex::new(HashMap::new()));
    for exchange in &config.exchanges {
        let Some(client) = execution_clients.get(&exchange.name) else {
            continue;
        };
        let symbols = config.symbols_for_exchange(&exchange.name);
        let client = client.clone();
        let order_updates_tx = order_updates_tx.clone();
        tokio::spawn(kraken_private_ws::run(symbols, client, order_updates_tx));
    }

    let execution_clients = Arc::new(execution_clients);

    let addr = "0.0.0.0:50051".parse()?;
    tracing::info!(%addr, "trading-core gRPC server starting");

    Server::builder()
        .add_service(MarketDataServiceServer::new(MarketDataServiceImpl::new(tx)))
        .add_service(OrderServiceServer::new(OrderServiceImpl::new(
            risk_engine,
            config,
            execution_clients,
            order_updates_tx,
            strategy_registry,
        )))
        .serve(addr)
        .await?;

    Ok(())
}
