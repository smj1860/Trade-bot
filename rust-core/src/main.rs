mod alerting;
mod checksum;
mod config;
mod guardrails;
mod heartbeat;
mod kraken;
mod kraken_private_ws;
mod kraken_rest;
mod market_data;
mod observability;
mod order;
mod orderbook;
mod performance;
mod persistence;
mod proto;
mod reconcile;
mod risk;
mod stop_loss;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use tonic::transport::Server;

use alerting::{possible_missed_fills_message, AlertSink};
use config::Config;
use heartbeat::HeartbeatMonitor;
use kraken_rest::{KrakenCredentials, KrakenRestClient};
use market_data::MarketDataServiceImpl;
use order::OrderServiceImpl;
use orderbook::new_shared_books;
use persistence::Store;
use proto::pb::market_data_service_server::MarketDataServiceServer;
use proto::pb::order_service_server::OrderServiceServer;
use observability::ObservabilityState;
use risk::RiskEngine;
use stop_loss::StopLossState;

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

    // Positions, the kill switch's daily PnL counter, and order/fill
    // history all live in SQLite so they survive a restart — see
    // persistence.rs. A database that fails to open is logged loudly but
    // doesn't stop the process: it degrades to the old in-memory-only
    // behavior (every symbol starts flat, the kill switch starts at zero)
    // rather than refusing to trade over what is, for now, an optional
    // piece of infrastructure.
    let store: Option<Arc<Store>> = match Store::open(&config.persistence.database_path) {
        Ok(store) => {
            tracing::info!(path = %config.persistence.database_path, "persistence store opened");
            Some(Arc::new(store))
        }
        Err(e) => {
            tracing::error!(
                path = %config.persistence.database_path,
                error = %e,
                "failed to open persistence store — running with in-memory-only state, \
                 positions and the kill switch will NOT survive a restart"
            );
            None
        }
    };
    if let Some(store) = &store {
        if let Err(e) = risk_engine.attach_store(store.clone()).await {
            tracing::error!(error = %e, "failed to load persisted risk state, continuing with in-memory-only state");
        }
    }

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
                    KrakenRestClient::with_rate_limit(
                        exchange.rest_url.clone(),
                        KrakenCredentials { api_key, api_secret },
                        &config.execution.rate_limit,
                    ),
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
    // A best-effort outbound alert sink (see alerting.rs) — no-ops if
    // ALERT_WEBHOOK_URL isn't set, same optional-infrastructure posture as
    // the persistence store and execution clients above.
    let alert_sink = AlertSink::from_env();
    tracing::info!(alerting_configured = alert_sink.is_configured(), "alert sink initialized");

    // Startup reconciliation: cross-check what's locally persisted as
    // "open" against what Kraken itself says is open, before this process
    // starts trusting that local picture again. Only possible when both a
    // store and a real execution client exist; a slow/unreachable Kraken
    // is bounded by a timeout so a network hiccup can't hang startup
    // forever — reconciliation not completing this run is logged loudly,
    // not fatal, same as every other optional-infrastructure failure here.
    if let Some(store) = &store {
        for exchange in &config.exchanges {
            let Some(client) = execution_clients.get(&exchange.name) else { continue };
            match tokio::time::timeout(
                std::time::Duration::from_secs(15),
                reconcile::reconcile_startup_state(store, client, &exchange.name),
            )
            .await
            {
                Ok(Ok(summary)) => {
                    tracing::info!(
                        exchange = %exchange.name,
                        locally_open_checked = summary.locally_open_checked,
                        confirmed_still_open = summary.confirmed_still_open,
                        closed_since_last_seen = summary.closed_since_last_seen,
                        possible_missed_fills = summary.possible_missed_fills,
                        kraken_open_with_no_local_record = summary.kraken_open_with_no_local_record,
                        "startup reconciliation complete"
                    );
                    // The one reconciliation outcome that needs a human to
                    // actually look, not just a log line — see alerting.rs
                    // and reconcile.rs's possible_missed_fills docs.
                    if summary.possible_missed_fills > 0 {
                        alert_sink
                            .send(&possible_missed_fills_message(&exchange.name, summary.possible_missed_fills))
                            .await;
                    }
                }
                Ok(Err(e)) => {
                    tracing::error!(
                        exchange = %exchange.name,
                        error = %e,
                        "startup reconciliation failed — continuing with whatever local order state was persisted, unverified"
                    );
                }
                Err(_) => {
                    tracing::error!(
                        exchange = %exchange.name,
                        "startup reconciliation timed out after 15s — continuing with whatever local order state was persisted, unverified"
                    );
                }
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
        let risk_engine = risk_engine.clone();
        let store = store.clone();
        tokio::spawn(kraken_private_ws::run(symbols, client, order_updates_tx, risk_engine, store));
    }

    let execution_clients = Arc::new(execution_clients);

    // Institutional audit Phase 1.1: dead-man's switch. `heartbeat_monitor`
    // is shared between the gRPC service (which records a heartbeat on
    // every SendHeartbeat call) and the watchdog task below (which reads
    // it); `order_service` itself is shared the same way so the watchdog
    // can submit a synthetic flatten order through the exact same
    // risk-evaluation path a real order takes. See heartbeat.rs.
    let heartbeat_monitor = Arc::new(HeartbeatMonitor::new());
    let order_service = OrderServiceImpl::new(
        risk_engine.clone(),
        config.clone(),
        execution_clients.clone(),
        order_updates_tx,
        strategy_registry,
        store.clone(),
        heartbeat_monitor.clone(),
    );
    tokio::spawn(heartbeat::run_watchdog(
        heartbeat_monitor,
        config.clone(),
        order_service.clone(),
        execution_clients,
        store.clone(),
        risk_engine.clone(),
        alert_sink.clone(),
    ));

    // Institutional audit Phase 2.3: per-position stop-loss / auto-reduce.
    // Independent of the dead-man's switch above (which reacts to a
    // strategy going silent, not to how a live strategy's position is
    // performing) — see stop_loss.rs's module docs for how the two
    // relate. Shares the same order_service so a stop-loss flatten goes
    // through the exact same risk-evaluation path as every other order.
    let stop_loss_state = Arc::new(StopLossState::new());
    tokio::spawn(stop_loss::run_stop_loss_monitor(
        stop_loss_state,
        config.clone(),
        order_service.clone(),
        risk_engine.clone(),
        alert_sink.clone(),
    ));

    // Institutional audit Phase 2.4: real observability — elevated
    // order-rejection-rate alerting plus a periodic PnL/exposure summary
    // log. See observability.rs's module docs.
    let observability_state = Arc::new(ObservabilityState::new());
    tokio::spawn(observability::run_observability_monitor(
        observability_state,
        config.clone(),
        store,
        risk_engine,
        alert_sink,
    ));

    let addr = "0.0.0.0:50051".parse()?;
    tracing::info!(%addr, "trading-core gRPC server starting");

    Server::builder()
        .add_service(MarketDataServiceServer::new(MarketDataServiceImpl::new(tx)))
        .add_service(OrderServiceServer::new(order_service))
        .serve(addr)
        .await?;

    Ok(())
}
