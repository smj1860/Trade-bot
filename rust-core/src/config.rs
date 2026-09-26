//! Loads config/config.toml (or whatever path CONFIG_PATH points at) into
//! strongly-typed structs. Nothing here is exchange- or asset-specific:
//! symbols, exchanges, and risk limits are all just data.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct GeneralConfig {
    pub log_level: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExchangeConfig {
    pub name: String,
    pub ws_url: String,
    pub rest_url: String,
    pub enabled: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SymbolRisk {
    pub max_position_usd: String,
    pub max_order_size: String,
    pub max_order_notional_usd: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SymbolConfig {
    /// Normalized internal representation, e.g. "BTC-USD".
    pub symbol: String,
    pub exchange: String,
    /// What the exchange calls this pair on its own WebSocket API.
    pub exchange_native_symbol: String,
    /// What the exchange calls this pair on its own REST order-placement
    /// API. For Kraken these are NOT the same string as the WS v2 pair
    /// (e.g. WS uses "BTC/USD", REST AddOrder historically wants an
    /// altname like "XBTUSD") — verify against Kraken's public
    /// `/0/public/AssetPairs` endpoint before trading live, since these
    /// names are Kraken's to change and this was not verified live here.
    pub rest_native_symbol: String,
    pub tick_size: String,
    pub lot_size: String,
    pub enabled: bool,
    pub risk: SymbolRisk,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GlobalRisk {
    pub max_total_position_usd: String,
    pub max_orders_per_minute: u32,
    pub kill_switch_max_daily_loss_usd: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RiskSection {
    pub global: GlobalRisk,
}

fn default_dry_run() -> bool {
    true
}

/// Execution safety switch. Defaults to `true` (dry-run / validate-only)
/// even if the `[execution]` section is missing from config entirely —
/// a config that says nothing about execution must never be read as
/// permission to place real orders.
#[derive(Debug, Deserialize, Clone)]
pub struct ExecutionConfig {
    #[serde(default = "default_dry_run")]
    pub dry_run: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { dry_run: true }
    }
}

fn default_database_path() -> String {
    "./data/trading-core.sqlite3".to_string()
}

/// Where positions, the kill-switch's daily PnL counter, and order/fill
/// history are persisted so they survive a restart. Defaults to a local
/// SQLite file even if the `[persistence]` section is missing entirely —
/// unlike `[execution]`, there's no unsafe direction to default toward
/// here, so the default just needs to always be present and writable.
#[derive(Debug, Deserialize, Clone)]
pub struct PersistenceConfig {
    #[serde(default = "default_database_path")]
    pub database_path: String,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self { database_path: default_database_path() }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub general: GeneralConfig,
    pub exchanges: Vec<ExchangeConfig>,
    pub symbols: Vec<SymbolConfig>,
    pub risk: RiskSection,
    #[serde(default)]
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            anyhow::anyhow!("failed to read config file {:?}: {e}", path.as_ref())
        })?;
        let cfg: Config = toml::from_str(&text)?;
        Ok(cfg)
    }

    /// Enabled symbols configured for a given exchange name.
    pub fn symbols_for_exchange(&self, exchange: &str) -> Vec<SymbolConfig> {
        self.symbols
            .iter()
            .filter(|s| s.enabled && s.exchange == exchange)
            .cloned()
            .collect()
    }
}
