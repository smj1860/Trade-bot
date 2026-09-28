//! Loads config/config.toml (or whatever path CONFIG_PATH points at) into
//! strongly-typed structs. Nothing here is exchange- or asset-specific:
//! symbols, exchanges, and risk limits are all just data.

use rust_decimal::Decimal;
use serde::Deserialize;
use std::path::Path;
use std::str::FromStr;

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

impl SymbolConfig {
    /// The number of decimal places Kraken's own order-book checksum
    /// expects prices on this pair to be padded to (see checksum.rs's
    /// top-level docs for why this matters). Derived from `tick_size`
    /// rather than hardcoded per-pair, since `tick_size` is already this
    /// codebase's source of truth for the pair's price precision and, per
    /// live testing, matches Kraken's `pair_decimals` exactly. Falls back
    /// to 0 for an unparseable `tick_size` rather than panicking — checksum
    /// validation degrading to (probably-wrong) padding on a config typo is
    /// preferable to taking down ingestion for every symbol over it.
    pub fn price_decimals(&self) -> u32 {
        decimal_places(&self.tick_size)
    }

    /// Same as `price_decimals`, but for quantities via `lot_size`
    /// (Kraken's `lot_decimals`).
    pub fn qty_decimals(&self) -> u32 {
        decimal_places(&self.lot_size)
    }
}

fn decimal_places(value: &str) -> u32 {
    Decimal::from_str(value).map(|d| d.scale()).unwrap_or(0)
}

fn default_max_slippage_pct() -> String {
    "0.005".to_string()
}

fn default_spread_multiplier() -> String {
    "3.0".to_string()
}

fn default_min_spread_samples() -> usize {
    30
}

fn default_vol_circuit_breaker_stddev() -> String {
    "4.0".to_string()
}

fn default_vol_short_window_secs() -> u64 {
    60
}

fn default_vol_baseline_bucket_secs() -> u64 {
    60
}

fn default_vol_freeze_secs() -> u64 {
    300
}

#[derive(Debug, Deserialize, Clone)]
pub struct GlobalRisk {
    pub max_total_position_usd: String,
    pub max_orders_per_minute: u32,
    pub kill_switch_max_daily_loss_usd: String,
    /// Reject an order whose simulated fill price (walking the live book
    /// depth, not just top-of-book — see guardrails::simulate_fill_price)
    /// would differ from the current midpoint by more than this fraction.
    /// Defaults preserve old config.toml files that predate this guardrail
    /// (a config that says nothing about slippage gets a conservative
    /// 0.5% cap, not an unlimited one).
    #[serde(default = "default_max_slippage_pct")]
    pub max_slippage_pct: String,
    /// Reject an order if the book's current spread exceeds this many
    /// times its own rolling average spread (see
    /// guardrails::rolling_average_spread) — a dynamic threshold that
    /// adapts to each symbol's own normal spread rather than one flat
    /// number across very different assets.
    #[serde(default = "default_spread_multiplier")]
    pub spread_multiplier: String,
    /// The dynamic spread check does not apply until the book has
    /// recorded at least this many rolling history samples — comparing
    /// against a baseline built from a handful of ticks right after
    /// startup would be noise, not signal.
    #[serde(default = "default_min_spread_samples")]
    pub min_spread_samples: usize,
    /// Freeze new orders (see risk.rs's reduce-only enforcement while
    /// tripped) whenever the short-window Parkinson volatility estimate
    /// (guardrails::assess_volatility) is this many standard deviations
    /// above its own recent baseline.
    #[serde(default = "default_vol_circuit_breaker_stddev")]
    pub vol_circuit_breaker_stddev: String,
    /// The "1-minute" in "1-minute price volatility surges" — the recent
    /// window judged against the baseline.
    #[serde(default = "default_vol_short_window_secs")]
    pub vol_short_window_secs: u64,
    /// Bucket width used to build the baseline distribution of past
    /// short-window volatility readings that the current one is compared
    /// against.
    #[serde(default = "default_vol_baseline_bucket_secs")]
    pub vol_baseline_bucket_secs: u64,
    /// Once tripped, how long the volatility circuit breaker stays in its
    /// reduce-only/flat freeze before re-evaluating — a deliberate
    /// cooldown so a single tick dropping back under the threshold right
    /// after a spike doesn't immediately reopen the door.
    #[serde(default = "default_vol_freeze_secs")]
    pub vol_circuit_breaker_freeze_secs: u64,
}

/// Institutional audit Phase 2.2: a correlation-aware portfolio exposure
/// bucket. `max_total_position_usd` is a flat sum across every symbol
/// regardless of how correlated they are — 5 highly-correlated altcoins
/// can each individually pass every per-symbol check while the
/// *effective* portfolio risk from a correlated drawdown across all 5 is
/// far higher than that single number implies. A cluster caps combined
/// exposure across a named GROUP of symbols (majors, high-beta L1s, DeFi,
/// etc.) that are expected to move together, independent of the flat
/// total. This is the pragmatic cluster-based first step the audit
/// suggested over a full covariance-weighted calculation, which would
/// need a return-history feed this project doesn't currently maintain.
///
/// A symbol can appear in more than one cluster (e.g. a "majors" and an
/// "L1s" cluster could both reasonably include ETH-USD) — every cluster
/// containing the order's symbol is checked, not just the first match.
/// A symbol in no configured cluster is only subject to the existing
/// flat `max_total_position_usd` cap, unchanged.
#[derive(Debug, Deserialize, Clone)]
pub struct ClusterConfig {
    /// Human-readable name for log/rejection messages (e.g. "majors",
    /// "high-beta-l1s", "defi") — not matched against anything, purely
    /// for operators to understand which bucket rejected an order.
    pub name: String,
    pub symbols: Vec<String>,
    pub max_exposure_usd: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RiskSection {
    pub global: GlobalRisk,
    /// Optional — an empty/absent list (the default for any config that
    /// predates this option) means no cluster caps apply, only the
    /// existing flat max_total_position_usd, exactly matching every prior
    /// config's behavior.
    #[serde(default)]
    pub clusters: Vec<ClusterConfig>,
}

fn default_dry_run() -> bool {
    true
}

fn default_rate_limit_max_counter() -> f64 {
    // Approximates Kraken's documented "Starter" verification tier
    // (max counter 15, decays ~1 every 3s) — the most conservative
    // tier, chosen as the safe default since this project has never
    // verified which tier a real account sits in. See
    // kraken_rest.rs::RateLimiter's docs for why these numbers are an
    // approximation, not a byte-for-byte match to Kraken's real model.
    15.0
}

fn default_rate_limit_decay_per_sec() -> f64 {
    1.0 / 3.0
}

fn default_rate_limit_cost_per_call() -> f64 {
    1.0
}

fn default_rate_limit_max_wait_secs() -> f64 {
    5.0
}

/// Approximate token-bucket model of Kraken's private-REST call counter —
/// see kraken_rest.rs::RateLimiter. Defaults are conservative
/// (Starter-tier-shaped) precisely because they've never been verified
/// against a real account's actual tier; re-tune against Kraken's current
/// docs (or observed 429 behavior) before trading live.
#[derive(Debug, Deserialize, Clone)]
pub struct RateLimitConfig {
    #[serde(default = "default_rate_limit_max_counter")]
    pub max_counter: f64,
    #[serde(default = "default_rate_limit_decay_per_sec")]
    pub decay_per_sec: f64,
    #[serde(default = "default_rate_limit_cost_per_call")]
    pub cost_per_call: f64,
    /// A call that would need to wait longer than this to fit under
    /// `max_counter` fails fast (KrakenRestError::RateLimited) instead of
    /// blocking the calling task indefinitely — bounded throttling, not
    /// unbounded queuing, on what's meant to be a time-sensitive
    /// execution path.
    #[serde(default = "default_rate_limit_max_wait_secs")]
    pub max_wait_secs: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_counter: default_rate_limit_max_counter(),
            decay_per_sec: default_rate_limit_decay_per_sec(),
            cost_per_call: default_rate_limit_cost_per_call(),
            max_wait_secs: default_rate_limit_max_wait_secs(),
        }
    }
}

/// Execution safety switch. Defaults to `true` (dry-run / validate-only)
/// even if the `[execution]` section is missing from config entirely —
/// a config that says nothing about execution must never be read as
/// permission to place real orders.
#[derive(Debug, Deserialize, Clone)]
pub struct ExecutionConfig {
    #[serde(default = "default_dry_run")]
    pub dry_run: bool,
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { dry_run: true, rate_limit: RateLimitConfig::default() }
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

fn default_heartbeat_timeout_secs() -> u64 {
    30
}

fn default_watchdog_check_interval_secs() -> u64 {
    5
}

fn default_auto_cancel_orders() -> bool {
    true
}

/// Institutional audit Phase 1.1: if a strategy process stops sending
/// heartbeats (see heartbeat.rs, trading.proto's SendHeartbeat), this
/// process notices and reacts rather than leaving open orders/positions
/// unattended. Defaults to enabled with cancel-only behavior — canceling
/// resting orders is the lower-risk reaction, so it's on by default;
/// automatically flattening positions is a stronger, opt-in action (see
/// `auto_flatten_positions`'s own docs).
#[derive(Debug, Deserialize, Clone)]
pub struct DeadManSwitchConfig {
    #[serde(default = "default_dead_man_switch_enabled")]
    pub enabled: bool,
    /// How long a strategy_id can go without a heartbeat before it's
    /// treated as dead. Should be comfortably longer than whatever
    /// interval the Python side actually heartbeats at.
    #[serde(default = "default_heartbeat_timeout_secs")]
    pub heartbeat_timeout_secs: u64,
    /// How often the watchdog task checks for stale strategies.
    #[serde(default = "default_watchdog_check_interval_secs")]
    pub check_interval_secs: u64,
    /// Cancel that strategy's resting orders once it's declared dead. On
    /// by default. Note: with today's market-order-only execution path
    /// (see the institutional audit's Phase 1.3), there is rarely
    /// anything resting to cancel — a market order fills or is rejected
    /// immediately. This becomes load-bearing once a maker/limit-order
    /// path lands.
    #[serde(default = "default_auto_cancel_orders")]
    pub auto_cancel_orders: bool,
    /// Submit a reduce-to-flat order for every symbol with a nonzero
    /// tracked position once a strategy is declared dead. Off by default
    /// — this is the stronger action (it changes what's held, not just
    /// what's resting), and today's flat-quantity-only order sizing means
    /// a flatten order could itself be a large, un-vol-scaled trade. Turn
    /// this on deliberately, with eyes open, not as a default.
    #[serde(default)]
    pub auto_flatten_positions: bool,
}

fn default_dead_man_switch_enabled() -> bool {
    true
}

impl Default for DeadManSwitchConfig {
    fn default() -> Self {
        Self {
            enabled: default_dead_man_switch_enabled(),
            heartbeat_timeout_secs: default_heartbeat_timeout_secs(),
            check_interval_secs: default_watchdog_check_interval_secs(),
            auto_cancel_orders: default_auto_cancel_orders(),
            auto_flatten_positions: false,
        }
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
    #[serde(default)]
    pub dead_man_switch: DeadManSwitchConfig,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(tick_size: &str, lot_size: &str) -> SymbolConfig {
        SymbolConfig {
            symbol: "TEST-USD".into(),
            exchange: "kraken".into(),
            exchange_native_symbol: "TEST/USD".into(),
            rest_native_symbol: "TESTUSD".into(),
            tick_size: tick_size.into(),
            lot_size: lot_size.into(),
            enabled: true,
            risk: SymbolRisk {
                max_position_usd: "1000".into(),
                max_order_size: "1".into(),
                max_order_notional_usd: "1000".into(),
            },
        }
    }

    #[test]
    fn price_and_qty_decimals_come_from_tick_and_lot_size_scale() {
        // DOGE-USD's real config.example.toml values — confirmed via live
        // testing against Kraken's real feed to be exactly the checksum
        // padding precision Kraken itself uses (see checksum.rs).
        let s = symbol("0.0000001", "0.00000001");
        assert_eq!(s.price_decimals(), 7);
        assert_eq!(s.qty_decimals(), 8);
    }

    #[test]
    fn decimals_of_a_whole_number_tick_size_is_zero() {
        let s = symbol("1", "1");
        assert_eq!(s.price_decimals(), 0);
        assert_eq!(s.qty_decimals(), 0);
    }

    #[test]
    fn unparseable_tick_size_falls_back_to_zero_decimals_instead_of_panicking() {
        let s = symbol("not-a-number", "also-not-a-number");
        assert_eq!(s.price_decimals(), 0);
        assert_eq!(s.qty_decimals(), 0);
    }
}
