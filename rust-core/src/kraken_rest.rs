//! Kraken REST execution client — the piece that turns a risk-approved
//! order into a real HTTP request against Kraken's private AddOrder
//! endpoint.
//!
//! This is the highest-stakes code in the project: a bug here can send
//! real money to the exchange. Two things are true about it, and both
//! matter more than usual:
//!
//! 1. **Safe by default.** Every request is sent with Kraken's own
//!    `validate` flag set whenever `ExecutionConfig::dry_run` is true
//!    (the default, even with no `[execution]` section at all — see
//!    `config.rs`). Kraken checks and responds to a validate=true request
//!    exactly like a real one, but never places it on the book.
//! 2. **Honest about what's unverified.** The HMAC-SHA512 request-signing
//!    scheme below is implemented from Kraken's public documentation, but
//!    nothing in this sandbox holds real Kraken API credentials, so the
//!    signature has never been checked against Kraken's server and
//!    accepted as valid. What *can* be shown to work end-to-end (see the
//!    `smoke_test` binary/test) is: the HTTP request is built correctly,
//!    reaches Kraken, and Kraken's JSON error response is parsed
//!    correctly — using deliberately-invalid credentials, which Kraken
//!    should reject with an authentication error. That proves the
//!    plumbing, not the cryptography's correctness against a live,
//!    authenticated account.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::{Digest, Sha256, Sha512};
use tokio::sync::Mutex;

use crate::config::RateLimitConfig;

type HmacSha512 = Hmac<Sha512>;

/// Approximate token-bucket model of Kraken's private-REST call counter.
///
/// **Honesty about what this is and isn't**, in the same spirit as this
/// module's top-level docs on the signing scheme: Kraken's real private
/// API rate limiting increments an account-wide counter by a
/// documented-but-endpoint-varying cost per call, decays it continuously
/// over time, and caps it at a value that depends on the account's
/// verification tier (Starter/Intermediate/Pro) — all of which are
/// Kraken's to change and none of which has been checked against a real
/// account from this project. This implements the *shape* of that model
/// (a counter that grows by a cost, decays continuously, and rejects/
/// throttles once it would exceed a cap) with a single flat
/// `cost_per_call` rather than Kraken's actual per-endpoint cost table,
/// configured conservatively (`config::RateLimitConfig`'s defaults
/// approximate the Starter tier, the most restrictive). Re-tune against
/// Kraken's current docs — or observed real 429/`EAPI:Rate limit
/// exceeded` behavior — before relying on this to actually prevent a
/// live-account suspension.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    max_counter: f64,
    decay_per_sec: f64,
    cost_per_call: f64,
    max_wait: Duration,
    state: Arc<Mutex<RateLimiterState>>,
}

#[derive(Debug)]
struct RateLimiterState {
    counter: f64,
    last_update: Instant,
}

impl RateLimiter {
    pub fn new(config: &RateLimitConfig) -> Self {
        Self {
            max_counter: config.max_counter,
            decay_per_sec: config.decay_per_sec,
            cost_per_call: config.cost_per_call,
            max_wait: Duration::from_secs_f64(config.max_wait_secs.max(0.0)),
            state: Arc::new(Mutex::new(RateLimiterState { counter: 0.0, last_update: Instant::now() })),
        }
    }

    /// Decays the counter for elapsed time, then either reserves
    /// `cost_per_call` and returns immediately, or — if the counter is
    /// currently too high to fit the call — sleeps until it would fit,
    /// as long as that wait is within `max_wait`. Returns
    /// `Err(needed_wait)` without reserving anything if the wait would
    /// exceed `max_wait`: this is a time-sensitive execution path, so a
    /// call that can't be throttled within a bounded window fails fast
    /// (the caller surfaces `KrakenRestError::RateLimited`) rather than
    /// blocking indefinitely.
    pub async fn acquire(&self) -> Result<(), Duration> {
        loop {
            let wait = {
                let mut state = self.state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_update).as_secs_f64();
                state.counter = (state.counter - elapsed * self.decay_per_sec).max(0.0);
                state.last_update = now;

                if state.counter + self.cost_per_call <= self.max_counter {
                    state.counter += self.cost_per_call;
                    return Ok(());
                }

                // How long until decay alone brings the counter down
                // enough for this call to fit.
                let overage = state.counter + self.cost_per_call - self.max_counter;
                if self.decay_per_sec <= 0.0 {
                    return Err(self.max_wait + Duration::from_secs(1)); // never decays — never fits
                }
                Duration::from_secs_f64(overage / self.decay_per_sec)
            };

            if wait > self.max_wait {
                return Err(wait);
            }
            tokio::time::sleep(wait).await;
            // Loop again: re-check under lock rather than assuming the
            // sleep left the counter exactly where predicted (a
            // concurrent call could have reserved capacity in between).
        }
    }
}

#[derive(Debug, Clone)]
pub struct KrakenCredentials {
    pub api_key: String,
    /// Base64-encoded, exactly as Kraken issues it.
    pub api_secret: String,
}

#[derive(Debug, Clone)]
pub struct KrakenRestClient {
    http: reqwest::Client,
    rest_url: String,
    credentials: KrakenCredentials,
    rate_limiter: RateLimiter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    fn as_kraken_str(self) -> &'static str {
        match self {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddOrderRequest {
    /// Kraken REST altname for the pair, e.g. "XBTUSD" — NOT the WS v2
    /// slash-delimited symbol. See `SymbolConfig::rest_native_symbol`.
    pub pair: String,
    pub side: OrderSide,
    /// "market" or "limit" — kept as a plain string here since Kraken's
    /// REST API supports several order types this project doesn't
    /// generate yet (stop-loss, take-profit, etc.); the OrderService
    /// layer is the one that maps our proto OrderType onto this.
    pub order_type: &'static str,
    pub volume: String,
    /// Required for order_type == "limit", absent for "market".
    pub price: Option<String>,
    /// Kraken's client-supplied order ID field (`cl_ord_id` in newer
    /// Kraken API versions / `userref` historically expects an integer —
    /// this project targets the string-based `cl_ord_id` field, which is
    /// what current Kraken REST docs describe; if targeting an older
    /// Kraken deployment that only accepts `userref` as an integer, this
    /// will need adjusting, and that has NOT been verified live).
    pub client_order_id: String,
    /// When true, Kraken validates the request without placing it.
    pub validate: bool,
    /// Institutional audit Phase 1.3: requests Kraken's `oflags=post`
    /// (post-only). Kraken rejects the order outright rather than filling
    /// it as a taker if its price would cross the book at submission
    /// time — this is the exchange-side backstop for the maker/limit
    /// order path (see order.rs and risk.rs's guardrails, which apply
    /// their own client-side non-crossing check before an order ever gets
    /// here). Meaningless for a market order; only set for `order_type ==
    /// "limit"`.
    pub post_only: bool,
}

#[derive(Debug, Deserialize)]
struct KrakenResponse<T> {
    error: Vec<String>,
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
pub struct AddOrderResult {
    #[serde(default)]
    pub txid: Vec<String>,
    #[serde(default)]
    pub descr: Option<AddOrderDescr>,
}

#[derive(Debug, Deserialize)]
pub struct AddOrderDescr {
    #[serde(default)]
    pub order: Option<String>,
}

/// What calling AddOrder actually produced. Deliberately not a single
/// "success" bool: a validate=true call that Kraken accepts still has no
/// exchange_order_id, because nothing was placed.
#[derive(Debug)]
pub enum AddOrderOutcome {
    /// Kraken accepted and (if not validate-only) assigned at least one
    /// transaction ID.
    Accepted { exchange_order_id: Option<String> },
    /// Kraken's `error` array was non-empty — a business-level rejection
    /// (bad pair, insufficient funds, invalid nonce, auth failure, etc).
    /// This is Kraken saying no, not a network/transport failure.
    KrakenRejected { messages: Vec<String> },
}

/// Outcome of `KrakenRestClient::cancel_order`.
#[derive(Debug, PartialEq, Eq)]
pub enum CancelOrderOutcome {
    /// Kraken canceled a resting order.
    Canceled,
    /// Kraken's `error` array was non-empty — a business-level rejection.
    KrakenRejected { messages: Vec<String> },
    /// No error, but `count` was 0 or missing — the order was already
    /// filled, already canceled, or never existed. Treated the same as
    /// success by callers that just want "make sure this isn't resting
    /// anymore," since that's already true.
    AlreadyClosed,
}

/// Maps a Kraken CancelOrder response's `error` array and `count` onto
/// `CancelOrderOutcome`. Pure and synchronous so it's testable without a
/// network call, the same split `reconcile.rs`'s `resolve_closed_status`
/// uses for its own response-classification logic.
fn classify_cancel_response(error: &[String], count: Option<u64>) -> CancelOrderOutcome {
    if !error.is_empty() {
        return CancelOrderOutcome::KrakenRejected { messages: error.to_vec() };
    }
    match count {
        Some(c) if c > 0 => CancelOrderOutcome::Canceled,
        _ => CancelOrderOutcome::AlreadyClosed,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KrakenRestError {
    #[error("request to Kraken failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("failed to parse Kraken response: {0}")]
    Parse(String),
    #[error("system clock error building nonce: {0}")]
    Clock(String),
    #[error(
        "local rate limiter throttled this call: would need to wait {0:?} to stay under the configured \
         counter cap, which exceeds max_wait_secs — refusing to send rather than block indefinitely \
         (see RateLimiter's docs)"
    )]
    RateLimited(Duration),
}

const ADD_ORDER_PATH: &str = "/0/private/AddOrder";
const GET_WEBSOCKETS_TOKEN_PATH: &str = "/0/private/GetWebSocketsToken";
const OPEN_ORDERS_PATH: &str = "/0/private/OpenOrders";
const QUERY_ORDERS_PATH: &str = "/0/private/QueryOrders";
const BALANCE_PATH: &str = "/0/private/Balance";
const CANCEL_ORDER_PATH: &str = "/0/private/CancelOrder";
/// Kraken's own documented cap on how many transaction IDs a single
/// QueryOrders call accepts.
const QUERY_ORDERS_MAX_TXIDS: usize = 50;

/// One order as Kraken's `OpenOrders`/`QueryOrders` endpoints describe it —
/// the same shape serves both (`QueryOrders` just adds a few fields this
/// project doesn't need, like `closetm`). Used by reconcile.rs to cross-
/// check this process's locally persisted order state against what Kraken
/// actually has.
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderInfo {
    /// Only present if this order was placed with a `cl_ord_id`, which is
    /// every order this project itself places — but a reconciliation pass
    /// can also see orders it never placed (see reconcile.rs), and those
    /// won't have one.
    #[serde(default)]
    pub cl_ord_id: Option<String>,
    /// "open" (OpenOrders only) / "closed" / "canceled" / "expired"
    /// (QueryOrders can return any terminal or non-terminal status).
    pub status: String,
    pub descr: KrakenOrderDescr,
    /// Total order volume, decimal-as-string like everywhere else in this
    /// project.
    #[serde(default)]
    pub vol: String,
    /// Volume executed so far, decimal-as-string.
    #[serde(default)]
    pub vol_exec: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderDescr {
    pub pair: String,
    #[serde(rename = "type")]
    pub side: String,
}

/// A token for Kraken's private (authenticated) WebSocket v2 feed —
/// separate entirely from the public market-data feed in kraken.rs, which
/// needs no authentication. Used by kraken_private_ws.rs to subscribe to
/// the `executions` channel (real order fills/status changes).
#[derive(Debug, Deserialize)]
pub struct WebSocketsToken {
    pub token: String,
    /// Seconds until the token expires if never used. Per Kraken's docs,
    /// a token already in use on an open, maintained connection does not
    /// expire — this project fetches a fresh token on every (re)connect
    /// rather than trying to track or reuse expiry, which is simpler and
    /// avoids ever presenting a stale token.
    #[serde(default)]
    pub expires: u64,
}

impl KrakenRestClient {
    pub fn new(rest_url: impl Into<String>, credentials: KrakenCredentials) -> Self {
        Self::with_rate_limit(rest_url, credentials, &RateLimitConfig::default())
    }

    pub fn with_rate_limit(
        rest_url: impl Into<String>,
        credentials: KrakenCredentials,
        rate_limit: &RateLimitConfig,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            rest_url: rest_url.into(),
            credentials,
            rate_limiter: RateLimiter::new(rate_limit),
        }
    }

    pub async fn add_order(&self, req: &AddOrderRequest) -> Result<AddOrderOutcome, KrakenRestError> {
        let mut form: Vec<(&str, String)> = vec![
            ("ordertype", req.order_type.to_string()),
            ("type", req.side.as_kraken_str().to_string()),
            ("volume", req.volume.clone()),
            ("pair", req.pair.clone()),
            ("cl_ord_id", req.client_order_id.clone()),
        ];
        if let Some(price) = &req.price {
            form.push(("price", price.clone()));
        }
        if req.post_only {
            form.push(("oflags", "post".to_string()));
        }
        if req.validate {
            form.push(("validate", "true".to_string()));
        }

        let text = self.signed_post(ADD_ORDER_PATH, form).await?;
        let parsed: KrakenResponse<AddOrderResult> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;

        if !parsed.error.is_empty() {
            return Ok(AddOrderOutcome::KrakenRejected { messages: parsed.error });
        }

        let exchange_order_id = parsed.result.and_then(|r| r.txid.into_iter().next());
        Ok(AddOrderOutcome::Accepted { exchange_order_id })
    }

    /// Cancels a single resting order by its Kraken transaction ID. Used
    /// by the dead-man's switch (`heartbeat.rs`) when a strategy process
    /// goes dark while an order is resting, and — as of institutional
    /// audit Phase 1.3 — by `OrderServiceImpl::cancel_order` (order.rs),
    /// which Python calls directly to manage a maker/limit order's
    /// cancel-and-reprice lifecycle.
    ///
    /// Kraken's `count` field in a successful response can be 0 even
    /// without an `error` — e.g. the order already filled or was already
    /// canceled a moment earlier — so this is reported as `AlreadyClosed`
    /// rather than treated as a hard failure; the caller almost never
    /// wants to distinguish "nothing to cancel" from "network trouble."
    pub async fn cancel_order(&self, txid: &str) -> Result<CancelOrderOutcome, KrakenRestError> {
        #[derive(Debug, Deserialize)]
        struct CancelOrderResult {
            #[serde(default)]
            count: u64,
        }

        let form = vec![("txid", txid.to_string())];
        let text = self.signed_post(CANCEL_ORDER_PATH, form).await?;
        let parsed: KrakenResponse<CancelOrderResult> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;

        Ok(classify_cancel_response(&parsed.error, parsed.result.map(|r| r.count)))
    }

    /// Fetches a fresh token for the private WebSocket feed. This uses the
    /// same classic private-REST signing scheme as `add_order` — this
    /// project has NOT independently confirmed that against a real,
    /// funded account any more than `add_order`'s signing has been (see
    /// this module's top-level docs); it's implemented consistently with
    /// Kraken's documented behavior for the private REST API as a whole,
    /// not verified byte-for-byte against this specific endpoint.
    pub async fn get_websockets_token(&self) -> Result<WebSocketsToken, KrakenRestError> {
        let text = self.signed_post(GET_WEBSOCKETS_TOKEN_PATH, vec![]).await?;
        let parsed: KrakenResponse<WebSocketsToken> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;

        if !parsed.error.is_empty() {
            return Err(KrakenRestError::Parse(format!("Kraken rejected GetWebSocketsToken: {:?}", parsed.error)));
        }
        parsed
            .result
            .ok_or_else(|| KrakenRestError::Parse("GetWebSocketsToken response had no error but also no result".to_string()))
    }

    /// Every order Kraken currently considers open on this account, keyed
    /// by exchange order ID (txid). Used by reconcile.rs at startup to
    /// find out which locally-"open" orders Kraken no longer agrees are
    /// open.
    pub async fn get_open_orders(&self) -> Result<HashMap<String, KrakenOrderInfo>, KrakenRestError> {
        #[derive(Debug, Deserialize)]
        struct OpenOrdersResult {
            #[serde(default)]
            open: HashMap<String, KrakenOrderInfo>,
        }

        let text = self.signed_post(OPEN_ORDERS_PATH, vec![]).await?;
        let parsed: KrakenResponse<OpenOrdersResult> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;
        if !parsed.error.is_empty() {
            return Err(KrakenRestError::Parse(format!("Kraken rejected OpenOrders: {:?}", parsed.error)));
        }
        Ok(parsed.result.map(|r| r.open).unwrap_or_default())
    }

    /// Looks up specific orders by exchange order ID (txid), whether open
    /// or closed — this is how reconcile.rs finds out what actually
    /// happened to a locally-"open" order that Kraken's OpenOrders no
    /// longer lists (filled, canceled, or expired). `txids` must not
    /// exceed Kraken's documented cap of 50 per call; the caller is
    /// responsible for chunking a longer list.
    pub async fn query_orders(&self, txids: &[String]) -> Result<HashMap<String, KrakenOrderInfo>, KrakenRestError> {
        if txids.is_empty() {
            return Ok(HashMap::new());
        }
        if txids.len() > QUERY_ORDERS_MAX_TXIDS {
            return Err(KrakenRestError::Parse(format!(
                "query_orders called with {} txids, exceeding Kraken's cap of {QUERY_ORDERS_MAX_TXIDS} per call",
                txids.len()
            )));
        }

        let form = vec![("txid", txids.join(","))];
        let text = self.signed_post(QUERY_ORDERS_PATH, form).await?;
        let parsed: KrakenResponse<HashMap<String, KrakenOrderInfo>> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;
        if !parsed.error.is_empty() {
            return Err(KrakenRestError::Parse(format!("Kraken rejected QueryOrders: {:?}", parsed.error)));
        }
        Ok(parsed.result.unwrap_or_default())
    }

    /// Current wallet balances, keyed by Kraken's own asset code (e.g.
    /// `"ZUSD"`, `"XXBT"`). Informational only — a spot wallet balance is
    /// not the same thing as `RiskEngine`'s tracked position, which starts
    /// at zero when this process first runs and only reflects fills seen
    /// since then. Reconciling the two would require knowing the account's
    /// pre-bot holdings, which this project has no way to learn; this is
    /// logged at startup purely so an operator can eyeball it.
    pub async fn get_account_balance(&self) -> Result<HashMap<String, String>, KrakenRestError> {
        let text = self.signed_post(BALANCE_PATH, vec![]).await?;
        let parsed: KrakenResponse<HashMap<String, String>> = serde_json::from_str(&text)
            .map_err(|e| KrakenRestError::Parse(format!("{e} — raw body: {text}")))?;
        if !parsed.error.is_empty() {
            return Err(KrakenRestError::Parse(format!("Kraken rejected Balance: {:?}", parsed.error)));
        }
        Ok(parsed.result.unwrap_or_default())
    }

    /// Shared signed-POST plumbing: throttles against the local rate
    /// limiter (see `RateLimiter`), builds the nonce, form-encodes
    /// `params` (with nonce prepended), signs, sends, and returns the raw
    /// response body for the caller to parse into its own result type.
    async fn signed_post(&self, path: &str, mut params: Vec<(&str, String)>) -> Result<String, KrakenRestError> {
        self.rate_limiter.acquire().await.map_err(KrakenRestError::RateLimited)?;

        let nonce = nonce_millis()?;
        params.insert(0, ("nonce", nonce.clone()));

        let body = serde_urlencoded::to_string(&params)
            .map_err(|e| KrakenRestError::Parse(format!("failed to encode form body: {e}")))?;

        let signature = sign(&self.credentials.api_secret, path, &nonce, &body)
            .map_err(|e| KrakenRestError::Parse(format!("failed to compute request signature: {e}")))?;

        let url = format!("{}{}", self.rest_url, path);
        let response = self
            .http
            .post(&url)
            .header("API-Key", &self.credentials.api_key)
            .header("API-Sign", signature)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await?;

        Ok(response.text().await?)
    }
}

fn nonce_millis() -> Result<String, KrakenRestError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| KrakenRestError::Clock(e.to_string()))?;
    Ok(now.as_millis().to_string())
}

/// Kraken's documented signing scheme:
///   HMAC-SHA512( base64_decode(api_secret),
///                uri_path + SHA256(nonce + POST_data) )
/// base64-encoded, sent as the API-Sign header.
fn sign(api_secret_b64: &str, uri_path: &str, nonce: &str, post_data: &str) -> Result<String, String> {
    let secret = base64::engine::general_purpose::STANDARD
        .decode(api_secret_b64)
        .map_err(|e| format!("api_secret is not valid base64: {e}"))?;

    let mut sha256 = Sha256::new();
    sha256.update(nonce.as_bytes());
    sha256.update(post_data.as_bytes());
    let nonce_and_post_hash = sha256.finalize();

    let mut mac = HmacSha512::new_from_slice(&secret).map_err(|e| format!("invalid HMAC key length: {e}"))?;
    mac.update(uri_path.as_bytes());
    mac.update(&nonce_and_post_hash);
    let signature = mac.finalize().into_bytes();

    Ok(base64::engine::general_purpose::STANDARD.encode(signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate_limit_config(max_counter: f64, decay_per_sec: f64, cost_per_call: f64, max_wait_secs: f64) -> RateLimitConfig {
        RateLimitConfig { max_counter, decay_per_sec, cost_per_call, max_wait_secs }
    }

    #[tokio::test]
    async fn rate_limiter_allows_calls_under_the_cap() {
        let limiter = RateLimiter::new(&rate_limit_config(10.0, 1.0, 1.0, 5.0));
        for _ in 0..10 {
            assert!(limiter.acquire().await.is_ok());
        }
    }

    #[tokio::test]
    async fn rate_limiter_throttles_within_max_wait() {
        // max_counter=2, cost=1 per call, decays fast (10/sec) — the third
        // call needs to wait ~0.1s for the counter to drop back to 1, well
        // inside max_wait, so it should succeed rather than error.
        let limiter = RateLimiter::new(&rate_limit_config(2.0, 10.0, 1.0, 5.0));
        assert!(limiter.acquire().await.is_ok());
        assert!(limiter.acquire().await.is_ok());
        let start = Instant::now();
        assert!(limiter.acquire().await.is_ok());
        assert!(start.elapsed() >= Duration::from_millis(50), "should have actually waited for decay");
    }

    #[tokio::test]
    async fn rate_limiter_fails_fast_when_wait_exceeds_max_wait() {
        // max_counter=1, cost=1, decays very slowly (0.01/sec) and
        // max_wait is tiny — the second call would need ~99s to fit,
        // which exceeds max_wait, so it should error rather than block.
        let limiter = RateLimiter::new(&rate_limit_config(1.0, 0.01, 1.0, 0.05));
        assert!(limiter.acquire().await.is_ok());
        let result = limiter.acquire().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn rate_limiter_never_decaying_always_errors_after_first_fill() {
        let limiter = RateLimiter::new(&rate_limit_config(1.0, 0.0, 1.0, 0.1));
        assert!(limiter.acquire().await.is_ok());
        assert!(limiter.acquire().await.is_err());
    }

    // These are structural tests only — they confirm the signing function
    // is deterministic and sensitive to its inputs. They do NOT and
    // cannot confirm the signature matches what Kraken's server expects,
    // since that requires a real account's credentials to check against.

    fn fake_secret() -> String {
        // 32 arbitrary bytes, base64-encoded — shaped like a real Kraken
        // secret, not a real one.
        base64::engine::general_purpose::STANDARD.encode([7u8; 32])
    }

    #[test]
    fn signature_is_deterministic_for_same_inputs() {
        let secret = fake_secret();
        let sig1 = sign(&secret, "/0/private/AddOrder", "1234567890", "pair=XBTUSD&volume=1").unwrap();
        let sig2 = sign(&secret, "/0/private/AddOrder", "1234567890", "pair=XBTUSD&volume=1").unwrap();
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn signature_changes_with_nonce() {
        let secret = fake_secret();
        let sig1 = sign(&secret, "/0/private/AddOrder", "1111111111", "pair=XBTUSD&volume=1").unwrap();
        let sig2 = sign(&secret, "/0/private/AddOrder", "2222222222", "pair=XBTUSD&volume=1").unwrap();
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn signature_changes_with_post_data() {
        let secret = fake_secret();
        let sig1 = sign(&secret, "/0/private/AddOrder", "1234567890", "pair=XBTUSD&volume=1").unwrap();
        let sig2 = sign(&secret, "/0/private/AddOrder", "1234567890", "pair=XBTUSD&volume=2").unwrap();
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn signature_changes_with_uri_path() {
        let secret = fake_secret();
        let sig1 = sign(&secret, "/0/private/AddOrder", "1234567890", "pair=XBTUSD&volume=1").unwrap();
        let sig2 = sign(&secret, "/0/private/CancelOrder", "1234567890", "pair=XBTUSD&volume=1").unwrap();
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn rejects_non_base64_secret() {
        let err = sign("not valid base64 ###", "/0/private/AddOrder", "1", "x=1").unwrap_err();
        assert!(err.contains("base64"));
    }

    // These test only JSON parsing against synthetic responses shaped like
    // Kraken's documented OpenOrders/QueryOrders/Balance schemas — they do
    // NOT confirm that shape against a real response, for the same reason
    // stated at the top of this module.

    #[test]
    fn parses_a_synthetic_open_orders_response() {
        let body = r#"{
            "error": [],
            "result": {
                "open": {
                    "OQCLML-BW3P3-BUCMWZ": {
                        "cl_ord_id": "co-1",
                        "status": "open",
                        "descr": {"pair": "XBTUSD", "type": "buy"},
                        "vol": "1.00000000",
                        "vol_exec": "0.00000000"
                    }
                }
            }
        }"#;
        let parsed: KrakenResponse<serde_json::Value> = serde_json::from_str(body).unwrap();
        assert!(parsed.error.is_empty());
        let open = &parsed.result.unwrap()["open"];
        let order: KrakenOrderInfo = serde_json::from_value(open["OQCLML-BW3P3-BUCMWZ"].clone()).unwrap();
        assert_eq!(order.cl_ord_id.as_deref(), Some("co-1"));
        assert_eq!(order.status, "open");
        assert_eq!(order.descr.pair, "XBTUSD");
        assert_eq!(order.descr.side, "buy");
    }

    #[test]
    fn parses_a_synthetic_query_orders_response_with_no_cl_ord_id() {
        // QueryOrders can return an order this project never placed
        // (reconcile.rs's "Kraken has an order we don't know about" case),
        // which never had a cl_ord_id to echo back.
        let order: KrakenOrderInfo = serde_json::from_str(
            r#"{"status": "closed", "descr": {"pair": "XBTUSD", "type": "sell"}, "vol": "0.5", "vol_exec": "0.5"}"#,
        )
        .unwrap();
        assert_eq!(order.cl_ord_id, None);
        assert_eq!(order.status, "closed");
    }

    #[test]
    fn parses_a_synthetic_balance_response() {
        let parsed: KrakenResponse<HashMap<String, String>> =
            serde_json::from_str(r#"{"error": [], "result": {"ZUSD": "1000.0000", "XXBT": "0.5000000000"}}"#).unwrap();
        let balances = parsed.result.unwrap();
        assert_eq!(balances.get("ZUSD").map(String::as_str), Some("1000.0000"));
    }

    #[test]
    fn classify_cancel_response_reports_canceled_when_count_is_positive() {
        assert_eq!(classify_cancel_response(&[], Some(1)), CancelOrderOutcome::Canceled);
    }

    #[test]
    fn classify_cancel_response_reports_already_closed_on_zero_count() {
        assert_eq!(classify_cancel_response(&[], Some(0)), CancelOrderOutcome::AlreadyClosed);
    }

    #[test]
    fn classify_cancel_response_reports_already_closed_on_missing_count() {
        assert_eq!(classify_cancel_response(&[], None), CancelOrderOutcome::AlreadyClosed);
    }

    #[test]
    fn classify_cancel_response_prioritizes_a_kraken_error_over_count() {
        let outcome = classify_cancel_response(&["EOrder:Unknown order".to_string()], Some(1));
        assert_eq!(outcome, CancelOrderOutcome::KrakenRejected { messages: vec!["EOrder:Unknown order".to_string()] });
    }
}
