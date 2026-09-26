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
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::{Digest, Sha256, Sha512};

type HmacSha512 = Hmac<Sha512>;

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

#[derive(Debug, thiserror::Error)]
pub enum KrakenRestError {
    #[error("request to Kraken failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("failed to parse Kraken response: {0}")]
    Parse(String),
    #[error("system clock error building nonce: {0}")]
    Clock(String),
}

const ADD_ORDER_PATH: &str = "/0/private/AddOrder";
const GET_WEBSOCKETS_TOKEN_PATH: &str = "/0/private/GetWebSocketsToken";
const OPEN_ORDERS_PATH: &str = "/0/private/OpenOrders";
const QUERY_ORDERS_PATH: &str = "/0/private/QueryOrders";
const BALANCE_PATH: &str = "/0/private/Balance";
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
        Self {
            http: reqwest::Client::new(),
            rest_url: rest_url.into(),
            credentials,
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

    /// Shared signed-POST plumbing: builds the nonce, form-encodes
    /// `params` (with nonce prepended), signs, sends, and returns the raw
    /// response body for the caller to parse into its own result type.
    async fn signed_post(&self, path: &str, mut params: Vec<(&str, String)>) -> Result<String, KrakenRestError> {
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
}
