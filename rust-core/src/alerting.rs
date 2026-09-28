//! A minimal outbound alert sink for conditions an operator needs to
//! actually see, not just find later in a log file — starting with
//! `reconcile.rs`'s `possible_missed_fills`, the one condition in this
//! codebase that was previously only a `tracing::error!` line (see the
//! institutional audit's Phase 1.5: "wire that counter to an actual alert
//! channel... making sure a human actually sees it, not silently trusting
//! a log line nobody's watching").
//!
//! Deliberately generic rather than tied to one paid service: posts a
//! JSON body of `{"text": "<message>"}` to a configured webhook URL, which
//! is the format Slack, Mattermost, and several other chat tools accept
//! for an incoming webhook out of the box. Point `ALERT_WEBHOOK_URL` at
//! whichever one you actually use; nothing here assumes a specific
//! vendor. If it's unset, alerting silently no-ops (the log line this is
//! meant to supplement still fires from the caller) — this project should
//! never fail to start, or fail a reconciliation pass, just because
//! alerting isn't configured yet.
//!
//! Like a webhook URL for any chat tool, `ALERT_WEBHOOK_URL` is
//! bearer-token-like (anyone holding it can post into your channel), so
//! it follows this project's existing credential pattern: read from an
//! env var, never written into a config file that might get committed.

use std::time::Duration;

/// How long a single alert POST is allowed to take before giving up.
/// Alerting must never be the reason startup or a reconciliation pass
/// hangs — a slow or unreachable webhook endpoint should just mean the
/// alert didn't go out this time, logged as such, not a stuck process.
const ALERT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct AlertSink {
    webhook_url: Option<String>,
    client: reqwest::Client,
}

impl AlertSink {
    /// Reads `ALERT_WEBHOOK_URL` from the environment. Treats an unset or
    /// empty value the same as "no alerting configured" rather than
    /// erroring — this is optional infrastructure, same posture as this
    /// project's persistence store and execution client construction.
    pub fn from_env() -> Self {
        let webhook_url = std::env::var("ALERT_WEBHOOK_URL").ok().filter(|s| !s.is_empty());
        Self::new(webhook_url)
    }

    pub fn new(webhook_url: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(ALERT_TIMEOUT)
            .build()
            .expect("building the alerting HTTP client should never fail (no custom TLS/proxy config here)");
        Self { webhook_url, client }
    }

    pub fn is_configured(&self) -> bool {
        self.webhook_url.is_some()
    }

    /// Sends `message` to the configured webhook, if any. Best-effort:
    /// logs and returns on any failure (unconfigured, network error,
    /// non-2xx response) rather than propagating an error, because a
    /// failed *alert* about a problem should never itself become a new
    /// problem for the caller to handle.
    pub async fn send(&self, message: &str) {
        let Some(url) = &self.webhook_url else {
            tracing::debug!("alerting not configured (ALERT_WEBHOOK_URL unset) — skipping outbound alert");
            return;
        };

        let body = serde_json::json!({ "text": message });
        match self.client.post(url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!("alert sent successfully");
            }
            Ok(resp) => {
                tracing::error!(status = %resp.status(), "alert webhook returned a non-success status");
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to send alert to webhook");
            }
        }
    }
}

/// Builds the human-readable alert text for a reconciliation pass that
/// found one or more fills Kraken confirms happened but this process
/// never recorded locally. Kept as a standalone function so the message
/// format is testable without a network call, and so future callers (a
/// scheduled/periodic reconciliation, if one is ever added — today this
/// only runs at startup) can reuse the exact same wording.
pub fn possible_missed_fills_message(exchange: &str, count: usize) -> String {
    format!(
        ":rotating_light: Trading bot startup reconciliation on `{exchange}`: {count} order(s) executed on \
         Kraken with no matching local fill record. Position/PnL tracking for the affected symbol(s) may be \
         understated — check Kraken's trade history against this process's local `fills` table by hand. \
         See reconcile.rs's `possible_missed_fills` docs for why this isn't auto-corrected."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn unconfigured_sink_reports_not_configured() {
        let sink = AlertSink::new(None);
        assert!(!sink.is_configured());
    }

    #[test]
    fn configured_sink_reports_configured() {
        let sink = AlertSink::new(Some("http://example.invalid/hook".to_string()));
        assert!(sink.is_configured());
    }

    #[test]
    fn from_env_treats_empty_string_as_unconfigured() {
        // SAFETY: single-threaded test, no other test in this module reads
        // or writes ALERT_WEBHOOK_URL concurrently.
        unsafe {
            std::env::set_var("ALERT_WEBHOOK_URL", "");
        }
        assert!(!AlertSink::from_env().is_configured());
        unsafe {
            std::env::remove_var("ALERT_WEBHOOK_URL");
        }
    }

    #[test]
    fn possible_missed_fills_message_includes_exchange_and_count() {
        let msg = possible_missed_fills_message("kraken", 3);
        assert!(msg.contains("kraken"));
        assert!(msg.contains('3'));
    }

    #[tokio::test]
    async fn send_no_ops_without_a_configured_webhook() {
        // Nothing to assert on the network side here — this just confirms
        // it returns promptly rather than panicking or hanging when
        // unconfigured, which is the behavior every caller depends on.
        let sink = AlertSink::new(None);
        sink.send("test message").await;
    }

    #[tokio::test]
    async fn send_posts_the_message_as_json_to_the_configured_url() {
        // A tiny real HTTP server on localhost, rather than a mocking
        // crate this project doesn't already depend on — enough to prove
        // the POST actually happens with the expected body, matching the
        // rest of this codebase's preference for real integration tests
        // over new test-only dependencies (see risk.rs's real evaluate()
        // integration tests).
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local test port");
        let addr = listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept the alert POST");
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).expect("read the request");
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").expect("write response");
            request
        });

        let sink = AlertSink::new(Some(format!("http://{addr}/hook")));
        sink.send("possible missed fills on kraken").await;

        let request = server.join().expect("server thread should not panic");
        assert!(request.contains("POST /hook"));
        assert!(request.contains("possible missed fills on kraken"));
    }
}
