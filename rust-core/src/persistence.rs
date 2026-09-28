//! SQLite-backed persistence for positions, the kill switch's daily PnL
//! counter, and order/fill history — the state that previously lived only
//! in `RiskEngine`'s in-memory maps and reset to zero/empty on every
//! restart.
//!
//! Uses `rusqlite` with the bundled SQLite (no system library dependency)
//! rather than an async driver: every method here is a short, synchronous
//! call (a handful of single-row reads/writes), and `Store`'s methods
//! never hold their lock across an `.await`, so a blocking `std::sync::
//! Mutex<Connection>` is simpler and just as correct as threading an async
//! driver through the codebase for this volume of writes (fills and order
//! submissions, not per-tick market data).
//!
//! Every numeric value is stored as `TEXT`, not `REAL` — the same
//! decimal-as-string discipline used everywhere else in this project
//! (config, gRPC), so nothing here is ever exposed to floating-point
//! rounding.

use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use rust_decimal::Decimal;

use crate::proto::pb::OrderStatus;

/// Renders a proto `OrderStatus` as the string this store persists it as.
/// Shared by `order.rs` (order submission outcomes) and
/// `kraken_private_ws.rs` (status changes arriving on the private feed) so
/// both write the same vocabulary — `load_open_orders`'s terminal-status
/// filter depends on matching these exact strings.
pub fn order_status_label(status: OrderStatus) -> &'static str {
    match status {
        OrderStatus::Unspecified => "UNSPECIFIED",
        OrderStatus::Accepted => "ACCEPTED",
        OrderStatus::Rejected => "REJECTED",
        OrderStatus::PartiallyFilled => "PARTIALLY_FILLED",
        OrderStatus::Filled => "FILLED",
        OrderStatus::Canceled => "CANCELED",
    }
}

/// A persisted net position for one symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedPosition {
    pub symbol: String,
    pub qty: Decimal,
    pub avg_entry_price: Decimal,
}

/// The kill switch's daily counter as last saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedKillSwitchState {
    pub day_index: u64,
    pub realized_pnl_usd: Decimal,
}

/// A row for the `orders` table — one per `client_order_id`, updated in
/// place as an order's status changes rather than appended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderRecord {
    pub client_order_id: String,
    pub exchange_order_id: String,
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    pub order_type: String,
    pub quantity: String,
    pub limit_price: Option<String>,
    pub strategy_id: String,
    pub status: String,
    pub reject_reason: String,
    pub created_at_ns: i64,
    pub updated_at_ns: i64,
}

/// A single applied fill, kept as an append-only audit trail. `exec_id` is
/// the exchange's own identifier for the execution when one is available
/// (Kraken's `exec_id`/`trade_id`) — the same value `RiskEngine::apply_fill`
/// uses to avoid double-applying a redelivered event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillRecord {
    pub exec_id: Option<String>,
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub side: String,
    pub qty: Decimal,
    pub price: Decimal,
    pub realized_pnl_usd: Decimal,
    pub applied_at_ns: i64,
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Opens (creating if needed) the SQLite database at `path`, including
    /// any parent directories, and runs migrations. Migrations are plain
    /// `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`
    /// statements — idempotent by construction, so there's no migration
    /// version to track for a schema this small.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// An in-memory database, for tests: same schema, no file on disk.
    pub fn open_in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn init_schema(conn: &Connection) -> anyhow::Result<()> {
        // WAL mode so a reader (an ops query against the file while the
        // process is running) doesn't block writers, and vice versa.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", true)?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS positions (
                symbol           TEXT PRIMARY KEY,
                qty              TEXT NOT NULL,
                avg_entry_price  TEXT NOT NULL,
                updated_at_ns    INTEGER NOT NULL
            );

            -- Singleton row (id is always 1) holding the kill switch's
            -- running daily-PnL counter, so a restart mid-day resumes
            -- with the correct total instead of a fresh zero.
            CREATE TABLE IF NOT EXISTS kill_switch_state (
                id                INTEGER PRIMARY KEY CHECK (id = 1),
                day_index         INTEGER NOT NULL,
                realized_pnl_usd  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                client_order_id   TEXT PRIMARY KEY,
                exchange_order_id TEXT NOT NULL DEFAULT '',
                symbol            TEXT NOT NULL,
                exchange          TEXT NOT NULL,
                side              TEXT NOT NULL,
                order_type        TEXT NOT NULL,
                quantity          TEXT NOT NULL,
                limit_price       TEXT,
                strategy_id       TEXT NOT NULL DEFAULT '',
                status            TEXT NOT NULL,
                reject_reason     TEXT NOT NULL DEFAULT '',
                created_at_ns     INTEGER NOT NULL,
                updated_at_ns     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

            CREATE TABLE IF NOT EXISTS fills (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                exec_id            TEXT UNIQUE,
                client_order_id    TEXT,
                symbol             TEXT NOT NULL,
                side               TEXT NOT NULL,
                qty                TEXT NOT NULL,
                price              TEXT NOT NULL,
                realized_pnl_usd   TEXT NOT NULL,
                applied_at_ns      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
            "#,
        )?;
        Ok(())
    }

    // -- positions --------------------------------------------------------

    pub fn load_positions(&self) -> anyhow::Result<Vec<PersistedPosition>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT symbol, qty, avg_entry_price FROM positions")?;
        let rows = stmt.query_map([], |row| {
            let symbol: String = row.get(0)?;
            let qty: String = row.get(1)?;
            let avg_entry_price: String = row.get(2)?;
            Ok((symbol, qty, avg_entry_price))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (symbol, qty, avg_entry_price) = row?;
            let qty = Decimal::from_str(&qty)
                .map_err(|e| anyhow::anyhow!("corrupt qty for {symbol} in positions table: {e}"))?;
            let avg_entry_price = Decimal::from_str(&avg_entry_price)
                .map_err(|e| anyhow::anyhow!("corrupt avg_entry_price for {symbol} in positions table: {e}"))?;
            out.push(PersistedPosition { symbol, qty, avg_entry_price });
        }
        Ok(out)
    }

    pub fn upsert_position(&self, symbol: &str, qty: Decimal, avg_entry_price: Decimal, updated_at_ns: i64) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO positions (symbol, qty, avg_entry_price, updated_at_ns)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(symbol) DO UPDATE SET
                qty = excluded.qty,
                avg_entry_price = excluded.avg_entry_price,
                updated_at_ns = excluded.updated_at_ns",
            params![symbol, qty.to_string(), avg_entry_price.to_string(), updated_at_ns],
        )?;
        Ok(())
    }

    // -- kill switch --------------------------------------------------------

    pub fn load_kill_switch_state(&self) -> anyhow::Result<Option<PersistedKillSwitchState>> {
        let conn = self.conn.lock().unwrap();
        let row: Option<(i64, String)> = conn
            .query_row(
                "SELECT day_index, realized_pnl_usd FROM kill_switch_state WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        match row {
            None => Ok(None),
            Some((day_index, realized_pnl_usd)) => {
                let realized_pnl_usd = Decimal::from_str(&realized_pnl_usd)
                    .map_err(|e| anyhow::anyhow!("corrupt realized_pnl_usd in kill_switch_state: {e}"))?;
                Ok(Some(PersistedKillSwitchState { day_index: day_index as u64, realized_pnl_usd }))
            }
        }
    }

    pub fn save_kill_switch_state(&self, day_index: u64, realized_pnl_usd: Decimal) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO kill_switch_state (id, day_index, realized_pnl_usd)
             VALUES (1, ?1, ?2)
             ON CONFLICT(id) DO UPDATE SET
                day_index = excluded.day_index,
                realized_pnl_usd = excluded.realized_pnl_usd",
            params![day_index as i64, realized_pnl_usd.to_string()],
        )?;
        Ok(())
    }

    // -- fills --------------------------------------------------------------

    /// True if a fill with this `exec_id` has already been recorded. Used
    /// to decide, before touching any position state, whether an incoming
    /// execution report is a redelivery — this is the durable half of the
    /// fill-dedup story; kraken_private_ws.rs's in-memory window catches
    /// the common case cheaply, this catches it across a reconnect or a
    /// full process restart, which the in-memory window cannot.
    pub fn fill_exists(&self, exec_id: &str) -> anyhow::Result<bool> {
        let conn = self.conn.lock().unwrap();
        let exists: Option<i64> = conn
            .query_row("SELECT 1 FROM fills WHERE exec_id = ?1", params![exec_id], |row| row.get(0))
            .optional()?;
        Ok(exists.is_some())
    }

    /// True if at least one fill has ever been recorded for this
    /// `client_order_id` — used by startup reconciliation to flag an order
    /// Kraken shows as executed (`vol_exec > 0`) but that this process
    /// never saw a fill for, which points at a fill missed while the
    /// process was down rather than a healthy, already-tracked position.
    pub fn has_fill_for_order(&self, client_order_id: &str) -> anyhow::Result<bool> {
        let conn = self.conn.lock().unwrap();
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM fills WHERE client_order_id = ?1 LIMIT 1",
                params![client_order_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    /// Appends a fill row. If `exec_id` is `Some` and a concurrent writer
    /// won the race since the caller's own `fill_exists` check (there is
    /// only ever one private-feed task per exchange in this project, so
    /// this is a theoretical rather than observed race), the `UNIQUE`
    /// constraint on `exec_id` makes this a no-op rather than a duplicate
    /// row — the position update the caller already applied in that
    /// narrow window is the one accepted inconsistency this design trades
    /// for not needing a cross-table transaction here.
    pub fn record_fill(&self, fill: &FillRecord) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO fills
                (exec_id, client_order_id, symbol, side, qty, price, realized_pnl_usd, applied_at_ns)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                fill.exec_id,
                fill.client_order_id,
                fill.symbol,
                fill.side,
                fill.qty.to_string(),
                fill.price.to_string(),
                fill.realized_pnl_usd.to_string(),
                fill.applied_at_ns,
            ],
        )?;
        Ok(())
    }

    // -- orders ---------------------------------------------------------

    /// Inserts a brand-new order record, or fully replaces an existing one
    /// with the same `client_order_id` — used at submission time, when the
    /// full record is always known.
    pub fn upsert_order(&self, order: &OrderRecord) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO orders
                (client_order_id, exchange_order_id, symbol, exchange, side, order_type,
                 quantity, limit_price, strategy_id, status, reject_reason, created_at_ns, updated_at_ns)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT(client_order_id) DO UPDATE SET
                exchange_order_id = excluded.exchange_order_id,
                symbol = excluded.symbol,
                exchange = excluded.exchange,
                side = excluded.side,
                order_type = excluded.order_type,
                quantity = excluded.quantity,
                limit_price = excluded.limit_price,
                strategy_id = excluded.strategy_id,
                status = excluded.status,
                reject_reason = excluded.reject_reason,
                updated_at_ns = excluded.updated_at_ns",
            params![
                order.client_order_id,
                order.exchange_order_id,
                order.symbol,
                order.exchange,
                order.side,
                order.order_type,
                order.quantity,
                order.limit_price,
                order.strategy_id,
                order.status,
                order.reject_reason,
                order.created_at_ns,
                order.updated_at_ns,
            ],
        )?;
        Ok(())
    }

    /// A lighter update for a status change arriving on the private feed
    /// (a fill or cancellation), where the full order record isn't in
    /// hand — only touches rows that already exist (an update for an
    /// order this store never saw a submission for, e.g. one placed
    /// outside this process, is skipped rather than guessed at).
    pub fn update_order_status(
        &self,
        client_order_id: &str,
        exchange_order_id: &str,
        status: &str,
        reject_reason: &str,
        updated_at_ns: i64,
    ) -> anyhow::Result<()> {
        if client_order_id.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE orders SET
                exchange_order_id = CASE WHEN ?2 != '' THEN ?2 ELSE exchange_order_id END,
                status = ?3,
                reject_reason = ?4,
                updated_at_ns = ?5
             WHERE client_order_id = ?1",
            params![client_order_id, exchange_order_id, status, reject_reason, updated_at_ns],
        )?;
        Ok(())
    }

    /// Orders not yet in a terminal state (i.e. not filled/canceled/
    /// rejected) — read at startup purely for an operator-visible log
    /// line ("N open orders from before restart"); nothing yet reconciles
    /// these against Kraken's own order state.
    pub fn load_open_orders(&self) -> anyhow::Result<Vec<OrderRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT client_order_id, exchange_order_id, symbol, exchange, side, order_type,
                    quantity, limit_price, strategy_id, status, reject_reason, created_at_ns, updated_at_ns
             FROM orders
             WHERE status NOT IN ('FILLED', 'CANCELED', 'REJECTED')",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(OrderRecord {
                client_order_id: row.get(0)?,
                exchange_order_id: row.get(1)?,
                symbol: row.get(2)?,
                exchange: row.get(3)?,
                side: row.get(4)?,
                order_type: row.get(5)?,
                quantity: row.get(6)?,
                limit_price: row.get(7)?,
                strategy_id: row.get(8)?,
                status: row.get(9)?,
                reject_reason: row.get(10)?,
                created_at_ns: row.get(11)?,
                updated_at_ns: row.get(12)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Looks up a single order by its `client_order_id` — used by
    /// institutional audit Phase 1.3's `CancelOrder` RPC to resolve the
    /// `exchange_order_id` Kraken's `CancelOrder` REST endpoint actually
    /// needs (Python only knows the client-generated id it submitted
    /// with). `None` for an id this store never saw a submission for.
    pub fn get_order(&self, client_order_id: &str) -> anyhow::Result<Option<OrderRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT client_order_id, exchange_order_id, symbol, exchange, side, order_type,
                    quantity, limit_price, strategy_id, status, reject_reason, created_at_ns, updated_at_ns
             FROM orders
             WHERE client_order_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![client_order_id], |row| {
            Ok(OrderRecord {
                client_order_id: row.get(0)?,
                exchange_order_id: row.get(1)?,
                symbol: row.get(2)?,
                exchange: row.get(3)?,
                side: row.get(4)?,
                order_type: row.get(5)?,
                quantity: row.get(6)?,
                limit_price: row.get(7)?,
                strategy_id: row.get(8)?,
                status: row.get(9)?,
                reject_reason: row.get(10)?,
                created_at_ns: row.get(11)?,
                updated_at_ns: row.get(12)?,
            })
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_round_trip() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.load_positions().unwrap().is_empty());

        store.upsert_position("BTC-USD", Decimal::from_str("0.5").unwrap(), Decimal::from_str("30000").unwrap(), 1).unwrap();
        let positions = store.load_positions().unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].symbol, "BTC-USD");
        assert_eq!(positions[0].qty, Decimal::from_str("0.5").unwrap());

        // Upserting the same symbol again replaces, not duplicates.
        store.upsert_position("BTC-USD", Decimal::from_str("0.75").unwrap(), Decimal::from_str("31000").unwrap(), 2).unwrap();
        let positions = store.load_positions().unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].qty, Decimal::from_str("0.75").unwrap());
    }

    #[test]
    fn kill_switch_state_round_trip() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.load_kill_switch_state().unwrap().is_none());

        store.save_kill_switch_state(19_999, Decimal::from_str("-150.25").unwrap()).unwrap();
        let state = store.load_kill_switch_state().unwrap().unwrap();
        assert_eq!(state.day_index, 19_999);
        assert_eq!(state.realized_pnl_usd, Decimal::from_str("-150.25").unwrap());

        // Overwrites in place, single row.
        store.save_kill_switch_state(20_000, Decimal::ZERO).unwrap();
        let state = store.load_kill_switch_state().unwrap().unwrap();
        assert_eq!(state.day_index, 20_000);
        assert_eq!(state.realized_pnl_usd, Decimal::ZERO);
    }

    #[test]
    fn fill_dedup_by_exec_id() {
        let store = Store::open_in_memory().unwrap();
        assert!(!store.fill_exists("EXEC-1").unwrap());

        let fill = FillRecord {
            exec_id: Some("EXEC-1".to_string()),
            client_order_id: Some("co-1".to_string()),
            symbol: "BTC-USD".to_string(),
            side: "BUY".to_string(),
            qty: Decimal::from_str("0.01").unwrap(),
            price: Decimal::from_str("30000").unwrap(),
            realized_pnl_usd: Decimal::ZERO,
            applied_at_ns: 1,
        };
        store.record_fill(&fill).unwrap();
        assert!(store.fill_exists("EXEC-1").unwrap());

        // A redelivery with the same exec_id is ignored, not duplicated.
        store.record_fill(&fill).unwrap();
        let conn = store.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM fills", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn has_fill_for_order_reflects_whether_any_fill_was_recorded() {
        let store = Store::open_in_memory().unwrap();
        assert!(!store.has_fill_for_order("co-1").unwrap());

        store
            .record_fill(&FillRecord {
                exec_id: Some("EXEC-1".to_string()),
                client_order_id: Some("co-1".to_string()),
                symbol: "BTC-USD".to_string(),
                side: "BUY".to_string(),
                qty: Decimal::from_str("0.01").unwrap(),
                price: Decimal::from_str("30000").unwrap(),
                realized_pnl_usd: Decimal::ZERO,
                applied_at_ns: 1,
            })
            .unwrap();

        assert!(store.has_fill_for_order("co-1").unwrap());
        assert!(!store.has_fill_for_order("co-2").unwrap());
    }

    #[test]
    fn fills_without_exec_id_are_never_treated_as_duplicates() {
        let store = Store::open_in_memory().unwrap();
        let fill = FillRecord {
            exec_id: None,
            client_order_id: None,
            symbol: "BTC-USD".to_string(),
            side: "BUY".to_string(),
            qty: Decimal::from_str("0.01").unwrap(),
            price: Decimal::from_str("30000").unwrap(),
            realized_pnl_usd: Decimal::ZERO,
            applied_at_ns: 1,
        };
        store.record_fill(&fill).unwrap();
        store.record_fill(&fill).unwrap();
        let conn = store.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM fills", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn orders_round_trip_and_status_updates() {
        let store = Store::open_in_memory().unwrap();
        let order = OrderRecord {
            client_order_id: "co-1".to_string(),
            exchange_order_id: String::new(),
            symbol: "BTC-USD".to_string(),
            exchange: "kraken".to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            quantity: "0.01".to_string(),
            limit_price: Some("30000".to_string()),
            strategy_id: "test-strategy".to_string(),
            status: "ACCEPTED".to_string(),
            reject_reason: String::new(),
            created_at_ns: 1,
            updated_at_ns: 1,
        };
        store.upsert_order(&order).unwrap();

        let open = store.load_open_orders().unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].client_order_id, "co-1");

        store.update_order_status("co-1", "OK4GJX", "FILLED", "", 2).unwrap();
        let open = store.load_open_orders().unwrap();
        assert!(open.is_empty(), "a filled order should no longer show as open");
    }

    #[test]
    fn update_order_status_is_a_no_op_for_an_unknown_order() {
        let store = Store::open_in_memory().unwrap();
        // Should not error even though "co-unknown" was never inserted.
        store.update_order_status("co-unknown", "", "FILLED", "", 1).unwrap();
        assert!(store.load_open_orders().unwrap().is_empty());
    }

    #[test]
    fn get_order_finds_an_inserted_order_by_client_order_id() {
        let store = Store::open_in_memory().unwrap();
        let order = OrderRecord {
            client_order_id: "co-2".to_string(),
            exchange_order_id: "OK4GJX".to_string(),
            symbol: "ETH-USD".to_string(),
            exchange: "kraken".to_string(),
            side: "SELL".to_string(),
            order_type: "LIMIT".to_string(),
            quantity: "0.5".to_string(),
            limit_price: Some("2500".to_string()),
            strategy_id: "test-strategy".to_string(),
            status: "ACCEPTED".to_string(),
            reject_reason: String::new(),
            created_at_ns: 1,
            updated_at_ns: 1,
        };
        store.upsert_order(&order).unwrap();

        let found = store.get_order("co-2").unwrap().expect("should find the order just inserted");
        assert_eq!(found.exchange_order_id, "OK4GJX");
        assert_eq!(found.symbol, "ETH-USD");
    }

    #[test]
    fn get_order_returns_none_for_an_unknown_client_order_id() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.get_order("co-unknown").unwrap().is_none());
    }
}
