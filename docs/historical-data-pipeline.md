# Historical data pipeline

Design for collecting historical market data across the 13-symbol universe
(`docs/training-universe.md`) to eventually train the ML models the Python
strategy layer already has stubs for (`SklearnModelWrapper`/
`TorchModelWrapper`, currently only exercised against toy data). Decided
with Stephen on 2026-09-26.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Data source | Kraken's own public REST (`/0/public/OHLC`, `/0/public/Trades`) | Free, no new dependency, and it's the same venue the bot actually trades on — training data matches execution reality exactly. |
| Storage | Postgres via Supabase (project `Rootstock-vercel`, ref `dfzxqigwpeqdsqetjger`) | Stephen already has this Supabase project (previously paused/unused — confirmed empty before reusing it), and Postgres is a solid fit for the bulk scan/aggregate access patterns model training needs, without standing up a new service. |
| Implementation language | Python | This is an offline batch job, not a latency-sensitive path — it doesn't belong in the Rust core's 24/7 hot loop. Lives alongside the strategy layer's own Python/pandas/pyarrow tooling. |

This keeps the historical pipeline entirely separate from `rust-core`'s
SQLite persistence store, which stays scoped to *operational* state
(positions, fills, orders) — this pipeline is *analytical* data for
training, a different access pattern and a different database on purpose.

## Where it lives

New top-level `historical-data/` directory (sibling to `rust-core` and
`python-strategy`), not inside `python-strategy/`, because it's a standalone
batch tool with its own dependencies (`requests`, `psycopg2`) — it doesn't
run as part of the live strategy engine and shouldn't drag its
dependencies into that process.

```
historical-data/
  requirements.txt
  symbols.py       # loads the same 13-symbol universe from config/config.example.toml
  kraken_client.py # Kraken public REST client: OHLC + Trades, with pagination
  db.py            # Postgres connection (psycopg2), schema application
  backfill_ohlc.py # CLI: backfill OHLC candles for one or all symbols/intervals
  backfill_trades.py # CLI: backfill raw trades for one symbol over a bounded window
migrations/
  0001_historical_ohlc_trades.sql
```

## Schema (Postgres / Supabase)

```sql
create table if not exists ohlc_candles (
    exchange         text        not null default 'kraken',
    symbol           text        not null,   -- normalized BASE-USD, matches config.example.toml
    interval_minutes integer     not null,   -- Kraken OHLC interval: 1,5,15,30,60,240,1440,10080,21600
    ts               timestamptz not null,   -- candle open time
    open             numeric     not null,
    high             numeric     not null,
    low              numeric     not null,
    close            numeric     not null,
    vwap             numeric,
    volume           numeric     not null,
    trade_count      integer,
    inserted_at      timestamptz not null default now(),
    primary key (exchange, symbol, interval_minutes, ts)
);

create table if not exists trades (
    exchange   text        not null default 'kraken',
    symbol     text        not null,
    trade_id   bigint      not null,  -- Kraken's own per-pair monotonic trade id
    ts         timestamptz not null,
    price      numeric     not null,
    volume     numeric     not null,
    side       text        not null, -- 'buy' | 'sell'
    order_type text        not null, -- 'market' | 'limit'
    inserted_at timestamptz not null default now(),
    primary key (exchange, symbol, trade_id)
);

create table if not exists ohlc_backfill_state (
    exchange         text        not null default 'kraken',
    symbol           text        not null,
    interval_minutes integer     not null,
    earliest_ts      timestamptz,
    latest_ts         timestamptz,
    last_backfill_at timestamptz not null default now(),
    primary key (exchange, symbol, interval_minutes)
);
```

`ohlc_backfill_state` exists so a backfill run is resumable/idempotent —
it records the oldest and newest candle actually stored per
`(symbol, interval)`, so re-running the backfill script only fills the gap
instead of re-fetching everything from scratch.

## A real limitation: Kraken's OHLC endpoint is NOT deep history

This is the one design point worth being explicit about rather than
letting the schema imply more than the source actually delivers. Kraken's
public `/0/public/OHLC` only retains a bounded window of history per
candle resolution — roughly the most recent ~720 candles *at that
resolution* are reliably available (a bit more via repeated `since`
paging, but nowhere near "years" at 1-minute granularity). Concretely:

- **1-minute candles**: only around the last half-day to a day is
  actually retrievable, regardless of how far back `since` is set.
- **1-hour / 4-hour candles**: several weeks to a few months.
- **1-day (1440) candles**: Kraken does retain meaningfully deep daily
  history (years, for pairs that have existed that long) — this is the
  resolution to lean on for a genuinely deep multi-year backfill.

**Practical consequence for the backfill scripts**: `backfill_ohlc.py`
backfills the 1440 (daily) and 60 (hourly) intervals as deep as Kraken
will actually return for all 13 symbols by default — that's real,
multi-year-or-as-far-as-the-pair-exists history, at a footprint of a few
thousand rows per symbol. Finer resolutions (1/5/15-minute) are available
too, but only cover what Kraken currently exposes (days, not years) at
backfill time — meaningful depth at 1-minute resolution only accumulates
by running the backfill script repeatedly over time (e.g. on a schedule),
not from a single one-off backfill. This is the same "accumulate going
forward" pattern this project already uses for other data, just applied
to candles instead of order-book state.

`backfill_trades.py` backfills raw trade-level data (price/volume/side per
individual trade), which Kraken does expose full history for via its
`since`-cursor pagination — but a genuinely complete tick-level history for
13 symbols going back years is a very large pull (Kraken's own rate limits
mean this could take a long time and a lot of storage). Rather than
silently doing that, the script defaults to a bounded lookback window
(configurable via `--since`) so a first run is a deliberate, sized choice
rather than an unbounded multi-day pull kicked off by accident.

## Credentials

The Postgres connection string is read from the `SUPABASE_DB_URL`
environment variable only — never written to a config file or committed,
the same pattern this project already uses for `KRAKEN_API_KEY`/
`KRAKEN_API_SECRET`. Get it from the Supabase dashboard for the
`Rootstock-vercel` project (Project Settings → Database → Connection
string), or via `mcp__Supabase__get_project_url`/the dashboard's
connection-pooling settings.

## Supabase project note

The `Rootstock-vercel` project (ref `dfzxqigwpeqdsqetjger`) was picked
because Stephen already had it — but it turned out not to be empty: it
already holds an unrelated homesteading/land-planning app's schema
(`users`, `projects`, `homestead_plans`, `budget_land_plans`, `zip_zones`,
`affiliate_products`, `tool_reviews`, etc.). The three tables this pipeline
added (`ohlc_candles`, `trades`, `ohlc_backfill_state`) are namespaced by
name only, with no foreign keys or dependency on anything already in that
project, and have Row Level Security enabled with no policies — the
anon/public API key has zero access to them, since this pipeline only ever
talks to Postgres directly. Confirmed with Stephen before proceeding
(2026-09-26) rather than assumed.

## Verified (2026-09-26)

- `kraken_client.py`'s `fetch_ohlc`/`fetch_ohlc_all` and
  `fetch_trades`/`fetch_trades_window` were run live against Kraken's real
  REST API: 721 daily (1440-min) candles for BTC-USD came back correctly
  parsed, going back to Oct 2024 (confirming daily-resolution depth really
  is years, not days), and a real page of 1000 trades came back correctly
  parsed too.
- The Postgres schema, the `upsert_candles`/`upsert_trades` conflict
  handling, and `ohlc_backfill_state` tracking were smoke-tested directly
  against the live Supabase database (via the Supabase SQL tool, using
  real candles fetched from Kraken above) — 5 rows inserted, still 5 rows
  after a repeat upsert of the same data, confirming the `on conflict`
  path is idempotent rather than duplicating.
- **Not yet exercised**: the actual `db.py`/`psycopg2` connection path
  from `backfill_ohlc.py`/`backfill_trades.py` themselves. That requires
  `SUPABASE_DB_URL` with the real database password, which isn't
  retrievable via the Supabase API — only Stephen has it (Project
  Settings → Database → Connection string) or can reset it from the
  dashboard. Once that's set, running `backfill_ohlc.py` for one symbol is
  the remaining end-to-end check.

## Scheduled ingestion (.github/workflows/historical-backfill.yml)

`backfill_ohlc.py` now also runs automatically, hourly, via GitHub
Actions — on GitHub's own servers, not Stephen's laptop or any server he
has to rent or maintain. Requires one repository secret,
`SUPABASE_DB_URL` (Settings -> Secrets and variables -> Actions -> New
repository secret in the GitHub UI — never committed to the repo itself).
Re-running the backfill hourly is safe and cheap: it upserts by
`(exchange, symbol, interval, ts)`, so re-fetching an already-stored
candle just updates it in place. The workflow also supports
`workflow_dispatch`, so Stephen can trigger a run by hand from the GitHub
Actions tab with no terminal at all.

## Bulk CSV import (historical-data/import_csv.py)

For loading a large, already-downloaded historical dataset all at once —
e.g. Kraken's own downloadable per-pair OHLCVT dumps, which cover much
deeper history than the public REST API's retention window allows
`backfill_ohlc.py` to reach (see the retention caveat above) — rather than
waiting for that depth to accumulate one REST call at a time.

Supports two input shapes:
- `--format kraken-dump`: Kraken's own no-header CSV dumps
  (`timestamp,open,high,low,close,volume,trades`).
- `--format generic`: any CSV with a header row, matching common column
  name variants case-insensitively (`timestamp`/`time`/`date`/`datetime`,
  `open`/`high`/`low`/`close`, `volume`/`vol`, optional `vwap`,
  optional `trades`/`trade_count`/`count`). The timestamp column can be
  unix seconds or a parseable date/time string.

This is meant to run **alongside** the scheduled job, not instead of it —
a CSV import gets deep history in one shot; the hourly job keeps it
current afterward. Both write to the same table with the same upsert key,
so an overlapping CSV import is safe to re-run.

Needs `pandas`, kept in a separate `requirements-csv.txt` so the core
REST pipeline doesn't need it (same "opt-in extra dependency" pattern as
`python-strategy/requirements-ml.txt`).

**Verified**: parsing logic tested against synthetic CSVs in both
formats — confirmed correct `Candle` construction for the `kraken-dump`
format, and column auto-detection plus timestamp parsing for the
`generic` format. Caught and fixed a real bug during that testing: the
first timestamp-parsing implementation assumed pandas always represents
parsed dates as nanosecond-resolution `datetime64`, which isn't
guaranteed (this pandas version parses to microsecond resolution) — it
was silently producing timestamps 1000x too small. Fixed by converting
through numpy's `datetime64[s]` cast, which does the unit conversion
explicitly rather than assuming a resolution. Not yet tested against a
real multi-megabyte Kraken dump file end-to-end (only synthetic
few-row CSVs so far).

## Not yet done

- No feature-engineering or training-set-assembly layer yet — this
  pipeline only gets raw OHLC/trades into Postgres. Turning that into
  labeled training examples for `SklearnModelWrapper`/`TorchModelWrapper`
  is separate, future work.
- No data-quality/gap-detection tooling yet (e.g. alerting if a symbol's
  backfill state falls behind, or if the scheduled GitHub Actions run
  starts failing).
- `backfill_trades.py` (raw trade-level data) isn't in the scheduled
  workflow yet — only OHLC candles run on a schedule so far.
