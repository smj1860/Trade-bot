-- Historical data pipeline schema — applied to the Rootstock-vercel Supabase
-- project (ref dfzxqigwpeqdsqetjger). See docs/historical-data-pipeline.md
-- for the design rationale. This project already holds an unrelated
-- homesteading-app schema (users/projects/homestead_plans/etc.) — these
-- three tables are new and namespaced by name only, no FK or dependency on
-- anything already there.
--
-- Applied live via the Supabase MCP connector on 2026-09-26 (mcp__Supabase__
-- apply_migration, migration name "historical_ohlc_trades"). This file is
-- the source-of-truth copy for the repo; re-running it against the same
-- database is idempotent (every statement is IF NOT EXISTS).

create table if not exists ohlc_candles (
    exchange         text        not null default 'kraken',
    symbol           text        not null,   -- normalized BASE-USD, matches config/config.example.toml
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

create index if not exists ohlc_candles_symbol_interval_ts_idx
    on ohlc_candles (symbol, interval_minutes, ts);

create table if not exists trades (
    exchange    text        not null default 'kraken',
    symbol      text        not null,
    trade_id    bigint      not null,  -- Kraken's own per-pair monotonic trade id
    ts          timestamptz not null,
    price       numeric     not null,
    volume      numeric     not null,
    side        text        not null, -- 'buy' | 'sell'
    order_type  text        not null, -- 'market' | 'limit'
    inserted_at timestamptz not null default now(),
    primary key (exchange, symbol, trade_id)
);

create index if not exists trades_symbol_ts_idx
    on trades (symbol, ts);

create table if not exists ohlc_backfill_state (
    exchange         text        not null default 'kraken',
    symbol           text        not null,
    interval_minutes integer     not null,
    earliest_ts      timestamptz,
    latest_ts        timestamptz,
    last_backfill_at timestamptz not null default now(),
    primary key (exchange, symbol, interval_minutes)
);

-- Locked down: this pipeline only ever connects via a direct Postgres
-- connection (SUPABASE_DB_URL), never Supabase's public REST/anon API, so
-- there's no policy this data needs to expose through PostgREST. RLS with
-- no policies blocks the anon/authenticated roles entirely without
-- affecting the direct connection this project's scripts use.
alter table ohlc_candles enable row level security;
alter table trades enable row level security;
alter table ohlc_backfill_state enable row level security;
