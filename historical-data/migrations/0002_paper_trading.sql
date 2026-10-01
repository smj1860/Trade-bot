-- Forward paper trading (python-strategy/scripts/paper_trade.py). Same
-- Supabase project as 0001. Idempotent. RLS is enabled with no policies so
-- the tables are NOT readable through the public PostgREST API; the jobs
-- connect with the direct database URL, which bypasses RLS.

create table if not exists paper_models (
    arm         text        primary key,
    model_bytes bytea       not null,   -- joblib-serialized sklearn model
    meta        jsonb       not null,   -- sidecar from train_model.py (feature order, windows, barriers, ...)
    created_at  timestamptz not null default now()
);

create table if not exists paper_trades (
    arm              text             not null,
    symbol           text             not null,
    entry_ts         timestamptz      not null,   -- open time of the signal bar; entry is at its close
    direction        smallint         not null check (direction in (1, -1)),   -- 1 long, -1 short
    proba_up         double precision not null,
    entry_price      numeric          not null,
    observed_price   numeric,                     -- price of the forming candle when the job ran (entry drift diagnostic)
    barrier_pct      double precision not null,
    horizon_bars     integer          not null,
    round_trip_cost  double precision not null,
    status           text             not null default 'open' check (status in ('open', 'closed')),
    exit_reason      text,                        -- target | stop | timeout | ambiguous_stop
    exit_ts          timestamptz,
    exit_price       numeric,
    holding_bars     integer,
    ambiguous        boolean,
    gross_return     double precision,            -- direction-adjusted, before costs, no borrow
    net_return       double precision,            -- gross_return - round_trip_cost
    created_at       timestamptz      not null default now(),
    closed_at        timestamptz,
    primary key (arm, symbol, entry_ts)
);

create index if not exists paper_trades_arm_status_idx on paper_trades (arm, status);

create table if not exists paper_runs (
    arm        text        not null,
    run_ts     timestamptz not null,
    n_opened   integer     not null,
    n_closed   integer     not null,
    notes      text,
    primary key (arm, run_ts)
);

alter table paper_models enable row level security;
alter table paper_trades enable row level security;
alter table paper_runs   enable row level security;
