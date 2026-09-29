# Crypto Trading Bot — Project Handoff

**Purpose of this document:** a complete state-of-the-project summary for starting a fresh chat once this one hits its context limit. If you're reading this as a new Claude session, this is everything you need to pick up where things left off — architecture, what's built, what's been tried and failed, what's running right now, standing rules, and what to do next.

**Last updated:** 2026-09-28, end of session. Written by Claude (session `01FfL97QbFKk8YmRF4yrzdAF`) at Stephen's request, for a brand-new chat to continue from.

---

## 1. Who / what this project is

Stephen (sjones@lakemartindelivery.com) is building a modular crypto trading bot: a **Rust core** (24/7 background service — exchange WebSockets, order books, execution, risk guardrails) talking over **gRPC** to a **Python strategy layer** (ML model loading, signal generation, portfolio management). Exchange: **Kraken**. Currency-agnostic by design.

- **GitHub repo:** `smj1860/Trade-bot` (note: GitHub shows a "repository moved" notice pointing at `smj1860/trade-bot` — same repo, case-only rename, pushes still work against the old remote URL, just prints a notice).
- **claude.ai Project:** "Crypto Trading Bot" (project ID `01a0dbd7-505b-7480-a437-3097a4c489b7`), holds three living docs read/written via the `Projects` tool:
  - `claude/model-training-progress.md` — the full, detailed history of every model-training experiment (this doc's contents are summarized below, but that doc has the full tables/numbers).
  - `claude/institutional-audit-implementation-plan.md` — the phased plan for hardening the system to institutional standards.
  - `claude/institutional-audit-2026-09-27.md` — the original audit document that plan implements (Critical / High-priority / Institutional-tier action items).
- **Database:** Supabase Postgres, project "Rootstock-vercel" (shared with Stephen's unrelated "Rootstock" homesteading-app project — just uses the same Postgres instance, `ohlc_candles`, `ohlc_backfill_state`, `trades` tables live there).
- **Develops from:** a Windows laptop, but all real engineering happens through Claude sessions running in a Linux container with the repo cloned, dispatching real work via GitHub Actions (this project has essentially no local dev loop — everything either runs as an Actions workflow or as Rust/Python code in-repo).

## 2. Standing rules — do not violate these

1. **Credentials are env-var only, never in chat, never in files.** `SUPABASE_DB_URL`, `KRAKEN_API_KEY`, `KRAKEN_API_SECRET`, `ALERT_WEBHOOK_URL` live only as GitHub Actions repo secrets. A fresh Claude session has **no local DB credential** — `SUPABASE_DB_URL` is not set in the container's environment. This means: any DB read/write (backfill, resampling, training) must run as a **GitHub Actions `workflow_dispatch`**, not as a local script — see §5 for the dispatch pattern. `GITHUB_TOKEN` (for the GitHub REST API, to dispatch/poll workflows) **is** available in the container's env.
2. **Trained model files are gitignored** (`python-strategy/models/`), never committed. Generated protobuf Python stubs are also gitignored.
3. **LINK-USD/SOL-USD backfill skip rule:** if a symbol fails a backfill retry a second time in any round, skip it and report as missing rather than retrying again. (Not triggered so far — no failures have occurred.)
4. **Every commit ends with this attribution footer** (binding, use on every commit):
   ```
   Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01FfL97QbFKk8YmRF4yrzdAF
   ```
   (A new session will have a different session URL in its own system reminder — use whatever that session's reminder gives, not the literal one above.)
5. **Workflow for every unit of engineering work:** implement → test → document (update the project doc) → commit with attribution footer → push. Don't skip the test or doc step even under time pressure.
6. **Phase 1.4 (live funded Kraken account verification) requires Stephen's physical presence** — never attempt this unattended, even if everything else seems ready. This is the one standing item that cannot be automated away.
7. **Ask before starting expensive/ambiguous work**; otherwise, for clear requests, just start and report. Stephen actively steers priorities turn-by-turn — check in before big new directions, but don't ask permission for obvious follow-through on something he's already approved.
8. **Never wire a model into `strategy_config.toml` (`strategy.model.kind = "sklearn"`)** until one actually, consistently beats naive baselines on both accuracy and simulated net P&L across every fold, clears `scripts/evaluate_holdout.py`'s sealed holdout, **and** clears `scripts/promotion_gate.py`'s deflated-Sharpe/MinTRL checks. None of that has happened yet — see §7.

## 3. Architecture at a glance

```
Rust core (rust-core/)                     Python strategy (python-strategy/)
├── kraken.rs, kraken_private_ws.rs        ├── strategy/
│     WebSocket feeds (book + trade)       │    indicators.py  — pure, dimensionless-ratio
├── kraken_rest.rs — REST + signing        │                     feature functions (sma_ratio,
├── orderbook.rs — book state              │                     rsi, macd_histogram, cci, etc.)
├── order.rs — order submission/tracking   │    bars.py        — BarAggregator, live tick→bar
├── risk.rs — position/PnL/guardrails      │    features.py    — FeatureEngine wraps indicators
├── guardrails.rs, stop_loss.rs            │    dsr.py         — Probabilistic/Deflated Sharpe
├── reconcile.rs — position reconciliation │                     (stdlib-only, used by
├── heartbeat.rs — dead-man's-switch       │                     evaluate_holdout.py)
├── observability.rs — periodic PnL/       ├── scripts/
│     exposure + performance summary       │    train_model.py — the whole training/validation
├── performance.rs — Sharpe/Sortino/       │                     pipeline (see §6)
│     Calmar/max-drawdown (NEW this        │    promotion_gate.py — NEW this session (§4.4)
│     session, §4.3)                       │    evaluate_holdout.py — sealed final-holdout check
├── persistence.rs — SQLite store          ├── engine.py — live strategy engine (dry_run_only
│     (positions, fills, orders)           │                mode exists; no live paper-trading
├── alerting.rs — webhook alerts           │                fill simulation yet)
├── checksum.rs, market_data.rs, proto.rs
└── main.rs

historical-data/ (Python, standalone from python-strategy/)
├── backfill_ohlc_from_trades.py  — deep OHLC backfill from Kraken's Trades endpoint
├── plan_backfill_deepening.py    — computes each symbol's next 90-day-older window
├── resample_ohlc.py              — NEW this session (§4.1): derives 2/4/6/12hr/daily
│                                    candles from 60-min data already in ohlc_candles
├── symbols.py, db.py, kraken_client.py, ohlc_from_trades.py

.github/workflows/
├── historical-backfill-trades.yml     — manual, one-symbol-at-a-time deep backfill
├── historical-backfill-deepening.yml  — automated, self-continuing, scheduled every 6h
│                                         (plans + dispatches next window per symbol)
├── resample-ohlc.yml                  — NEW this session: manual dispatch of resample_ohlc.py
└── train-model.yml                    — manual dispatch of train_model.py, now with
                                          --gate/--gate-enforce inputs (NEW this session)
```

**Key design fact:** every feature `train_model.py`/`strategy/indicators.py` computes is a dimensionless ratio (never a raw price or raw volume) — this is what made pooling symbols together safe (no symbol's price scale dominates) and what makes the pipeline candle-interval-agnostic (works identically at any `--interval`, no code changes needed).

## 4. What shipped THIS session (2026-09-28, second half)

Four things were done in direct response to Stephen asking "is there other work we can do while the backfill runs":

### 4.1 Fixed a real logging bug
`train-model.yml`'s `tee train_output.log` step only captured **stdout**, but `train_model.py`'s actual result-summary prints go to **stderr** — every training run's uploaded log artifact was nearly empty. Fixed to `2>&1 | tee train_output.log`. Every result documented in `claude/model-training-progress.md` from before this fix was read via the raw GitHub Actions job console log instead (`GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs`, following the redirect with `curl -L`) — that workaround is no longer needed for new runs.

### 4.2 Multi-timeframe candle infrastructure + test
- Built `historical-data/resample_ohlc.py`: derives 120/240/360/720/1440-minute (2/4/6/12hr, daily) candles by aggregating the 60-minute candles already in `ohlc_candles` (correct OHLCV math: open=first, high=max, low=min, close=last, volume=sum, vwap=volume-weighted, trade_count=sum) — much cheaper than re-pulling raw trades per timeframe. 7 unit tests. New `resample-ohlc.yml` workflow (manual dispatch) ran this for all 14 symbols.
- **Stephen's stated live-trading intent** (asked mid-session, refined twice): he plans to trade primarily on **1hr, 4hr, 6hr, and daily** candles (not the full 1/2/4/6/12hr set he first mentioned), with the exception of longer-held positions.
- **Stephen also asked**: should the strategy/model be candle-agnostic entirely, since indicators work the same across timeframes? Answer given: the feature/model machinery already IS candle-agnostic (see the dimensionless-ratio point above) — but pooling *multiple timeframes of the same symbol* together is a different, untested bet than pooling *symbols* together, because a daily candle is literally built from the 24 hourly candles under it (highly overlapping/autocorrelated), unlike genuinely independent symbol price series. **This is flagged as a real open test, not yet run** — see §8.
- **Test actually run**: trained/validated (`--symbol all --label-scheme triple-barrier --horizon 4 --folds 5 --kind gboost`, same config as the existing hourly baseline) at 240min/360min/1440min. Result:

  | interval | avg accuracy | vs majority/persistence | avg net P&L/trade | folds beating both (of 5) |
  |---|---|---|---|---|
  | 60min (1hr, baseline) | 0.524 | 0.505 / 0.512 | -0.0162 | 2 |
  | 240min (4hr) | 0.514 | 0.513 / 0.506 | -0.0165 | 1 |
  | 360min (6hr) | 0.513 | 0.523 / 0.509 | -0.0165 | **0 (below its own majority baseline)** |
  | 1440min (daily) | 0.493 | 0.478 / 0.488 | -0.0172 | 1 |

  **Verdict: coarser candles make the sample-size problem worse, not better.** None beat hourly; daily (only 300-650 candles per symbol currently) was worst. This confirms, from a different angle, the same root cause the per-symbol test found (see §7): not enough rows, full stop.

### 4.3 Portfolio-level performance instrumentation (Rust) — institutional audit Phase 3.4
New `rust-core/src/performance.rs`: the live risk engine now computes and logs **Sharpe, Sortino, Calmar, and max drawdown** directly from its own realized fills (`persistence::Store::fills_since`, a new query), on the same periodic cadence `observability.rs` already used for its PnL/exposure summary, over a new configurable `performance_window_days` (default 90). 20 new tests (176 Rust tests total, all passing). Computed on daily realized-PnL-in-USD directly (not a percentage return series) — documented in the module as a deliberate, provably-lossless simplification (each ratio is scale-invariant to whatever capital base you'd otherwise divide through by, since this project doesn't track a separate account-equity figure).

### 4.4 Formal model promotion gate — institutional audit Phase 3.2
New `python-strategy/scripts/promotion_gate.py`: implements, from the actual academic literature (not approximations):
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) — corrects for how many configurations were actually tried (multiple-testing bias).
- **Minimum Track Record Length** (same paper) — how many observations are needed before the candidate's own Sharpe is statistically meaningful.
- **Probability of Backtest Overfitting via CSCV** (Bailey, Borwein, Lopez de Prado & Zhu, 2015) — combinatorially-symmetric cross-validation over multiple trial configurations' return series.

24 unit tests against constructed toy data with known expected behavior. Wired into `train_model.py`: the production (`--folds 1`) path now collects its own per-trade net-P&L series (refactored `simulate_net_pnl_series`/`simulate_triple_barrier_net_pnl_series` out of the existing total/count functions), and new `--gate`/`--gate-enforce`/`--gate-trial-sharpes-file`/`--gate-min-dsr`/`--gate-max-pbo` CLI flags run the gate before saving — `--gate-enforce` refuses to save on FAIL. `train-model.yml` exposes `gate`/`gate_enforce` as dispatch inputs. **No model has cleared walk-forward validation yet, so nothing has actually been gated for real** — this is infrastructure waiting for the day a model earns it.

**All commits this session** (chronological, all pushed to `main`):
1. `ab57b89` — fix train-model.yml stderr logging
2. `9142d07` — add resample_ohlc.py + resample-ohlc.yml
3. `6ba29ec` — add performance.rs (Sharpe/Sortino/Calmar/drawdown)
4. `4e79877` — add promotion_gate.py + train_model.py/train-model.yml wiring

Full test count after this session: **279 Python tests, 176 Rust tests, all passing.**

## 5. How to actually do DB-touching work (important — read before trying anything)

A fresh Claude session has **no `SUPABASE_DB_URL` in its environment**. Every script that touches the database (`backfill_ohlc_from_trades.py`, `plan_backfill_deepening.py`, `resample_ohlc.py`, `train_model.py`) must run as a **GitHub Actions workflow dispatch**, using `GITHUB_TOKEN` (which IS available) to call the REST API. Pattern used throughout this session:

```bash
# 1. Find a workflow's numeric ID (only needed once per workflow, or after renaming):
curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/smj1860/Trade-bot/actions/workflows"

# 2. Dispatch it (note: MUST include Content-Type header or the session's proxy
#    returns a synthetic HTTP 415 — this bit multiple people in this project's history):
curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  "https://api.github.com/repos/smj1860/Trade-bot/actions/workflows/<ID>/dispatches" \
  -d '{"ref":"main","inputs":{"key":"value"}}'

# 3. Poll for the run and its jobs:
curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/smj1860/Trade-bot/actions/workflows/<ID>/runs?per_page=5"

# 4. Read a job's full console log (needed for anything that prints to stderr,
#    or before the 2026-09-28 tee fix landed for train-model.yml specifically):
curl -sS -L -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/smj1860/Trade-bot/actions/jobs/<JOB_ID>/logs"
```

Known workflow IDs as of this session (re-verify if workflows have been renamed/added since):
- `historical-backfill-deepening.yml` → `369228229`
- `train-model.yml` → `368644989`
- `resample-ohlc.yml` → `369641930`

## 6. The training pipeline (`python-strategy/scripts/train_model.py`) — what it does

Queries Supabase's `ohlc_candles`, computes the same 14 dimensionless-ratio features the live engine computes (`strategy/indicators.py`), labels bars, and trains a classifier (logistic or gradient-boosted). Key flags:

- `--symbol` — one symbol, comma-list, or `all` (pools every symbol at that interval).
- `--interval` — candle resolution in minutes (60=1hr default; 240/360/1440 etc. now populated via resampling, see §4.2).
- `--horizon N` — label bar `i` by the move to bar `i+N`.
- `--label-scheme {fixed-horizon,triple-barrier}` — triple-barrier (walks forward checking real high/low for upper/lower barrier touches) is the one actually used in every real sweep; fixed-horizon is the older, simpler scheme.
- `--profit-margin X` — how much real edge above breakeven cost the barrier requires.
- `--folds N` — `1` (default) = single time-ordered split, trains AND saves a `.joblib`; `>1` = walk-forward validation only, no save, used for every sweep.
- `--holdout-days N` — seals the most recent N days out of the sweep/train path entirely (institutional audit Phase 2.5); `scripts/evaluate_holdout.py` is the separate, one-time script allowed to look at that sealed window, gated on DSR ≥ 0.95.
- `--gate` / `--gate-enforce` / `--gate-trial-sharpes-file` — NEW this session, see §4.4.
- Purged + embargoed walk-forward CV is implemented correctly (no label leakage across fold boundaries), in both the validation path and the production single-split path.

## 7. Model training: the honest bottom line (full detail in `claude/model-training-progress.md`)

**Eleven-plus rounds of real experiments, none has produced a model that beats naive baselines (majority-class, persistence) consistently across folds, on accuracy or on simulated net P&L.** In order:

1. Three rounds of adding technical indicators (11 → 14 features) — no edge appeared.
2. Multi-symbol pooling (13→14 symbols) — no edge, but didn't hurt either.
3. Label engineering: horizon+magnitude threshold, then net-P&L simulation, then triple-barrier labeling tied to real trading costs.
4. Walk-forward validation + purged/embargoed CV (fixed a real leakage bug that existed in both the validation tool and the production training path).
5. A 5-horizon sweep (4/8/12/24/48 bars) — horizon=4 was the least-bad, but only beat both baselines in 2/5 folds.
6. A profit-margin sweep (0%→5%) — looked like a clean monotonic improvement through 2% (4/5 folds beating baselines), then **broke down at 3%+** as sample size collapsed (test-fold size fell from ~6,360 rows at 0% to ~324 at 5%).
7. **The per-symbol-vs-pooled comparison (28 runs, all 14 symbols individually at 0% and 2% margin) — the key finding**: every single symbol run alone does WORSE than the pooled model, and gets worse (not better) going from 0%→2% margin per-symbol. **Conclusion: pooling was never masking a hidden per-symbol edge — it was the thing making results look as good as they did**, by giving the model enough rows to train/test on at all. Sample size, not the pooling methodology, is the load-bearing constraint.
8. **The multi-timeframe test (this session, §4.2)** confirmed the same root cause from a different angle: going to coarser candles (fewer bars covering the same wall-clock history) makes things worse, not better.

**The single most important open lever**: the 36-month backfill deepening (already running automatically, see §8) — once symbols individually have enough history to survive 5-fold walk-forward validation *after* margin filtering, re-run the per-symbol comparison; it isn't informative yet at current depths.

**Genuinely open, not-yet-tested idea flagged this session**: pooling multiple timeframes of the SAME symbol together (1hr+4hr+6hr+daily rows concatenated) — distinct from switching to one coarser timeframe, since same-symbol timeframes are highly autocorrelated in a way cross-symbol pooling isn't. Needs its own empirical test.

## 8. Backfill deepening — current status and how to check/kick it

**What it is**: `.github/workflows/historical-backfill-deepening.yml` runs on a schedule (every 6 hours) plus can be manually dispatched. It computes (via `plan_backfill_deepening.py`) each symbol's next 90-day-older window and dispatches all symbols still short of the target (currently **1080 days = 36 months**) in parallel, each in its own `concurrency:` group (so an overrunning window — BTC-USD/ETH-USD can take hours — is never double-dispatched, just queues).

**Status as of end of this session (2026-09-28, ~00:31 UTC)**: every symbol has a window either running or queued. **BTC-USD is the persistent bottleneck** — it's the slowest symbol by far (a single 90-day window can take 2-4+ hours) and was still stuck around 271→361 days (~9-12 months) while every other symbol had already raced ahead to 630-720 days (21-24 months). ETH-USD is the second-slowest but well ahead of BTC-USD (already past 450+ days). **To manually kick another round** (e.g., if it looks stalled or you want to accelerate before the next 6-hour tick):

```bash
curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  "https://api.github.com/repos/smj1860/Trade-bot/actions/workflows/369228229/dispatches" \
  -d '{"ref":"main","inputs":{"target_days":"1080","interval":"60"}}'
```

This is safe to run anytime — concurrency groups mean it never duplicates in-flight work, it just queues a symbol's next window if one is already running. **No action is needed for this to keep progressing on its own** — it will keep running unattended until every symbol reaches 1080 days.

## 9. Institutional audit — full status

From `claude/institutional-audit-2026-09-27.md` / `claude/institutional-audit-implementation-plan.md`:

- **Critical tier (6 items)**: all done, except Phase 1.4 (live funded Kraken verification — blocked on Stephen's physical presence, see §2 rule 6).
- **High-priority tier (6 items)**: all done, except Phase 2.6 (calibrate rate limiter against real account tier — blocked on 1.4).
- **Institutional tier (5 items)**:
  1. Multi-venue execution / smart order routing — **not started**.
  2. Formal model promotion gate — **done this session** (§4.4).
  3. HA/redundancy for the Rust core (single process holds book/risk/execution state, no hot standby) — **not started**.
  4. Portfolio-level performance instrumentation — **done this session** (§4.3).
  5. Stress/scenario testing beyond the historical sample (synthetic or historical-crisis replay) — **not started**.

## 10. Immediate next steps (pick up here)

In rough priority order, none blocking on the others:

1. **Just keep letting the backfill deepen** — no action needed, but worth checking in on periodically (§8) since BTC-USD is the pacing item.
2. **Once backfill is meaningfully deeper**, re-run the per-symbol-vs-pooled comparison (§7) — this is the test that's currently underpowered and will become informative.
3. **Test pooling multiple timeframes of the same symbol** (§4.2's open question) — doesn't need to wait on backfill depth, could be tried now with current data.
4. Remaining institutional-tier items not started: multi-venue routing, Rust core HA/redundancy, stress/scenario testing (none blocked on model progress).
5. Live/forward paper trading (extending `engine.py`'s `dry_run_only` path to simulate fills going forward) — discussed, not built.
6. Stochastic Oscillator/ATR/ADX flagged as possible next indicators if more signal is ever needed, but several rounds of indicator additions haven't shown signal — probably not the next lever to pull.

## 11. Where the full detail lives

This document is a summary. For exact numbers, tables, and reasoning behind every experiment, read (via the `Projects` tool, `project_read`):
- `claude/model-training-progress.md` — the complete model-training history (this doc's §4.2/§7 are condensed from it).
- `claude/institutional-audit-implementation-plan.md` — the full phased audit plan.
- `claude/institutional-audit-2026-09-27.md` — the original audit findings.
