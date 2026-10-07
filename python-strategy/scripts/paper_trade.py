"""Forward paper trading for a saved triple-barrier model.

Run hourly (see .github/workflows/paper-trade.yml). Each run, for every
registered arm (a model stored by scripts/register_paper_model.py):

  1. fetches the latest hourly candles for each symbol from Kraken's public
     REST API (no keys, no orders -- nothing here can place a trade);
  2. resolves any open paper trade whose barrier was touched, or whose
     vertical barrier (``horizon`` bars) has passed, using the later bars'
     real high/low (strategy/paper.py);
  3. scores the most recently CLOSED bar of every symbol with the same
     feature code the model was trained on (train_model.features_at) and
     opens one simulated trade per symbol: long if the model says up, short
     otherwise, entered at that bar's close -- the same entry the backtest
     assumes.

Every signal is traded, as in the backtest's per-trade P&L, and trades may
overlap (unit notional each); the figure of merit is mean net return per
trade, compared against the model's walk-forward expectation. Unlike the
backtest, timeouts and same-bar double touches are scored, not dropped (see
strategy/paper.py). A skipped hourly run just leaves a gap in signals; exits
are always resolved from candle data, so they stay correct.

Needs SUPABASE_DB_URL (the model and the trade log live there).
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import connect, dataset_warmup, features_at  # noqa: E402
from strategy.paper import Bar, closed_bars, direction_from_prediction, resolve_trade  # noqa: E402

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.example.toml"


def load_pairs(path: Path = CONFIG_PATH) -> dict[str, str]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return {e["symbol"]: e["rest_native_symbol"] for e in raw.get("symbols", []) if e.get("enabled", True)}


def fetch_candles(pair: str, interval_minutes: int, retries: int = 3) -> list[list]:
    """Raw Kraken rows [time, o, h, l, c, vwap, volume, count], oldest first,
    including the still-forming last candle."""
    import requests

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(KRAKEN_OHLC_URL, params={"pair": pair, "interval": interval_minutes}, timeout=20)
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            result = body["result"]
            key = next(k for k in result if k != "last")
            return result[key]
        except Exception as e:  # noqa: BLE001 -- retried, then surfaced to the caller
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Kraken OHLC fetch failed for {pair}: {last_err}")


def prepare(raw: list[list], interval_seconds: int, now: int) -> tuple[list[list], float | None]:
    """(closed candles, price of the still-forming candle or None)."""
    closed = closed_bars(raw, interval_seconds, now)
    forming = raw[len(closed)] if len(raw) > len(closed) else None
    return closed, (float(forming[4]) if forming is not None else None)


def decide_signals(model, meta: dict, candles: dict[str, list[list]], observed: dict[str, float | None], now: int) -> tuple[list[dict], list[str]]:
    """Scores the latest closed bar of each symbol. Returns (signals, notes)
    where notes explain every symbol that was skipped."""
    interval_seconds = int(meta["interval_minutes"]) * 60
    expected_ts = (now // interval_seconds) * interval_seconds - interval_seconds
    w = meta["windows"]
    warmup = dataset_warmup(
        w["sma_window"], w["ema_window"], w["rsi_window"], w["vol_window"], w["bar_momentum_window"],
        w["bollinger_window"], w["ao_slow_window"], w["macd_slow_window"], w["macd_signal_window"],
        w["cci_window"], w["williams_r_window"], meta["extended"], meta["fib"],
    )
    order = meta["feature_order"]
    signals, notes = [], []
    for symbol, rows in candles.items():
        barrier = meta["barrier_by_symbol"].get(symbol)
        if barrier is None:
            notes.append(f"{symbol}: no barrier in model meta (not in training set) -- skipped")
            continue
        if len(rows) < warmup + 1:
            notes.append(f"{symbol}: only {len(rows)} closed candles, need {warmup + 1} -- skipped")
            continue
        if int(rows[-1][0]) != expected_ts:
            notes.append(f"{symbol}: latest closed candle {rows[-1][0]} != expected {expected_ts} (stale feed) -- skipped")
            continue
        highs = [float(r[2]) for r in rows]
        lows = [float(r[3]) for r in rows]
        closes = [float(r[4]) for r in rows]
        volumes = [float(r[6]) for r in rows]
        mids = [(h + l) / 2.0 for h, l in zip(highs, lows)]
        feats = features_at(closes, mids, highs, lows, volumes, len(rows) - 1, **w, extended=meta["extended"], fib=meta["fib"])
        vector = [feats[name] for name in order]
        pred = int(model.predict([vector])[0])
        proba = model.predict_proba([vector])[0]
        proba_up = float(proba[list(model.classes_).index(1)])
        signals.append({
            "symbol": symbol,
            "entry_ts": expected_ts,
            "direction": direction_from_prediction(pred),
            "proba_up": proba_up,
            "entry_price": closes[-1],
            "observed_price": observed.get(symbol),
            "barrier_pct": float(barrier),
            "horizon_bars": int(meta["horizon"]),
            "round_trip_cost": float(meta["round_trip_cost"]),
        })
    return signals, notes


def resolve_open(open_trades: list[dict], candles: dict[str, list[list]]) -> list[tuple[dict, object]]:
    """Pairs each open trade with its Resolution (only those now resolvable)."""
    out = []
    for t in open_trades:
        rows = candles.get(t["symbol"])
        if not rows:
            continue
        after = [Bar(int(r[0]), float(r[2]), float(r[3]), float(r[4])) for r in rows if int(r[0]) > t["entry_ts"]]
        res = resolve_trade(
            t["entry_price"], t["direction"], t["barrier_pct"], t["horizon_bars"], t["round_trip_cost"], after
        )
        if res is not None:
            out.append((t, res))
    return out


# --- database ---------------------------------------------------------------


def load_model(conn, arm: str):
    import joblib

    with conn.cursor() as cur:
        cur.execute("select model_bytes, meta from paper_models where arm = %s", (arm,))
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"error: no model registered for arm {arm!r} (run register_paper_model.py)")
    return joblib.load(io.BytesIO(bytes(row[0]))), row[1]


def list_arms(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("select arm from paper_models order by arm")
        return [r[0] for r in cur.fetchall()]


def load_open_trades(conn, arm: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "select symbol, extract(epoch from entry_ts)::bigint, direction, entry_price, barrier_pct, "
            "horizon_bars, round_trip_cost from paper_trades where arm = %s and status = 'open'",
            (arm,),
        )
        return [
            {"symbol": r[0], "entry_ts": int(r[1]), "direction": int(r[2]), "entry_price": float(r[3]),
             "barrier_pct": float(r[4]), "horizon_bars": int(r[5]), "round_trip_cost": float(r[6])}
            for r in cur.fetchall()
        ]


def _ts(unix: int) -> datetime:
    return datetime.fromtimestamp(unix, tz=timezone.utc)


def run_arm(conn, arm: str, pairs: dict[str, str], now: int, dry_run: bool) -> None:
    model, meta = load_model(conn, arm)
    interval_minutes = int(meta["interval_minutes"])
    candles: dict[str, list[list]] = {}
    observed: dict[str, float | None] = {}
    fetch_notes = []
    for symbol in meta["symbols"]:
        pair = pairs.get(symbol)
        if pair is None:
            fetch_notes.append(f"{symbol}: not in config -- skipped")
            continue
        try:
            closed, forming_price = prepare(fetch_candles(pair, interval_minutes), interval_minutes * 60, now)
        except RuntimeError as e:
            fetch_notes.append(str(e))
            continue
        candles[symbol] = closed
        observed[symbol] = forming_price

    resolved = resolve_open(load_open_trades(conn, arm), candles)
    signals, notes = decide_signals(model, meta, candles, observed, now)
    notes = fetch_notes + notes

    print(f"[{arm}] resolved {len(resolved)} trade(s), {len(signals)} new signal(s), {len(notes)} note(s)")
    for t, r in resolved:
        print(f"  closed {t['symbol']} {'LONG' if t['direction'] == 1 else 'SHORT'} {r.exit_reason} net={r.net_return:+.4f} bars={r.holding_bars}")
    for s in signals:
        print(f"  open   {s['symbol']} {'LONG' if s['direction'] == 1 else 'SHORT'} p_up={s['proba_up']:.3f} @ {s['entry_price']}")
    for n in notes:
        print(f"  note: {n}")
    if dry_run:
        print("  (dry run -- nothing written)")
        return

    with conn.cursor() as cur:
        for t, r in resolved:
            cur.execute(
                "update paper_trades set status='closed', exit_reason=%s, exit_ts=%s, exit_price=%s, "
                "holding_bars=%s, ambiguous=%s, gross_return=%s, net_return=%s, closed_at=now() "
                "where arm=%s and symbol=%s and entry_ts=%s",
                (r.exit_reason, _ts(r.exit_ts), r.exit_price, r.holding_bars, r.ambiguous, r.gross_return,
                 r.net_return, arm, t["symbol"], _ts(t["entry_ts"])),
            )
        for s in signals:
            cur.execute(
                "insert into paper_trades (arm, symbol, entry_ts, direction, proba_up, entry_price, observed_price, "
                "barrier_pct, horizon_bars, round_trip_cost) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "on conflict (arm, symbol, entry_ts) do nothing",
                (arm, s["symbol"], _ts(s["entry_ts"]), s["direction"], s["proba_up"], s["entry_price"],
                 s["observed_price"], s["barrier_pct"], s["horizon_bars"], s["round_trip_cost"]),
            )
        cur.execute(
            "insert into paper_runs (arm, run_ts, n_opened, n_closed, notes) values (%s, %s, %s, %s, %s) "
            "on conflict (arm, run_ts) do nothing",
            (arm, _ts(now), len(signals), len(resolved), "; ".join(notes) or None),
        )
    conn.commit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="all", help='Arm name, or "all" for every registered arm.')
    p.add_argument("--dry-run", action="store_true", help="Fetch, score and print; write nothing.")
    args = p.parse_args()

    pairs = load_pairs()
    now = int(time.time())
    conn = connect()
    try:
        arms = list_arms(conn) if args.arm == "all" else [args.arm]
        if not arms:
            print("no arms registered -- nothing to do")
            return
        for arm in arms:
            run_arm(conn, arm, pairs, now, args.dry_run)
    finally:
        conn.close()
        try:
            import resource  # Linux/macOS only; a cheap check that the job fits a small server

            peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
            print(f"peak memory {peak_mb:.0f} MB")
        except Exception:  # noqa: BLE001 -- diagnostics only
            pass


if __name__ == "__main__":
    main()
