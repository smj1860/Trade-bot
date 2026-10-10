"""Rules-only indicator strategy with confidence-tiered take-profit / stop-loss.

    python scripts/rule_backtest.py --interval 60            # needs SUPABASE_DB_URL
    python scripts/rule_backtest.py --csv-dir candles/       # or CSVs: <SYMBOL>.csv with ts,high,low,close,volume

No model. Each bar close, five readings are combined into one score (about -1 to 1):

  trend       EMA12 vs EMA26, close vs EMA50, EMA50 slope
  momentum    RSI(14) and the MACD histogram (scaled by ATR)
  oscillators Williams %R(14) and CCI(20)
  bands       Bollinger %B(20, 2), read as "riding the band" (trend-following)
  volume      volume / its 20-bar mean scales the whole score (0.7x .. 1.15x)
  volatility  bars with ATR(14) below 0.35% of price are skipped: a target several
              percent away is out of reach there

The sign of the score is the direction and |score| picks a confidence tier. Higher
tiers get a wider take-profit; the stop is a per-config table.
One position per symbol at a time. Exits use each later bar's real high/low; a bar
that touches both barriers counts as a stop; otherwise the position is closed at the
vertical barrier. Everything below is fixed in advance (no tuning on results); three
barrier configs are reported, and that is the whole search.

The report also runs two controls on the SAME entry bars: the reversed direction and
random directions. If the rules had no information the real result would look like
those, and all of them are shown net of the same costs.

Honesty notes: entries fill at the bar close with no slippage and barrier exits at the
barrier price, apart from an extra --stop-slip charged on stop exits. Trades on
different symbols overlap in time and are correlated; confidence intervals therefore
resample whole UTC days.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WARMUP = 60
MIN_ATR_PCT = 0.0035
TIER_THRESHOLDS = (0.35, 0.55, 0.75)
WEIGHTS = {"trend": 0.30, "momentum": 0.30, "osc": 0.20, "bands": 0.20}
_BARRIERS = {
    "A (tp 2/3.5/5%, sl 1/1.5/2%": {"tp": (0.02, 0.035, 0.05), "sl": (0.01, 0.015, 0.02), "hold": 48},
    "B (tp 3/5/8%, sl 1.5/2.5/3.5%": {"tp": (0.03, 0.05, 0.08), "sl": (0.015, 0.025, 0.035), "hold": 96},
    # Stephen's tiers: wider target AND tighter stop as confidence rises
    "C (tp 2.5/4/5%, sl 3/2/1.75%": {"tp": (0.025, 0.04, 0.05), "sl": (0.03, 0.02, 0.0175), "hold": 48},
}


def configs_for(interval_minutes: int) -> dict[str, dict]:
    """Hold times are 48/96 bars on hourly and 4h bars (2/4 days, 8/16 days); daily bars use 20/40."""
    out = {}
    for name, c in _BARRIERS.items():
        hold = c["hold"] if interval_minutes < 1440 else (20 if c["hold"] == 48 else 40)
        out[f"{name}, {hold} bars)"] = {**c, "hold": hold}
    return out


CONFIGS = configs_for(60)
COSTS = (("taker 1.7%", 0.017), ("passive 1.0%", 0.010), ("maker 0.8%", 0.008))


# ---- indicators (vectorised, causal) ---------------------------------------------------

def ema(x: np.ndarray, n: int, alpha: float | None = None) -> np.ndarray:
    a = alpha if alpha is not None else 2.0 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    """fn over the trailing n values (NaN until n values exist)."""
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = fn(w, axis=1)
    return out


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up, dn = ema(np.maximum(d, 0), n, 1.0 / n), ema(np.maximum(-d, 0), n, 1.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dn == 0, 100.0, 100 - 100 / (1 + up / np.where(dn == 0, 1, dn)))


def features(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    prev = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = ema(tr, 14, 1.0 / 14)
    e12, e26, e50 = ema(c, 12), ema(c, 26), ema(c, 50)
    macd = e12 - e26
    hist = macd - ema(macd, 9)
    mid, sd = rolling(c, 20, np.mean), rolling(c, 20, np.std)
    with np.errstate(divide="ignore", invalid="ignore"):
        pctb = (c - (mid - 2 * sd)) / (4 * sd)
        hh, ll = rolling(h, 14, np.max), rolling(l, 14, np.min)
        willr = -100 * (hh - c) / (hh - ll)
        tp = (h + l + c) / 3
        tp_ma = rolling(tp, 20, np.mean)
        md = rolling(tp, 20, lambda w, axis: np.mean(np.abs(w - w.mean(axis=axis, keepdims=True)), axis=axis))
        cci = (tp - tp_ma) / (0.015 * md)
        vr = v / rolling(v, 20, np.mean)
    slope = np.concatenate((np.zeros(10), e50[10:] - e50[:-10]))
    regime = np.sign(rolling(c, 50, np.mean) - rolling(c, 200, np.mean))  # +1 golden-cross regime, -1 death-cross regime
    return {"regime": regime, "e12": e12, "e26": e26, "e50": e50, "slope": slope, "rsi": rsi(c), "hist": hist, "atr": atr,
            "atr_pct": atr / c, "pctb": pctb, "willr": willr, "cci": cci, "vr": vr}


def score(f: dict[str, np.ndarray], c: np.ndarray) -> np.ndarray:
    """Combined score, roughly [-1.15, 1.15] (the volume factor can lift it past 1); NaN where an input is missing."""
    trend = (np.sign(f["e12"] - f["e26"]) + np.sign(c - f["e50"]) + np.sign(f["slope"])) / 3
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = (np.clip((f["rsi"] - 50) / 25, -1, 1) + np.clip(f["hist"] / (0.5 * f["atr"]), -1, 1)) / 2
    osc = (np.clip((f["willr"] + 50) / 50, -1, 1) + np.clip(f["cci"] / 150, -1, 1)) / 2
    bands = np.clip((f["pctb"] - 0.5) * 2, -1, 1)
    raw = WEIGHTS["trend"] * trend + WEIGHTS["momentum"] * mom + WEIGHTS["osc"] * osc + WEIGHTS["bands"] * bands
    return raw * np.clip(0.7 + 0.3 * f["vr"], 0.7, 1.15)


def tier_of(s: float) -> int:
    a = abs(s)
    return sum(a >= t for t in TIER_THRESHOLDS)  # 0 = no trade, 1..3


# ---- trade simulation ----------------------------------------------------------------------

@dataclass(slots=True)
class Trade:
    symbol: str
    ts: int
    i: int
    direction: int
    tier: int
    reason: str  # target | stop | timeout
    gross: float
    bars: int
    score: float = 0.0


def simulate_exit(h, l, c, i: int, direction: int, tp: float, sl: float, hold: int) -> tuple[str, float, int] | None:
    """(reason, gross return, bars held) for a position entered at c[i]; None if the data ends first."""
    if i + hold >= len(c):
        return None
    e = c[i]
    if direction > 0:
        up, dn = e * (1 + tp), e * (1 - sl)
        for j in range(i + 1, i + hold + 1):
            if l[j] <= dn:
                return "stop", -sl, j - i  # also covers a bar touching both
            if h[j] >= up:
                return "target", tp, j - i
    else:
        dn, up = e * (1 - tp), e * (1 + sl)
        for j in range(i + 1, i + hold + 1):
            if h[j] >= up:
                return "stop", -sl, j - i
            if l[j] <= dn:
                return "target", tp, j - i
    return "timeout", direction * (c[i + hold] - e) / e, hold


def contiguous_flags(ts: np.ndarray, interval_s: int, back: int, ahead: int) -> np.ndarray:
    """True where bars [i-back, i+ahead] are consecutive (no missing candles)."""
    gaps = np.concatenate(([0], np.cumsum(np.diff(ts) != interval_s)))
    ok = np.zeros(len(ts), dtype=bool)
    for i in range(back, len(ts) - ahead):
        ok[i] = gaps[i + ahead] == gaps[i - back]
    return ok


def backtest_symbol(symbol: str, ts, h, l, c, v, cfg: dict, interval_s: int = 3600, min_score: float = 0.0,
                    regime: bool = False) -> tuple[list[Trade], list[tuple]]:
    """Trades taken by the rules, plus the entry bars (i, tier, score) for the controls."""
    f = features(h, l, c, v)
    s = score(f, c)
    warm = 200 if regime else WARMUP
    ok = contiguous_flags(ts, interval_s, warm, cfg["hold"])
    trades: list[Trade] = []
    entries: list[tuple] = []
    i = warm
    n = len(c)
    while i < n - cfg["hold"]:
        if ok[i] and np.isfinite(s[i]) and f["atr_pct"][i] >= MIN_ATR_PCT:
            t = tier_of(s[i]) if abs(s[i]) >= min_score else 0
            d = 1 if s[i] > 0 else -1
            if regime and f["regime"][i] != d:
                t = 0
            if t > 0:
                res = simulate_exit(h, l, c, i, d, cfg["tp"][t - 1], cfg["sl"][t - 1], cfg["hold"])
                if res is not None:
                    trades.append(Trade(symbol, int(ts[i]), i, d, t, *res, float(s[i])))
                    entries.append((i, t, s[i]))
                    i += max(res[2], 1)
                    continue
        i += 1
    return trades, entries


def control_trades(symbol: str, ts, h, l, c, entries: list[tuple], cfg: dict, mode: str, seed: int = 0, sign: list[int] | None = None) -> list[Trade]:
    """Same entry bars, different direction: 'reverse' flips the rules' direction, 'random' draws it."""
    rng = random.Random(seed)
    out = []
    for (i, t, sc), s0 in zip(entries, sign):
        d = -s0 if mode == "reverse" else rng.choice((-1, 1))
        res = simulate_exit(h, l, c, i, d, cfg["tp"][t - 1], cfg["sl"][t - 1], cfg["hold"])
        if res is not None:
            out.append(Trade(symbol, int(ts[i]), i, d, t, *res))
    return out


# ---- statistics ----------------------------------------------------------------------------

def net_of(t: Trade, cost: float, stop_slip: float) -> float:
    return t.gross - cost - (stop_slip if t.reason == "stop" else 0.0)


def day_bootstrap(trades: list[Trade], cost: float, stop_slip: float, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean net return per trade with a 95% interval from resampling whole UTC days."""
    if not trades:
        return 0.0, 0.0, 0.0
    days: dict[int, list[float]] = {}
    for t in trades:
        days.setdefault(t.ts // 86400, []).append(net_of(t, cost, stop_slip))
    sums = [(sum(v), len(v)) for v in days.values()]
    rng = random.Random(seed)
    k = len(sums)
    means = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(k):
            a, b = sums[rng.randrange(k)]
            tot += a
            cnt += b
        means.append(tot / cnt)
    means.sort()
    allv = [x for v in days.values() for x in v]
    return sum(allv) / len(allv), means[int(n_boot * 0.025)], means[int(n_boot * 0.975)]


def row(label: str, trades: list[Trade], stop_slip: float) -> str:
    if not trades:
        return f"{label:<22}{0:>6}"
    n = len(trades)
    mix = {r: sum(t.reason == r for t in trades) / n for r in ("target", "stop", "timeout")}
    gross = sum(t.gross for t in trades) / n * 1e4
    cells = []
    for name, cost in COSTS:
        m, lo, hi = day_bootstrap(trades, cost, stop_slip, n_boot=400)
        cells.append(f"{m * 1e4:>+7.0f} [{lo * 1e4:>+5.0f},{hi * 1e4:>+5.0f}]")
    return (f"{label:<22}{n:>6} {100 * mix['target']:>4.0f}/{100 * mix['stop']:>3.0f}/{100 * mix['timeout']:>3.0f} "
            f"{gross:>+7.0f}  " + "  ".join(cells))


def report(name: str, cfg: dict, by_symbol: dict, controls: dict, stop_slip: float, n_bars: int = 0, bars_hours: float = 1.0) -> str:
    trades = [t for ts_ in by_symbol.values() for t in ts_]
    head = f"{'':<22}{'n':>6} {'tgt/stp/tmo%':>13} {'gross':>7}  " + "  ".join(f"{'net bps @ ' + n:<24}" for n, _ in COSTS)
    out = [f"== config {name} =="]
    if n_bars and trades:
        days = n_bars * bars_hours / 24 / max(len(by_symbol), 1)
        out.append(f"activity: {len(trades) / len(by_symbol) / days:.2f} trades per symbol per day; in a position {100 * sum(t.bars for t in trades) / n_bars:.0f}% of bars; "
                   f"average hold {sum(t.bars for t in trades) / len(trades):.1f} bars")
    out += [head, row("all trades", trades, stop_slip)]
    for tier in (1, 2, 3):
        tt = [t for t in trades if t.tier == tier]
        be = (cfg["sl"][tier - 1]) / (cfg["tp"][tier - 1] + cfg["sl"][tier - 1])
        out.append(row(f"tier {tier} (tp {cfg['tp'][tier - 1] * 100:g}/sl {cfg['sl'][tier - 1] * 100:g})", tt, stop_slip)
                   + f"   gross break-even target rate {100 * be:.0f}%")
    out.append(row("long", [t for t in trades if t.direction > 0], stop_slip))
    out.append(row("short", [t for t in trades if t.direction < 0], stop_slip))
    if trades:
        lo_ts, hi_ts = min(t.ts for t in trades), max(t.ts for t in trades)
        span = (hi_ts - lo_ts) / 3 or 1
        for k in range(3):
            part = [t for t in trades if lo_ts + k * span <= t.ts <= lo_ts + (k + 1) * span]
            out.append(row(f"period {k + 1} of 3", part, stop_slip))
    out.append("-- by |score| at entry (is a stricter filter better?) --")
    edges = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 9.0)
    for lo, hi in zip(edges, edges[1:]):
        out.append(row(f"|score| {lo:.2f}-{hi:.2f}" if hi < 9 else f"|score| >= {lo:.2f}", [t for t in trades if lo <= abs(t.score) < hi], stop_slip))
    out.append("-- controls on the same entry bars --")
    out.append(row("reversed direction", controls["reverse"], stop_slip))
    for k, rt in enumerate(controls["random"]):
        out.append(row(f"random direction #{k + 1}", rt, stop_slip))
    per = sorted(((sum(net_of(t, 0.010, stop_slip) for t in ts_) / len(ts_), sym, len(ts_)) for sym, ts_ in by_symbol.items() if ts_), reverse=True)
    if per:
        out.append("by symbol (net bps @ passive 1.00%): " + ", ".join(f"{s} {m * 1e4:+.0f} (n={n})" for m, s, n in per))
    return "\n".join(out)


# ---- data ----------------------------------------------------------------------------------------

def load_db(interval: int) -> dict[str, tuple]:
    from scripts.train_model import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select distinct symbol from ohlc_candles where exchange='kraken' and interval_minutes=%s order by 1", (interval,))
            syms = [r[0] for r in cur.fetchall()]
            out = {}
            for s in syms:
                cur.execute("select extract(epoch from ts)::bigint, high, low, close, volume from ohlc_candles "
                            "where exchange='kraken' and symbol=%s and interval_minutes=%s order by ts", (s, interval))
                r = cur.fetchall()
                a = np.array([[float(x) if x is not None else 0.0 for x in row_] for row_ in r])
                out[s] = (a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4])
            return out
    finally:
        conn.close()


def load_csv(folder: str) -> dict[str, tuple]:
    out = {}
    for p in sorted(Path(folder).glob("*.csv")):
        rows = list(csv.DictReader(p.open()))
        out[p.stem] = (np.array([int(r["ts"]) for r in rows], dtype=np.int64), *(np.array([float(r[k]) for r in rows]) for k in ("high", "low", "close", "volume")))
    return out


def run(data: dict[str, tuple], interval_s: int, stop_slip: float, n_random: int = 3, min_score: float = 0.0,
        regime: bool = False) -> str:
    sections = [f"{len(data)} symbols, {sum(len(v[0]) for v in data.values())} bars; costs are round-trip, plus {stop_slip * 100:.2f}% extra on stop exits",
                f"tier thresholds on |score|: {TIER_THRESHOLDS}; weights {WEIGHTS}; skip if ATR% < {MIN_ATR_PCT * 100:.2f}%"]
    sections.append(f"variant: min |score| {min_score:g}, regime filter (trade only in the direction of SMA50 vs SMA200) {'on' if regime else 'off'}; bar = {interval_s // 60} min")
    for name, cfg in configs_for(interval_s // 60).items():
        by_symbol, rev, rnd = {}, [], [[] for _ in range(n_random)]
        for sym, (ts, h, l, c, v) in data.items():
            tr, entries = backtest_symbol(sym, ts, h, l, c, v, cfg, interval_s, min_score, regime)
            by_symbol[sym] = tr
            signs = [t.direction for t in tr]
            rev += control_trades(sym, ts, h, l, c, entries, cfg, "reverse", sign=signs)
            for k in range(n_random):
                rnd[k] += control_trades(sym, ts, h, l, c, entries, cfg, "random", seed=k + 1, sign=signs)
        sections.append(report(name, cfg, by_symbol, {"reverse": rev, "random": rnd}, stop_slip, sum(len(v[0]) for v in data.values()), interval_s / 3600))
    return "\n\n".join(sections)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=60, help="candle minutes")
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--stop-slip", type=float, default=0.001)
    p.add_argument("--symbols", default="", help="comma-separated subset")
    p.add_argument("--min-score", type=float, default=0.0, help="only enter when |score| is at least this (tiers start at 0.35)")
    p.add_argument("--regime", action="store_true", help="only trade in the direction of SMA50 vs SMA200 (golden / death cross regime)")
    args = p.parse_args(argv)
    data = load_csv(args.csv_dir) if args.csv_dir else load_db(args.interval)
    if args.symbols:
        keep = set(args.symbols.split(","))
        data = {k: v for k, v in data.items() if k in keep}
    text = run(data, args.interval * 60, args.stop_slip, min_score=args.min_score, regime=args.regime)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
