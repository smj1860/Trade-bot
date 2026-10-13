"""Regime-split rules (ADX trend / range / cash) backtest, rules fixed before looking at any result.

    python scripts/regime_backtest.py --csv-dir candles/4h --interval 240 --label orig \
        --symbols BTC,ETH,... --start 2023-10-14 --end 2026-10-13

Long only (spot). Signals at a bar's close, entry at that close, exits on later bars' real high/low.
One position per symbol. ADX is Wilder's 14.

REGIMES (per bar close)
  trend-up    ADX > 25, +DI > -DI, and baseline up:
                EMA baseline      EMA20 > EMA50 and close > EMA50
                Ichimoku baseline close above the cloud (tenkan 9, kijun 26, span B 52, shifted 26)
  trend-down  ADX > 25, -DI > +DI, baseline down  -> CASH, no trades
  range       ADX < 20
  (20 <= ADX <= 25 -> no trades)

TREND MODE entries (either): Donchian-20 breakout (close above the prior 20 bars' highest high), or an
  RSI reset (RSI(14) was <= 50 on the previous bar, is > 50 now, and stayed >= 40 over the 5 bars before).
  Exits: close below EMA20 (fast MA break), or a -5% price hard stop, or 90 bars.
RANGE MODE entries: RSI(14) < 30 or close below the lower Bollinger band (20, 2).
  Exits (variants): a resting target that nets 0.8% / 1.0% / 1.2% after maker fees on both legs
  (0.40% each), or the text's own exit (close >= middle band, or RSI > 50). All: a stop that loses 2%
  net of fees (about -1.2% price), 30 bars max, a bar touching both target and stop counts as the stop.

COSTS: round trip 0.8% (maker both legs) and 1.0% (passive-ish), plus 0.10% extra on stop exits.
CONTROLS: the same exits from random entry bars drawn from the bars where the regime allows an entry
  (same count per symbol), 3 draws. Day-block bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_backtest import ema, rsi  # noqa: E402

FEE = 0.004  # maker fee per leg used to gross up the targets
STOP_NET = 0.02
HARD_STOP = 0.05
STOP_SLIP = 0.001
COSTS = (("maker 0.8%", 0.008), ("passive 1.0%", 0.010))


# ---- indicators ---------------------------------------------------------------------------

def wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    out[n] = np.nansum(x[1 : n + 1]) / 1.0  # sum seed (Wilder's smoothing keeps sums for TR/DM)
    for i in range(n + 1, len(x)):
        out[i] = out[i - 1] - out[i - 1] / n + x[i]
    return out


def adx_di(h, l, c, n: int = 14):
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr, p, m = wilder(tr, n), wilder(pdm, n), wilder(mdm, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * p / atr, 100 * m / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full(len(c), np.nan)
    first = 2 * n
    if len(c) > first:
        adx[first] = np.nanmean(dx[n + 1 : first + 1])
        for i in range(first + 1, len(c)):
            adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx, pdi, mdi


def roll(x, n, fn):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


def prev_extreme(x, n, fn):
    """fn over the n bars BEFORE each bar (excludes the bar itself)."""
    r = roll(x, n, fn)
    return np.concatenate([[np.nan], r[:-1]])


@dataclass
class Ind:
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    adx: np.ndarray
    pdi: np.ndarray
    mdi: np.ndarray
    ema20: np.ndarray
    ema50: np.ndarray
    cloud_top: np.ndarray
    cloud_bot: np.ndarray
    rsi: np.ndarray
    don_hi: np.ndarray
    bb_mid: np.ndarray
    bb_lo: np.ndarray


def indicators(h, l, c) -> Ind:
    adx, pdi, mdi = adx_di(h, l, c)
    tenkan = (roll(h, 9, np.max) + roll(l, 9, np.min)) / 2
    kijun = (roll(h, 26, np.max) + roll(l, 26, np.min)) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (roll(h, 52, np.max) + roll(l, 52, np.min)) / 2
    sh = lambda x: np.concatenate([np.full(26, np.nan), x[:-26]])  # noqa: E731  cloud is drawn 26 bars ahead
    a, b = sh(span_a), sh(span_b)
    mid = roll(c, 20, np.mean)
    sd = roll(c, 20, np.std)
    return Ind(h, l, c, adx, pdi, mdi, ema(c, 20), ema(c, 50), np.fmax(a, b), np.fmin(a, b), rsi(c, 14),
               prev_extreme(h, 20, np.max), mid, mid - 2 * sd)


def regimes(ix: Ind, baseline: str):
    """(trend_up, trend_down, range) boolean arrays."""
    strong = ix.adx > 25
    if baseline == "ema":
        up = (ix.ema20 > ix.ema50) & (ix.c > ix.ema50)
        dn = (ix.ema20 < ix.ema50) & (ix.c < ix.ema50)
    else:
        up, dn = ix.c > ix.cloud_top, ix.c < ix.cloud_bot
    return strong & (ix.pdi > ix.mdi) & up, strong & (ix.mdi > ix.pdi) & dn, ix.adx < 20


# ---- targets / exits ---------------------------------------------------------------------------

def gross_target(net: float) -> float:
    """Price rise that nets `net` after the maker fee on both legs."""
    return (1 + FEE) * (1 + net) / (1 - FEE) - 1


def stop_drop(net_loss: float) -> float:
    return 1 - (1 + FEE) * (1 - net_loss) / (1 - FEE)


@dataclass
class Trade:
    symbol: str
    ts: int
    bars: int
    ret: float  # gross price return
    reason: str


def exit_trend(ix: Ind, i: int, n: int):
    entry = ix.c[i]
    stop = entry * (1 - HARD_STOP)
    for j in range(i + 1, min(i + 91, n)):
        if ix.l[j] <= stop:
            return j, stop / entry - 1, "stop"
        if ix.c[j] < ix.ema20[j]:
            return j, ix.c[j] / entry - 1, "ma_break"
    j = min(i + 90, n - 1)
    return j, ix.c[j] / entry - 1, "timeout"


def exit_range_target(net: float):
    tgt, drop = gross_target(net), stop_drop(STOP_NET)

    def f(ix: Ind, i: int, n: int):
        entry = ix.c[i]
        t, s = entry * (1 + tgt), entry * (1 - drop)
        for j in range(i + 1, min(i + 31, n)):
            if ix.l[j] <= s:
                return j, -drop, "stop"
            if ix.h[j] >= t:
                return j, tgt, "target"
        j = min(i + 30, n - 1)
        return j, ix.c[j] / entry - 1, "timeout"

    return f


def exit_range_mid(ix: Ind, i: int, n: int):
    entry, drop = ix.c[i], stop_drop(STOP_NET)
    s = entry * (1 - drop)
    for j in range(i + 1, min(i + 31, n)):
        if ix.l[j] <= s:
            return j, -drop, "stop"
        if ix.c[j] >= ix.bb_mid[j] or ix.rsi[j] > 50:
            return j, ix.c[j] / entry - 1, "mid_or_rsi"
    j = min(i + 30, n - 1)
    return j, ix.c[j] / entry - 1, "timeout"


def run_entries(sym: str, ts, ix: Ind, signal: np.ndarray, exit_fn, start_i: int) -> list[Trade]:
    out, n, busy = [], len(ix.c), -1
    for i in np.flatnonzero(signal):
        if i <= busy or i < start_i or i >= n - 2:
            continue
        j, r, why = exit_fn(ix, i, n)
        out.append(Trade(sym, int(ts[i]), j - i, r, why))
        busy = j
    return out


def entries_trend(ix: Ind, up: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        don = ix.c > ix.don_hi
        r1 = np.concatenate([[np.nan], ix.rsi[:-1]])
        rmin = prev_extreme(ix.rsi, 5, np.min)
        rsi_reset = (r1 <= 50) & (ix.rsi > 50) & (rmin >= 40)
    return up & (don | rsi_reset)


def entries_range(ix: Ind, rg: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return rg & ((ix.rsi < 30) | (ix.c < ix.bb_lo))


# ---- stats ---------------------------------------------------------------------------

def net(t: Trade, cost: float) -> float:
    return t.ret - cost - (STOP_SLIP if t.reason == "stop" else 0.0)


def boot(trades: list[Trade], cost: float, n_boot=1000, seed=0):
    if not trades:
        return 0.0, 0.0, 0.0
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.ts // 86400].append(net(t, cost))
    days = list(by_day.values())
    rng = np.random.default_rng(seed)
    mean = np.mean([x for d in days for x in d])
    ms = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        v = [x for k in pick for x in days[k]]
        ms.append(np.mean(v))
    return mean, float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def bps(x: float) -> float:
    return x * 1e4


def fmt_row(label: str, tr: list[Trade], ctrl: list[list[Trade]], n_sym: int, months: float) -> str:
    if not tr:
        return f"{label:<26}{0:>6}"
    g = np.mean([t.ret for t in tr])
    cg = np.mean([np.mean([t.ret for t in c]) for c in ctrl if c]) if any(ctrl) else float("nan")
    m8, lo8, hi8 = boot(tr, COSTS[0][1])
    m10 = np.mean([net(t, COSTS[1][1]) for t in tr])
    win = np.mean([net(t, COSTS[0][1]) > 0 for t in tr])
    srt = sorted(tr, key=lambda t: t.ts)
    th = [np.mean([net(t, COSTS[0][1]) for t in srt[k * len(srt) // 3 : (k + 1) * len(srt) // 3]]) for k in range(3)]
    why = defaultdict(int)
    for t in tr:
        why[t.reason] += 1
    mix = "/".join(f"{k[:3]}{100 * v / len(tr):.0f}" for k, v in sorted(why.items()))
    return (f"{label:<26}{len(tr):>6} {len(tr) / n_sym / max(months, 1):>5.2f} {np.mean([t.bars for t in tr]):>6.1f} {100 * win:>4.0f}% "
            f"{bps(g):>+7.0f} {bps(cg):>+7.0f} {bps(m8):>+7.0f} {'[' + format(bps(lo8), '+.0f') + ',' + format(bps(hi8), '+.0f') + ']':>15} "
            f"{bps(m10):>+7.0f}  {bps(th[0]):>+5.0f}/{bps(th[1]):>+5.0f}/{bps(th[2]):>+5.0f}  {mix}")


HEAD = (f"{'variant':<26}{'n':>6} {'/sym/mo':>5} {'bars':>6} {'win':>5} {'gross':>7} {'ctrl':>7} {'net0.8':>7} {'(95% CI)':>15} "
        f"{'net1.0':>7}  {'thirds (net0.8)':<19} exits")


def load(folder: str, symbols: list[str]):
    out = {}
    for s in symbols:
        p = Path(folder) / f"{s}-USD.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open()))
        out[s] = tuple(np.array([float(r[k]) for r in rows]) for k in ("ts", "high", "low", "close"))
    return out


def evaluate(data, start_ts: int, end_ts: int, n_ctrl: int = 3) -> str:
    variants = [
        ("trend  EMA baseline", "trend", "ema", exit_trend),
        ("trend  Ichimoku baseline", "trend", "ich", exit_trend),
        ("range  target net 0.8%", "range", "ema", exit_range_target(0.008)),
        ("range  target net 1.0%", "range", "ema", exit_range_target(0.010)),
        ("range  target net 1.2%", "range", "ema", exit_range_target(0.012)),
        ("range  mid-band / RSI>50", "range", "ema", exit_range_mid),
    ]
    res = {v[0]: [] for v in variants}
    ctl = {v[0]: [[] for _ in range(n_ctrl)] for v in variants}
    cash_bars = tot_bars = 0
    bh = []
    for sym, (ts, h, l, c) in data.items():
        if len(c) < 400:
            continue
        ix = indicators(h, l, c)
        win = (ts >= start_ts) & (ts <= end_ts)
        if win.sum() < 100:
            continue
        idx = np.flatnonzero(win)
        s0, s1 = idx[0], idx[-1]
        warm = max(s0, 120)
        bh.append(c[s1] / c[warm] - 1)
        up_e, dn_e, rg = regimes(ix, "ema")
        tot_bars += s1 - warm
        cash_bars += int(dn_e[warm:s1].sum())
        for k, (label, mode, base, fn) in enumerate(variants):
            up, dn, rg_b = regimes(ix, base)
            sig = entries_trend(ix, up) if mode == "trend" else entries_range(ix, rg_b)
            sig[: warm] = False
            sig[s1 - 1 :] = False
            tr = run_entries(sym, ts, ix, sig, fn, warm)
            res[label] += tr
            elig = np.flatnonzero((up if mode == "trend" else rg_b) & (np.arange(len(c)) >= warm) & (np.arange(len(c)) < s1 - 1))
            for d in range(n_ctrl):
                if len(tr) and len(elig):
                    rng = random.Random(zlib.crc32(f"{sym}|{label}|{d}".encode()))
                    pick = np.zeros(len(c), bool)
                    pick[rng.sample(list(elig), min(len(elig), max(len(tr) * 3, 1)))] = True  # oversample, one-at-a-time rule thins it
                    ctrl_tr = run_entries(sym, ts, ix, pick, fn, warm)[: len(tr)]
                    ctl[label][d] += ctrl_tr
    months = (end_ts - start_ts) / (30.44 * 86400)
    lines = [f"symbols: {len(data)}; buy-and-hold per coin over the window (mean): {100 * np.mean(bh):+.0f}%; "
             f"bars in a trend-down (cash) regime: {100 * cash_bars / max(tot_bars, 1):.0f}%", HEAD]
    for label, *_ in variants:
        lines.append(fmt_row(label, res[label], ctl[label], len(data), months))
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--symbols", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args(argv)
    ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())  # noqa: E731
    data = load(a.csv_dir, [s.strip().upper() for s in a.symbols.split(",") if s.strip()])
    print(f"== {a.label} | {a.csv_dir} | {a.start} to {a.end}")
    print(evaluate(data, ts(a.start), ts(a.end)))


if __name__ == "__main__":
    main()
