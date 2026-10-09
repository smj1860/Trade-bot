"""Replay the forward paper-trade signals against the recorded order book.

    python -m execution_sim.evaluate --data /srv/recorder --arms ext-fib-m4,ext-fib-m2
    python -m execution_sim.evaluate --fetch 2026/10/08 --cache /root/rec-cache --trades-json trades.json

Trades come from the ``paper_trades`` table (needs SUPABASE_DB_URL) or a JSON
file of rows (arm, symbol, entry_ts, direction, barrier_pct, horizon_bars,
status, exit_reason, net_return, round_trip_cost). Each distinct trade is
replayed once per scenario and fill-model variant, see lifecycle.py for the
rules. Only trades whose whole life lies inside gap-free recording count.

The question this answers: of the ~1.7% round-trip cost the paper trades
assume, how much of it can passive execution realistically recover, and does
the strategy's gross edge then clear what is left? Everything about passive
fills is counterfactual; the queue variants bracket the main uncertainty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from .engine import NS, Fees, SimConfig
from .lifecycle import SCENARIOS, Outcome, RunStats, TradeSpec, returns, run_lifecycles
from .replay import MultiReplayer, find_recordings
from .report import BPS, bootstrap_mean

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.example.toml"
DEFAULT_FEES = "0.004:0.008"
DEFAULT_VARIANTS = "1:0,0.5:0.5,2:0"


def load_config(path: Path) -> dict[str, dict]:
    """internal symbol (BTC-USD) -> {native, tick, decimals}"""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    out = {}
    for e in raw.get("symbols", []):
        tick = e["tick_size"]
        out[e["symbol"]] = {"native": e["exchange_native_symbol"], "tick": float(tick),
                            "decimals": len(tick.split(".")[1]) if "." in tick else 0}
    return out


def parse_pairs(text: str) -> list[tuple[float, float]]:
    return [tuple(float(x) for x in part.split(":")) for part in text.split(",")]  # type: ignore[misc]


def load_rows_db(arms: list[str] | None, since_ts: int | None) -> list[dict]:
    import psycopg2

    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise SystemExit("error: SUPABASE_DB_URL is not set (see deploy/paper-trader/README.md)")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            q = ("select t.arm, t.symbol, extract(epoch from t.entry_ts)::bigint, t.direction, t.barrier_pct, "
                 "t.horizon_bars, t.status, t.exit_reason, t.net_return, t.round_trip_cost, "
                 "coalesce((m.meta->>'interval_minutes')::int, 60) from paper_trades t left join paper_models m on m.arm = t.arm")
            cond, args = [], []
            if arms:
                cond.append("t.arm = any(%s)")
                args.append(arms)
            if since_ts:
                cond.append("extract(epoch from t.entry_ts) >= %s")
                args.append(since_ts)
            cur.execute(q + (" where " + " and ".join(cond) if cond else ""), args)
            keys = ("arm", "symbol", "entry_ts", "direction", "barrier_pct", "horizon_bars", "status", "exit_reason",
                    "net_return", "round_trip_cost", "interval_minutes")
            return [dict(zip(keys, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def norm_direction(v) -> int:
    if isinstance(v, str):
        return 1 if v.lower() in ("long", "1", "buy") else -1
    return 1 if int(v) > 0 else -1


def build_specs(rows: list[dict], cfg: dict[str, dict]) -> tuple[list[TradeSpec], dict[int, list[dict]], list[str]]:
    """Distinct trades -> specs; also uid -> the rows (arms) that took each."""
    specs: dict[tuple, TradeSpec] = {}
    owners: dict[int, list[dict]] = defaultdict(list)
    skipped = Counter()
    for r in rows:
        info = cfg.get(r["symbol"])
        if info is None:
            skipped[r["symbol"]] += 1
            continue
        interval = int(r.get("interval_minutes") or 60) * 60
        direction = norm_direction(r["direction"])
        key = (info["native"], int(r["entry_ts"]), direction, float(r["barrier_pct"]), int(r["horizon_bars"]), interval)
        if key not in specs:
            specs[key] = TradeSpec(len(specs), info["native"], direction, (int(r["entry_ts"]) + interval) * NS,
                                   float(r["barrier_pct"]), int(r["horizon_bars"]) * interval * NS)
        owners[specs[key].uid].append(r)
    notes = [f"skipped {n} trade(s) in symbols missing from the config: {s}" for s, n in skipped.items()]
    return list(specs.values()), owners, notes


# ---- report ----------------------------------------------------------------------------

def _paper_ref(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get("status") == "closed" and r.get("net_return") is not None:
            return r
    return None


def _norm_reason(r: str | None) -> str | None:
    return "stop" if r == "ambiguous_stop" else r


def format_report(outcomes: list[Outcome], owners: dict[int, list[dict]], fees_list: list[tuple[str, Fees]],
                  variants: list[tuple[float, float]], stats: RunStats, primary_arms: list[str] | None = None) -> str:
    by: dict[tuple[str, int], dict[int, Outcome]] = defaultdict(dict)
    for o in outcomes:
        if not o.aborted:
            by[(o.scenario, o.variant)][o.uid] = o
    names = [s.name for s in SCENARIOS]
    passive = {s.name: s.passive for s in SCENARIOS}

    def get(scen: str, vi: int) -> dict[int, Outcome]:
        return by.get((scen, vi if passive[scen] else 0), {})

    lines = [
        f"trades: {stats.specs} distinct; {stats.started} had recording at entry, {stats.no_data_at_entry} did not; "
        f"{stats.lifecycles} lifecycles run, {stats.aborted} aborted by recording gaps, {stats.unfinished} unfinished when data ended",
    ]
    base = get("all-taker", 0)
    for vi, (qf, cc) in enumerate(variants):
        common = set.intersection(*(set(get(n, vi)) for n in names)) if all(get(n, vi) for n in names) else set()
        paper_rows = {u: _paper_ref(owners[u]) for u in common}
        paper = [r for r in paper_rows.values() if r]
        for label, fees in fees_list:
            lines += ["", f"== queue_factor {qf:g}, cancel_credit {cc:g} | {label} (maker {fees.maker * 100:.2f}% / taker {fees.taker * 100:.2f}%) | "
                          f"{len(common)} trades completed in every scenario =="]
            lines.append(f"{'scenario':<15}{'n':>5} {'gross bps':>10} {'net bps':>9} {'(95% CI)':>19} {'vs all-taker':>13} {'(95% CI)':>19} {'win%':>5} "
                         f"{'tgt/stp/tmo %':>14} {'maker%':>7}")
            ref_net = {u: returns(base[u], fees)[2] * BPS for u in common} if common else {}
            for n in names:
                outs = [get(n, vi)[u] for u in sorted(common)]
                if not outs:
                    lines.append(f"{n:<15}{0:>5}")
                    continue
                gross = [returns(o, fees)[0] * BPS for o in outs]
                net = [returns(o, fees)[2] * BPS for o in outs]
                m, lo, hi = bootstrap_mean(net)
                diff = [net[i] - ref_net[u] for i, u in enumerate(sorted(common))]
                dm, dlo, dhi = bootstrap_mean(diff) if n != "all-taker" else (0, 0, 0)
                reasons = Counter(o.reason for o in outs)
                tot = len(outs)
                mq = sum(f.qty for o in outs for f in o.entry_fills + o.exit_fills if f.maker)
                aq = sum(f.qty for o in outs for f in o.entry_fills + o.exit_fills)
                d = f"{'-':>13} {'':>19}" if n == "all-taker" else f"{dm:>+13.1f} {'[' + format(dlo, '+.1f') + ', ' + format(dhi, '+.1f') + ']':>19}"
                lines.append(
                    f"{n:<15}{tot:>5} {sum(gross) / tot:>+10.1f} {m:>+9.1f} {'[' + format(lo, '+.1f') + ', ' + format(hi, '+.1f') + ']':>19} {d} "
                    f"{100 * sum(1 for x in net if x > 0) / tot:>4.0f}% "
                    f"{100 * reasons['target'] / tot:>4.0f}/{100 * reasons['stop'] / tot:>2.0f}/{100 * reasons['timeout'] / tot:>2.0f} "
                    f"{100 * mq / aq if aq else 0:>6.0f}%")
            if paper:
                pg = [(r["net_return"] + r["round_trip_cost"]) * BPS for r in paper]
                pn = [r["net_return"] * BPS for r in paper]
                lines.append(f"{'paper (candles)':<15}{len(paper):>5} {sum(pg) / len(pg):>+10.1f} {sum(pn) / len(pn):>+9.1f}   "
                             f"(its own {paper[0]['round_trip_cost'] * 100:.2f}% round-trip cost; closed trades only)")
        # validation: do replayed exits agree with the candle-based paper exits?
        agree, tot, mism = 0, 0, Counter()
        for u, o in base.items():
            pr = _paper_ref(owners[u])
            if pr is None or not pr.get("exit_reason"):
                continue
            tot += 1
            a, b = o.reason, _norm_reason(pr["exit_reason"])
            if a == b:
                agree += 1
            else:
                mism[(b, a)] += 1
        if vi == 0 and tot:
            lines += ["", f"validation: replayed exit reason (all-taker) matches the paper's candle-based exit on {agree}/{tot} closed trades ({100 * agree / tot:.0f}%)"]
            if mism:
                lines.append("  mismatches (paper -> replay): " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in mism.most_common()))
    # per arm, first variant, first fee tier
    if fees_list:
        label, fees = fees_list[0]
        lines += ["", f"== per arm ({label}, queue_factor {variants[0][0]:g}/cancel_credit {variants[0][1]:g}): mean net bps, replayed trades only =="]
        lines.append(f"{'arm':<22}{'n':>5} " + "".join(f"{n:>15}" for n in names) + f"{'paper':>10}")
        arm_uids: dict[str, list[int]] = defaultdict(list)
        for u, rows in owners.items():
            for r in rows:
                arm_uids[r["arm"]].append(u)
        for arm in sorted(arm_uids):
            if primary_arms and arm not in primary_arms:
                continue
            uids = [u for u in arm_uids[arm] if all(u in get(n, 0) for n in names)]
            if not uids:
                continue
            cells = [sum(returns(get(n, 0)[u], fees)[2] for u in uids) / len(uids) * BPS for n in names]
            pr = [r for u in uids for r in owners[u] if r["arm"] == arm and r.get("status") == "closed" and r.get("net_return") is not None]
            pcell = f"{sum(r['net_return'] for r in pr) / len(pr) * BPS:>+10.1f}" if pr else f"{'-':>10}"
            lines.append(f"{arm:<22}{len(uids):>5} " + "".join(f"{c:>+15.1f}" for c in cells) + pcell)
    lines += ["", "Passive fills are counterfactual (our orders never changed the recorded book); the queue variants bracket that uncertainty.",
              "The paper column uses its own flat round-trip cost and exact barrier prices, so it is not directly comparable to the replayed columns."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*", default=[], help="recorder files or directories")
    p.add_argument("--fetch", nargs="*", default=[], help="day prefixes (2026/10/08 ...) to download from the bucket in RECORDER_S3_*")
    p.add_argument("--cache", default="/tmp/recorder-cache")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--arms", default="", help="comma-separated arms (default: all)")
    p.add_argument("--symbols", default="", help="comma-separated internal symbols, e.g. BTC-USD,ETH-USD (default: all with trades)")
    p.add_argument("--trades-json", default=None, help="read trades from this JSON file instead of the database")
    p.add_argument("--since", default=None, help="only trades entered at/after this UTC date (YYYY-MM-DD[THH:MM])")
    p.add_argument("--notional", type=float, default=1000.0)
    p.add_argument("--latency-ms", type=float, default=100.0)
    p.add_argument("--variants", default=DEFAULT_VARIANTS, help="queue_factor:cancel_credit pairs, comma separated")
    p.add_argument("--fees", default=DEFAULT_FEES, help="maker:taker per leg, comma separated")
    p.add_argument("--entry-s", type=float, default=600.0)
    p.add_argument("--exit-s", type=float, default=600.0)
    args = p.parse_args(argv)

    paths = list(args.data)
    if args.fetch:
        from .fetch import download
        for day in args.fetch:
            paths += [str(x) for x in download(day, args.cache)]
    files = find_recordings(paths)
    if not files:
        raise SystemExit("error: no recorder files found; pass --data or --fetch")
    cfg = load_config(Path(args.config))
    arms = [a for a in args.arms.split(",") if a]
    since = None
    if args.since:
        from datetime import datetime, timezone
        since = int(datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp())
    if args.trades_json:
        rows = json.loads(Path(args.trades_json).read_text())
        if arms:
            rows = [r for r in rows if r["arm"] in arms]
        if since:
            rows = [r for r in rows if int(r["entry_ts"]) >= since]
    else:
        rows = load_rows_db(arms or None, since)
    only = {s for s in args.symbols.split(",") if s}
    if only:
        rows = [r for r in rows if r["symbol"] in only]
    specs, owners, notes = build_specs(rows, cfg)
    for n in notes:
        print(n, file=sys.stderr)
    if not specs:
        raise SystemExit("error: no trades to replay")
    variants = parse_pairs(args.variants)
    fees_list = [(f"fees {part}", Fees(*(float(x) for x in part.split(":")))) for part in args.fees.split(",")]
    symbols = sorted({s.symbol for s in specs})
    cfgs = {cfg_i["native"]: SimConfig(tick=cfg_i["tick"], price_decimals=cfg_i["decimals"], latency_ns=int(args.latency_ms * 1e6))
            for cfg_i in cfg.values() if cfg_i["native"] in symbols}
    rep = MultiReplayer(files, symbols)
    t0 = time.time()
    outcomes, stats = run_lifecycles(specs, rep, cfgs, variants, SCENARIOS, args.notional, args.entry_s, args.exit_s)
    print(f"{len(files)} file(s), {len(symbols)} symbol(s), {len(rows)} trade rows; replayed in {time.time() - t0:.0f}s", file=sys.stderr)
    print(format_report(outcomes, owners, fees_list, variants, stats, arms or None))


if __name__ == "__main__":
    main()
