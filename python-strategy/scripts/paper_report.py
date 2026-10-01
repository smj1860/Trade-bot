"""Summarises forward paper-trading results per arm, next to the model's
walk-forward expectation.

    python scripts/paper_report.py [--arm all] [--borrow-per-hour 0.0]

Shorts are scored without borrow/funding in the stored net_return;
--borrow-per-hour (a fraction of notional per hour held, e.g. 0.00005)
subtracts it from short trades here, using the recorded holding_bars.
Writes to $GITHUB_STEP_SUMMARY too when set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import connect  # noqa: E402
from strategy.paper import mean_and_ci  # noqa: E402


def adjusted_net(direction: int, net: float, holding_bars: int, interval_minutes: int, borrow_per_hour: float) -> float:
    if direction == -1 and borrow_per_hour:
        return net - borrow_per_hour * holding_bars * interval_minutes / 60.0
    return net


def format_group(label: str, rets: list[float]) -> str:
    m, hw, n = mean_and_ci(rets)
    wins = sum(1 for r in rets if r > 0)
    return f"  {label:<14} n={n:<5} win%={100 * wins / n if n else 0:5.1f}  mean net={m * 100:+.3f}% (±{hw * 100:.3f}%)"


def report_arm(conn, arm: str, borrow_per_hour: float) -> str:
    with conn.cursor() as cur:
        cur.execute("select meta from paper_models where arm = %s", (arm,))
        row = cur.fetchone()
        meta = row[0] if row else {}
        cur.execute(
            "select direction, exit_reason, net_return, holding_bars, ambiguous, proba_up, entry_ts "
            "from paper_trades where arm = %s and status = 'closed'",
            (arm,),
        )
        closed = cur.fetchall()
        cur.execute("select count(*), min(entry_ts), max(entry_ts) from paper_trades where arm = %s", (arm,))
        total, first, last = cur.fetchone()
        cur.execute("select count(*) from paper_trades where arm = %s and status = 'open'", (arm,))
        n_open = cur.fetchone()[0]
        cur.execute("select count(*), coalesce(sum(n_opened), 0) from paper_runs where arm = %s", (arm,))
        runs, _ = cur.fetchone()

    interval = int(meta.get("interval_minutes", 60))
    rets = [adjusted_net(r[0], float(r[2]), int(r[3]), interval, borrow_per_hour) for r in closed]
    lines = [f"== arm {arm} ==", f"  trades: {total} total, {len(closed)} closed, {n_open} open; {runs} hourly run(s); first={first} last={last}"]
    bt = meta.get("backtest", {})
    if bt:
        lines.append(f"  walk-forward expectation (trades with a resolvable touch only): mean net {bt['mean_net_pnl'] * 100:+.3f}% over {bt['trades']} trades, accuracy {bt['accuracy']:.3f}")
    if rets:
        lines.append(format_group("all", rets))
        for name, d in (("long", 1), ("short", -1)):
            lines.append(format_group(name, [x for x, r in zip(rets, closed) if r[0] == d]))
        for reason in ("target", "stop", "timeout", "ambiguous_stop"):
            sub = [x for x, r in zip(rets, closed) if r[1] == reason]
            if sub:
                lines.append(format_group(reason, sub))
        resolvable = [x for x, r in zip(rets, closed) if r[1] in ("target", "stop")]
        lines.append(format_group("target/stop only", resolvable) + "   <- like-for-like with the backtest")
        confident = [x for x, r in zip(rets, closed) if abs(float(r[5]) - 0.5) >= 0.1]
        lines.append(format_group("|p-0.5|>=0.1", confident))
        if borrow_per_hour:
            lines.append(f"  (short trades charged borrow {borrow_per_hour:.6f}/hour held)")
    else:
        lines.append("  no closed trades yet")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="all")
    p.add_argument("--borrow-per-hour", type=float, default=0.0)
    args = p.parse_args()
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select arm from paper_models order by arm")
            arms = [r[0] for r in cur.fetchall()] if args.arm == "all" else [args.arm]
        text = "\n\n".join(report_arm(conn, a, args.borrow_per_hour) for a in arms) or "no arms registered"
    finally:
        conn.close()
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
