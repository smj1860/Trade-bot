"""Weekly challenger bookkeeping for forward paper trading.

    python scripts/challenger.py retire --keep 4
    python scripts/challenger.py report --champion ext-fib-m4

A challenger is an arm named ``ch-...`` (registered each week by
.github/workflows/weekly-challenger.yml). ``retire`` marks all but the newest
``--keep`` challengers as retired: they stop opening trades but still resolve
the ones they have open (paper_trade.py). Champions (any arm not named
``ch-...``) are never touched. Nothing here promotes a model to live money.

``report`` compares each challenger with the champion on the SAME
(symbol, entry bar) trades only, so market regime cancels out, and puts a
bootstrap confidence interval on the mean difference. With a few dozen pairs
that interval will straddle zero; treat it as "no difference shown yet", not
as a result either way.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHALLENGER_PREFIX = "ch-"


def plan_retirements(arms: list[tuple[str, object, bool]], keep: int) -> list[str]:
    """arms: (name, created_at, already_retired). Returns challengers to
    retire: every active ``ch-`` arm except the ``keep`` newest."""
    active = sorted(
        (a for a in arms if a[0].startswith(CHALLENGER_PREFIX) and not a[2]),
        key=lambda a: a[1],
        reverse=True,
    )
    return [name for name, _, _ in active[max(keep, 0):]]


def paired_diffs(challenger: dict[tuple, float], champion: dict[tuple, float]) -> list[float]:
    """challenger - champion net return for trades both arms took at the same
    (symbol, entry_ts)."""
    return [challenger[k] - champion[k] for k in sorted(challenger.keys() & champion.keys())]


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap of the mean. (0, 0, 0) for no data."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(n_boot * alpha / 2)]
    hi = means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return sum(values) / n, lo, hi


def _closed(conn, arm: str) -> dict[tuple, float]:
    with conn.cursor() as cur:
        cur.execute(
            "select symbol, extract(epoch from entry_ts)::bigint, net_return from paper_trades "
            "where arm = %s and status = 'closed'",
            (arm,),
        )
        return {(r[0], int(r[1])): float(r[2]) for r in cur.fetchall()}


def cmd_retire(conn, keep: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("select arm, created_at, coalesce((meta->>'retired')::boolean, false) from paper_models")
        arms = [(r[0], r[1], r[2]) for r in cur.fetchall()]
        names = plan_retirements(arms, keep)
        for name in names:
            cur.execute(
                "update paper_models set meta = meta || %s::jsonb where arm = %s",
                (json.dumps({"retired": True}), name),
            )
    conn.commit()
    print(f"retired {len(names)} challenger(s): {', '.join(names) or 'none'}; keeping the newest {keep}")
    return names


def format_report(champion: str, champ: dict[tuple, float], challengers: dict[str, dict[tuple, float]]) -> str:
    lines = [f"Challengers vs champion {champion} (closed trades only, same symbol + entry bar)"]
    cm = sum(champ.values()) / len(champ) if champ else 0.0
    lines.append(f"  champion: n={len(champ)} mean net={cm * 100:+.3f}%")
    if not challengers:
        lines.append("  no challengers registered yet")
    for arm, trades in sorted(challengers.items()):
        diffs = paired_diffs(trades, champ)
        own = sum(trades.values()) / len(trades) if trades else 0.0
        if diffs:
            m, lo, hi = bootstrap_ci(diffs)
            verdict = "no difference shown" if lo <= 0 <= hi else ("better" if m > 0 else "worse")
            lines.append(
                f"  {arm}: own n={len(trades)} mean net={own * 100:+.3f}% | paired n={len(diffs)} "
                f"diff={m * 100:+.3f}% 95% CI [{lo * 100:+.3f}%, {hi * 100:+.3f}%] -> {verdict}"
            )
        else:
            lines.append(f"  {arm}: own n={len(trades)} mean net={own * 100:+.3f}% | no paired closed trades yet")
    return "\n".join(lines)


def cmd_report(conn, champion: str) -> str:
    with conn.cursor() as cur:
        cur.execute("select arm from paper_models where arm like %s order by arm", (CHALLENGER_PREFIX + "%",))
        names = [r[0] for r in cur.fetchall()]
    text = format_report(champion, _closed(conn, champion), {n: _closed(conn, n) for n in names})
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")
    return text


def main() -> None:
    from scripts.train_model import connect

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retire")
    r.add_argument("--keep", type=int, default=4, help="Active challengers to keep (default 4).")
    rep = sub.add_parser("report")
    rep.add_argument("--champion", default="ext-fib-m4")
    args = p.parse_args()

    conn = connect()
    try:
        if args.cmd == "retire":
            cmd_retire(conn, args.keep)
        else:
            cmd_report(conn, args.champion)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
