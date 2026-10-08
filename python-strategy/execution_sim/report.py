"""Pairwise comparison of execution policies.

Every policy runs on identical episodes (same start time, side and size), so
the useful number is the per-episode cost *difference* against the baseline
policy, with a bootstrap interval. Costs are in basis points of notional
(100 bps = 1%), measured against the mid at arrival, fees included. Episodes
where any policy is missing, incomplete or aborted by a recording gap are
dropped from all policies.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .engine import EpisodeResult, Fees, episode_cost

BPS = 1e4


def bootstrap_mean(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi), 95% percentile interval. Zeros for no data."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(n_boot * 0.025)], means[min(n_boot - 1, int(n_boot * 0.975))]


def paired(results: list[EpisodeResult]) -> tuple[dict[int, dict[str, EpisodeResult]], int, int]:
    """spec_id -> {policy: result} for specs every policy completed; also (kept, dropped)."""
    by_spec: dict[int, dict[str, EpisodeResult]] = defaultdict(dict)
    for r in results:
        by_spec[r.spec_id][r.policy] = r
    names = {r.policy for r in results}
    keep = {sid: d for sid, d in by_spec.items() if set(d) == names and all(r.complete for r in d.values())}
    return keep, len(keep), len(by_spec) - len(keep)


def summarize(results: list[EpisodeResult], fees: Fees, baseline: str = "market", side: str | None = None) -> list[dict]:
    keep, _, _ = paired(results)
    specs = [d for d in keep.values() if side is None or next(iter(d.values())).side == side]
    names = sorted({p for d in specs for p in d}, key=lambda p: (p != baseline, p))
    rows = []
    for name in names:
        costs = [episode_cost(d[name], fees) for d in specs]
        total = [c[0] * BPS for c in costs]
        diff = [(episode_cost(d[name], fees)[0] - episode_cost(d[baseline], fees)[0]) * BPS for d in specs] if baseline in names else []
        maker_qty = sum(f.qty for d in specs for f in d[name].fills if f.maker)
        all_qty = sum(f.qty for d in specs for f in d[name].fills)
        row = {
            "policy": name,
            "n": len(specs),
            "cost": bootstrap_mean(total),
            "shortfall": sum(c[1] for c in costs) / max(len(costs), 1) * BPS,
            "fees": sum(c[2] for c in costs) / max(len(costs), 1) * BPS,
            "diff": bootstrap_mean(diff) if name != baseline else None,
            "maker_share": maker_qty / all_qty if all_qty else 0.0,
            "secs": sum((d[name].completed_ns - d[name].start_ns) / 1e9 for d in specs) / max(len(specs), 1),
            "posts": sum(d[name].posts for d in specs) / max(len(specs), 1),
            "markout": {},
        }
        for h in sorted({h for d in specs for h in d[name].markouts}):
            pts = [(q, m) for d in specs for q, m in d[name].markouts.get(h, [])]
            qty = sum(q for q, _ in pts)
            row["markout"][h] = (sum(q * m for q, m in pts) / qty * BPS if qty else None, len(pts))
        rows.append(row)
    return rows


def format_rows(rows: list[dict], title: str) -> str:
    lines = [title, f"{'policy':<22}{'n':>5} {'cost bps':>9} {'(95% CI)':>17} {'vs market bps':>14} {'(95% CI)':>17} {'maker%':>7} {'secs':>6} {'posts':>6}  maker markout bps"]
    for r in rows:
        m, lo, hi = r["cost"]
        if r["diff"] is None:
            d = f"{'-':>14} {'':>17}"
        else:
            dm, dlo, dhi = r["diff"]
            d = f"{dm:>+14.1f} {'[' + format(dlo, '+.1f') + ', ' + format(dhi, '+.1f') + ']':>17}"
        mk = "  ".join(f"{h}s:{(v if v is not None else float('nan')):+.1f}(n={n})" for h, (v, n) in r["markout"].items())
        lines.append(f"{r['policy']:<22}{r['n']:>5} {m:>9.1f} {'[' + format(lo, '.1f') + ', ' + format(hi, '.1f') + ']':>17} {d} {100 * r['maker_share']:>6.0f}% {r['secs']:>6.0f} {r['posts']:>6.1f}  {mk}")
    return "\n".join(lines)


def full_report(results: list[EpisodeResult], scenarios: list[tuple[str, Fees]], baseline: str = "market") -> str:
    _, kept, dropped = paired(results)
    out = [f"episodes: {kept} usable paired, {dropped} dropped (aborted by a gap, incomplete, or unpaired)"]
    for label, fees in scenarios:
        out.append("")
        out.append(format_rows(summarize(results, fees, baseline), f"== {label} (maker {fees.maker * 100:.2f}% / taker {fees.taker * 100:.2f}%) =="))
        for side in ("buy", "sell"):
            out.append("")
            out.append(format_rows(summarize(results, fees, baseline, side), f"-- {side} only --"))
    return "\n".join(out)
