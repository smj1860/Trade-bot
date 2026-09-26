"""
Reads a strategy JSONL log (strategy/logging_utils.py's output) and plots,
per symbol: mid-price over time with buy/sell markers where orders were
actually submitted. This is deliberately a post-hoc replay tool, not a
live dashboard — there's no live trading history yet worth building a
real-time view for, and a static plot after a run is enough to sanity
check what a strategy actually did.

Usage:
    python scripts/plot_log.py logs/strategy.jsonl [output.png]

If no output path is given, shows an interactive window (requires a
display; use the output-path form when running headless/in CI).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load_events(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def plot(events: list[dict], output_path: str | None) -> None:
    import matplotlib

    if output_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("event") == "signal" and "symbol" in e:
            by_symbol[e["symbol"]].append(e)

    if not by_symbol:
        print("No 'signal' events found in log — nothing to plot.")
        return

    fig, axes = plt.subplots(len(by_symbol), 1, figsize=(11, 4 * len(by_symbol)), squeeze=False)

    for ax, (symbol, points) in zip(axes[:, 0], sorted(by_symbol.items())):
        t0 = points[0]["ts"]
        times = [p["ts"] - t0 for p in points]
        mids = [float(p["mid_price"]) for p in points]
        ax.plot(times, mids, label="mid price", color="#4c72b0", linewidth=1)

        buys = [(p["ts"] - t0, float(p["mid_price"])) for p in points if p.get("order_side") == "BUY"]
        sells = [(p["ts"] - t0, float(p["mid_price"])) for p in points if p.get("order_side") == "SELL"]
        if buys:
            ax.scatter(*zip(*buys), color="green", marker="^", s=60, label="buy", zorder=3)
        if sells:
            ax.scatter(*zip(*sells), color="red", marker="v", s=60, label="sell", zorder=3)

        ax.set_title(symbol)
        ax.set_xlabel("seconds since run start")
        ax.set_ylabel("mid price")
        ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"wrote {output_path}")
    else:
        plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    events = load_events(log_path)
    plot(events, output_path)


if __name__ == "__main__":
    main()
