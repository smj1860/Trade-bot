"""Summarises recording gaps: how often the recorder disconnected or a book failed its checksum.

    python -m execution_sim.gaps --data /root/rec-cache

Gaps are what exclude trades from the lifecycle evaluation (any gap during a trade's
life drops it), so this shows how much of that is the recorder's feed rather than the method.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from .replay import find_recordings, open_recording


def scan(paths) -> dict:
    ev = Counter()
    reasons = Counter()
    mismatch_by_symbol = Counter()
    by_hour = Counter()
    downtimes = []
    last_disc = None
    first = last = None
    for path in find_recordings(paths):
        with open_recording(path) as f:
            for line in f:
                tab = line.find(b"\t")
                if tab < 0:
                    continue
                t = int(line[:tab])
                first = t if first is None else first
                last = t
                if b'"_event"' not in line:
                    continue
                try:
                    msg = json.loads(line[tab + 1:])
                except ValueError:
                    continue
                name = msg.get("_event")
                ev[name] += 1
                hour = datetime.fromtimestamp(t / 1e9, timezone.utc).strftime("%m-%d %H:00")
                if name == "disconnect":
                    by_hour[hour] += 1
                    reasons[str(msg.get("reason", ""))[:70]] += 1
                    last_disc = t
                elif name == "checksum_mismatch":
                    mismatch_by_symbol[msg.get("symbol")] += 1
                elif name == "connect" and last_disc is not None:
                    downtimes.append((t - last_disc) / 1e9)
                    last_disc = None
    return {"events": ev, "reasons": reasons, "mismatch": mismatch_by_symbol, "disc_by_hour": by_hour,
            "downtimes": downtimes, "span_h": ((last - first) / 3.6e12) if first is not None else 0.0}


def format_summary(s: dict) -> str:
    d = sorted(s["downtimes"])
    lines = [f"recorded span: {s['span_h']:.1f} h", "events: " + ", ".join(f"{k}={v}" for k, v in sorted(s["events"].items()))]
    if d:
        lines.append(f"disconnect -> reconnect: n={len(d)}, median {d[len(d) // 2]:.0f}s, max {d[-1]:.0f}s")
    lines.append("disconnect reasons: " + "; ".join(f"{r} x{n}" for r, n in s["reasons"].most_common(5)))
    lines.append("checksum mismatches by symbol: " + ", ".join(f"{k} {v}" for k, v in s["mismatch"].most_common(8)))
    lines.append("disconnects by hour (UTC): " + ", ".join(f"{h} {n}" for h, n in sorted(s["disc_by_hour"].items())[-24:]))
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True)
    print(format_summary(scan(p.parse_args(argv).data)))


if __name__ == "__main__":
    main()
