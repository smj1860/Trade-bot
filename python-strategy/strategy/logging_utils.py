"""
Structured JSON-lines logging. Every meaningful event in a strategy run —
a signal computed, a decision made, an order submitted, an order's
result, a portfolio snapshot — gets one JSON object per line, appended to
`logging.log_path` from strategy_config.toml. This is the "log
visualization" piece from the project's original scope: a live dashboard
is unwarranted for a paper-trading baseline strategy with no track record
yet, but a structured, replayable log is exactly the raw material a
plotting script (scripts/plot_log.py) needs, and it's easy to swap for a
real-time dashboard later without changing anything that writes to it.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any


class _DecimalSafeEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


class JsonlEventLogger:
    """Writes one JSON object per line to `path`, and also mirrors a short
    human-readable line to stdout via the standard `logging` module so a
    person watching the process doesn't have to tail raw JSON."""

    def __init__(self, path: str, level: str = "INFO") -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", buffering=1)  # line-buffered

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s",
            stream=sys.stdout,
        )
        self._console = logging.getLogger("strategy")

    def log(self, event_type: str, /, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event_type, **fields}
        self._file.write(json.dumps(record, cls=_DecimalSafeEncoder) + "\n")
        self._console.info("%s %s", event_type, _short_repr(fields))

    def close(self) -> None:
        self._file.close()


def _short_repr(fields: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())
