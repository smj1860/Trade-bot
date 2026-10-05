"""Hourly-rotated gzip files of raw feed lines.

Line format: `<recv_ns>\t<json>\n`. `recv_ns` is our wall-clock receive time in
nanoseconds since the epoch (UTC); `<json>` is the message exactly as Kraken
sent it, or an `{"_event": ...}` object written by the recorder itself
(connect, disconnect, checksum_mismatch, ...). Replaying the file in order
reproduces exactly what the feed delivered, including its gaps.

The gzip stream is flushed every few seconds (Z_SYNC_FLUSH), so a crash loses
seconds of data rather than the whole hour.
"""

from __future__ import annotations

import gzip
import time
from datetime import datetime, timezone
from pathlib import Path


class RotatingWriter:
    def __init__(self, root: Path, run_id: str, flush_seconds: float = 5.0, compresslevel: int = 4):
        self.root = Path(root)
        self.run_id = run_id
        self.flush_seconds = flush_seconds
        self.compresslevel = compresslevel
        self._hour_key: str | None = None
        self._path: Path | None = None
        self._file: gzip.GzipFile | None = None
        self._last_flush = time.monotonic()
        self.closed_paths: list[Path] = []
        self.bytes_in = 0  # uncompressed bytes written

    def _path_for(self, recv_ns: int) -> tuple[str, Path]:
        dt = datetime.fromtimestamp(recv_ns / 1e9, tz=timezone.utc)
        key = dt.strftime("%Y%m%d%H")
        return key, self.root / dt.strftime("%Y/%m/%d") / f"{dt.strftime('%H')}-{self.run_id}.tsv.gz"

    def write(self, recv_ns: int, payload: str) -> None:
        key, path = self._path_for(recv_ns)
        if key != self._hour_key:
            self._rotate(key, path)
        line = f"{recv_ns}\t{payload}\n".encode()
        self._file.write(line)
        self.bytes_in += len(line)
        now = time.monotonic()
        if now - self._last_flush >= self.flush_seconds:
            self._file.flush()
            self._last_flush = now

    def _rotate(self, key: str, path: Path) -> None:
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.GzipFile(path, "ab", compresslevel=self.compresslevel)
        self._hour_key, self._path = key, path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self.closed_paths.append(self._path)
            self._file = None
            self._hour_key = None

    @property
    def current_path(self) -> Path | None:
        return self._path if self._file is not None else None

    def pop_closed(self) -> list[Path]:
        out, self.closed_paths = self.closed_paths, []
        return out
