"""Optional upload of finished hourly files to any S3-compatible bucket
(AWS S3, Cloudflare R2, Backblaze B2, Supabase Storage's S3 endpoint).

Configured by environment: RECORDER_S3_BUCKET (required to enable),
RECORDER_S3_ENDPOINT, RECORDER_S3_PREFIX, RECORDER_S3_REGION, plus the usual
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. A local file is deleted only after
its upload succeeds; failures leave it in place and it is retried on the next
sweep, so a bucket outage costs nothing but local disk.
"""

from __future__ import annotations

import gzip
import logging
import lzma
import os
import shutil
import threading
from pathlib import Path

log = logging.getLogger("recorder.upload")


XZ_PRESET = 6  # ~95 MiB of encoder memory; preset 9 needs ~670 MiB, too much for a 1 GB server


def _compress(path: Path, tmp: Path) -> None:
    with gzip.open(path, "rb") as src, lzma.open(tmp, "wb", preset=XZ_PRESET) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


def _compress_low_priority(path: Path, tmp: Path) -> None:
    """Run the compression in its own short-lived thread at nice 19, so on a
    one-core server the live recorder always gets the CPU first. Linux nice is
    per thread, and an unprivileged thread cannot raise its priority back, so
    this must be a throwaway thread rather than a pool worker."""
    err: list[BaseException] = []

    def run() -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass  # not Linux, or not permitted: just run at normal priority
        try:
            _compress(path, tmp)
        except BaseException as e:  # re-raised in the caller
            err.append(e)

    t = threading.Thread(target=run, name="xz-recompress")
    t.start()
    t.join()
    if err:
        raise err[0]


def recompress_xz(path: Path) -> Path:
    """Rewrite a finished hourly .tsv.gz as .tsv.xz (about 35-40% smaller on
    live data) and delete the gzip. Streams in chunks, writes to a temp name and
    renames, so a crash never leaves a half-written .tsv.xz. The record format
    inside is unchanged."""
    out = path.with_name(path.name[: -len(".gz")] + ".xz")
    tmp = out.with_name(out.name + ".tmp")
    _compress_low_priority(path, tmp)
    tmp.replace(out)
    path.unlink()
    return out


class Uploader:
    def __init__(self, root: Path, bucket: str, endpoint: str | None, prefix: str, region: str | None):
        import boto3  # imported lazily: only needed when uploading

        self.root = Path(root)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint or None, region_name=region or None)

    @classmethod
    def from_env(cls, root: Path) -> "Uploader | None":
        bucket = os.environ.get("RECORDER_S3_BUCKET")
        if not bucket:
            return None
        return cls(
            root,
            bucket,
            os.environ.get("RECORDER_S3_ENDPOINT"),
            os.environ.get("RECORDER_S3_PREFIX", "kraken-l2"),
            os.environ.get("RECORDER_S3_REGION"),
        )

    def key_for(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def upload(self, path: Path) -> bool:
        try:
            if path.name.endswith(".tsv.gz"):
                path = recompress_xz(path)
            self.client.upload_file(str(path), self.bucket, self.key_for(path))
            path.unlink()
            return True
        except Exception as exc:  # noqa: BLE001 - keep the file, retry later
            log.warning("upload of %s failed: %s", path, exc)
            return False

    def sweep(self, skip: set[Path]) -> int:
        """Upload every finished file under root except those in `skip` (the
        file currently being written). Returns how many were uploaded."""
        done = 0
        for path in sorted([*self.root.rglob("*.tsv.gz"), *self.root.rglob("*.tsv.xz")]):
            if path in skip:
                continue
            done += self.upload(path)
        return done
