"""Optional upload of finished hourly files to any S3-compatible bucket
(AWS S3, Cloudflare R2, Backblaze B2, Supabase Storage's S3 endpoint).

Configured by environment: RECORDER_S3_BUCKET (required to enable),
RECORDER_S3_ENDPOINT, RECORDER_S3_PREFIX, RECORDER_S3_REGION, plus the usual
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. A local file is deleted only after
its upload succeeds; failures leave it in place and it is retried on the next
sweep, so a bucket outage costs nothing but local disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("recorder.upload")


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
        for path in sorted(self.root.rglob("*.tsv.gz")):
            if path in skip:
                continue
            done += self.upload(path)
        return done
