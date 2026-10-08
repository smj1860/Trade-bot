"""Download recorder files from an S3-compatible bucket (DigitalOcean Spaces).

Uses the same environment as the recorder's uploader: RECORDER_S3_BUCKET,
RECORDER_S3_ENDPOINT, RECORDER_S3_PREFIX (default kraken-l2), RECORDER_S3_REGION
and the usual AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. Files already present
locally with the same size are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path


def make_client():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ.get("RECORDER_S3_ENDPOINT") or None,
                        region_name=os.environ.get("RECORDER_S3_REGION") or None)


def download(day_prefix: str, dest: str | Path, client=None, bucket: str | None = None, prefix: str | None = None) -> list[Path]:
    """``day_prefix`` like "2026/10/07" (or "2026/10"). Returns local paths."""
    client = client or make_client()
    bucket = bucket or os.environ["RECORDER_S3_BUCKET"]
    prefix = os.environ.get("RECORDER_S3_PREFIX", "kraken-l2") if prefix is None else prefix
    root = f"{prefix}/" if prefix else ""
    dest = Path(dest)
    out: list[Path] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{root}{day_prefix}"}
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith((".tsv.gz", ".tsv.xz")):
                continue
            local = dest / key[len(root):]
            if not (local.exists() and local.stat().st_size == obj["Size"]):
                local.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(local))
            out.append(local)
        if not page.get("IsTruncated"):
            return sorted(out)
        token = page.get("NextContinuationToken")
