"""Stores a trained model (and its sidecar meta JSON, written by
train_model.py next to the .joblib) in the paper_models table under an arm
name, replacing any model already registered for that arm. paper_trade.py
loads models from there, so the hourly job needs no artifact download.

    python scripts/register_paper_model.py --arm ext-fib-m4 --model models/pooled_gboost.joblib

Replacing an arm's model mid-experiment breaks the comparability of its
trade history -- register a new arm name instead unless that is intended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import connect  # noqa: E402

REQUIRED_META = ("feature_order", "windows", "extended", "fib", "interval_minutes", "horizon",
                 "barrier_by_symbol", "round_trip_cost", "symbols", "barrier_mode")


def validate_meta(meta: dict) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta JSON is missing keys: {missing}")
    if meta["barrier_mode"] != "fixed":
        raise ValueError("paper trading supports only fixed barriers (--barrier-mode fixed)")
    if meta.get("label_scheme") != "triple-barrier":
        raise ValueError("paper trading supports only --label-scheme triple-barrier models")
    if any(v is None for v in meta["barrier_by_symbol"].values()):
        raise ValueError("barrier_by_symbol has a symbol without a fixed barrier")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--model", required=True, help="Path to the .joblib; the meta is read from the same path with .json.")
    p.add_argument("--replace", action="store_true", help="Allow overwriting an arm that already has a model.")
    args = p.parse_args()

    model_path = Path(args.model)
    meta = json.loads(model_path.with_suffix(".json").read_text())
    validate_meta(meta)
    blob = model_path.read_bytes()

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select 1 from paper_models where arm = %s", (args.arm,))
            if cur.fetchone() and not args.replace:
                print(f"error: arm {args.arm!r} already has a model; pass --replace to overwrite.", file=sys.stderr)
                sys.exit(1)
            cur.execute(
                "insert into paper_models (arm, model_bytes, meta) values (%s, %s, %s) "
                "on conflict (arm) do update set model_bytes = excluded.model_bytes, meta = excluded.meta, created_at = now()",
                (args.arm, blob, json.dumps(meta)),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"registered arm {args.arm!r}: {len(blob)} bytes, {len(meta['feature_order'])} features, "
          f"{len(meta['symbols'])} symbols, horizon {meta['horizon']}")


if __name__ == "__main__":
    main()
