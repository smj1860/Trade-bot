# Kraken order-book recorder

Records Kraken spot level-2 order book and trade messages for every enabled symbol in
`config/config.example.toml`, exactly as the exchange sends them, into hourly gzip files.
It exists because Kraken offers no historical order book: the only way to get Kraken's own
book history is to record it as it happens. It never trades and shares no code or process with
`rust-core`, so a recorder crash cannot touch execution (and the reverse).

## File format

`<out>/YYYY/MM/DD/HH-<run id>.tsv.gz`, one line per message: `<recv_ns>\t<json>`.
`recv_ns` is our UTC receive time in nanoseconds; `<json>` is Kraken's raw v2 message (`book`
snapshot/update, `trade`, subscription acks/status) or a recorder event (`{"_event": "connect" |
"disconnect" | "checksum_mismatch", ...}`). Heartbeats are dropped. Replay a file in order to
reproduce the feed, gaps included. Every `snapshot` message resets a symbol's book; a
`disconnect` event marks a gap, and the next `snapshot` after the following `connect` re-syncs.

## Integrity

The recorder keeps a local copy of each book (top `--depth` levels) and checks it against the
CRC-32 Kraken sends with every book message (same algorithm as `rust-core/src/checksum.rs`).
Mismatches are written to the file as events; three in a row for one symbol force a reconnect
for fresh snapshots. Treat data between a `checksum_mismatch` and the next `snapshot` as suspect.

## Run

    pip install -r data-recorder/requirements.txt
    cd data-recorder && python -m recorder.main --out ./data            # forever, depth 25
    python -m recorder.main --out ./data --duration 120 --symbols BTC/USD,ETH/USD   # quick test

Depth 25 is the default (10, 25, 100, 500, 1000 are valid). The checksum only covers the top 10 either
way. Measured on live Kraken, all 14 symbols, 3-minute evening (UTC) windows, so treat as order of magnitude:

| depth | compressed | per hour | per day | per month |
|---|---|---|---|---|
| 10  | 2.5 MB / 3 min | ~50 MB  | ~1.2 GB | ~36 GB |
| 25  | 3.6 MB / 3 min | ~72 MB  | ~1.7 GB | ~52 GB |
| 100 | 5.4 MB / 3 min | ~107 MB | ~2.6 GB | ~77 GB |

About 1,000 messages per second across the 14 symbols. Checksum pass rate was 100% in all three runs
(81,800 / 123,968 / 186,330 checks, zero reconnects). Tardis captures depth 1000 for deeper research.

## Run it 24/7

It needs an always-on machine. Do not rely on GitHub Actions for this: scheduled runs skip hours.
A small VPS is enough (1 vCPU, 1 GB RAM; disk depends on how long you keep files locally).

    docker build -f data-recorder/Dockerfile -t kraken-recorder .
    docker run -d --name recorder --restart unless-stopped -v /srv/recorder:/data \
      --env-file recorder.env kraken-recorder

`recorder.env` (all optional) enables upload of finished hourly files to any S3-compatible
bucket (AWS S3, Cloudflare R2, Backblaze B2, Supabase Storage's S3 endpoint). A local file is deleted
only after it uploaded successfully:

    RECORDER_S3_BUCKET=my-bucket
    RECORDER_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
    RECORDER_S3_PREFIX=kraken-l2
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...

Keys belong in the env file on the server only, never in the repo.

## Test

    cd data-recorder && python -m pytest tests -q

`.github/workflows/recorder-smoke.yml` records a couple of minutes of live Kraken data on a GitHub
runner and prints the checksum pass rate and data volume.
