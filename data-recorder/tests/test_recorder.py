import asyncio
import gzip
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recorder.book import LocalBook, compute_checksum, decimals_of, format_component  # noqa: E402
from recorder.config import SymbolInfo, load_symbols  # noqa: E402
from recorder.main import Recorder, Resync  # noqa: E402
from recorder.writer import RotatingWriter  # noqa: E402

D = Decimal

# Kraken's published worked example (also used by rust-core/src/checksum.rs).
ASKS = [("45285.2", "0.00100000"), ("45286.4", "1.54571953"), ("45286.6", "1.54571109"), ("45289.6", "1.54560911"),
        ("45290.2", "0.15890660"), ("45291.8", "1.54553491"), ("45294.7", "0.04454749"), ("45296.1", "0.35380000"),
        ("45297.5", "0.09945542"), ("45299.5", "0.18772827")]
BIDS = [("45283.5", "0.10000000"), ("45283.4", "1.54582015"), ("45282.1", "0.10000000"), ("45281.0", "0.10000000"),
        ("45280.3", "1.54592586"), ("45279.0", "0.07990000"), ("45277.6", "0.03310103"), ("45277.5", "0.30000000"),
        ("45277.3", "1.54602737"), ("45276.6", "0.15445238")]


def lv(rows):
    return [(D(p), D(q)) for p, q in rows]


def test_matches_krakens_worked_example():
    assert compute_checksum(lv(ASKS), lv(BIDS), 1, 8) == 3_310_070_434


def test_padding_makes_checksum_independent_of_stripped_zeros():
    a = compute_checksum([(D("0.0931"), D("1265.625"))], [], 7, 8)
    b = compute_checksum([(D("0.0931000"), D("1265.62500000"))], [], 7, 8)
    assert a == b
    assert format_component(D("0.0931"), 7) == "931000"
    assert format_component(D("0"), 1) == "0"


def test_decimals_of():
    assert decimals_of("0.00000001") == 8 and decimals_of("0.1") == 1 and decimals_of("1") == 0


def test_local_book_applies_updates_and_trims_to_depth():
    book = LocalBook(depth=2, price_decimals=1, qty_decimals=1)
    book.apply(lv([("10", "1"), ("9", "1"), ("8", "1")]), lv([("11", "1"), ("12", "1"), ("13", "1")]), snapshot=True)
    assert sorted(book.bids) == [D("9"), D("10")] and sorted(book.asks) == [D("11"), D("12")]
    book.apply([], lv([("11", "0")]), snapshot=False)  # qty 0 removes the level
    assert D("11") not in book.asks
    book.apply(lv([("10", "2")]), [], snapshot=False)
    assert book.bids[D("10")] == D("2")
    book.apply([], [], snapshot=True)
    assert not book.bids and not book.asks


def test_config_loads_repo_symbols():
    syms = load_symbols()
    assert "BTC/USD" in syms and len(syms) >= 10
    assert syms["BTC/USD"].price_decimals == 1 and syms["BTC/USD"].qty_decimals == 8
    assert list(load_symbols(only=["ETH-USD"])) == ["ETH/USD"]


def book_msg(sym, bids, asks, typ, checksum):
    fmt = lambda rows: [{"price": float(p), "qty": float(q)} for p, q in rows]
    return json.dumps({"channel": "book", "type": typ, "data": [{
        "symbol": sym, "bids": fmt(bids), "asks": fmt(asks), "checksum": checksum, "timestamp": "2026-10-05T00:00:00.000000Z"}]})


def good_checksum(bids, asks, pd=1, qd=8):
    return compute_checksum(sorted(lv(asks))[:10], sorted(lv(bids), reverse=True)[:10], pd, qd)


@pytest.fixture
def rec(tmp_path):
    syms = {"BTC/USD": SymbolInfo("BTC/USD", 1, 8)}
    return Recorder(syms, RotatingWriter(tmp_path, "test"), depth=10)


def test_valid_checksum_passes_and_mismatch_streak_forces_resync(rec):
    bids, asks = [("99.5", "1.5"), ("99.0", "0.001")], [("100.0", "2"), ("100.5", "0.25")]
    rec.handle(book_msg("BTC/USD", bids, asks, "snapshot", good_checksum(bids, asks)), 1)
    assert rec.stats["checksum_checks"] == 1 and rec.stats["checksum_mismatches"] == 0
    upd = book_msg("BTC/USD", [("99.5", "1.0")], [], "update", 12345)
    rec.handle(upd, 2)
    rec.handle(upd, 3)
    assert rec.stats["checksum_mismatches"] == 2
    with pytest.raises(Resync):
        rec.handle(upd, 4)


def test_good_message_resets_the_streak(rec):
    bids, asks = [("99.5", "1.5")], [("100.0", "2")]
    rec.handle(book_msg("BTC/USD", bids, asks, "snapshot", good_checksum(bids, asks)), 1)
    bad = book_msg("BTC/USD", [("99.5", "1.5")], [], "update", 1)
    rec.handle(bad, 2); rec.handle(bad, 3)
    ok = book_msg("BTC/USD", [("99.5", "1.5")], [], "update", good_checksum(bids, asks))
    rec.handle(ok, 4)
    rec.handle(bad, 5); rec.handle(bad, 6)  # streak restarted from zero, so still no Resync
    assert rec._bad_streak["BTC/USD"] == 2


def test_heartbeats_are_not_recorded_but_trades_and_acks_are(rec, tmp_path):
    rec.handle(json.dumps({"channel": "heartbeat"}), 1_700_000_000_000_000_000)
    rec.handle(json.dumps({"channel": "trade", "type": "update", "data": [{"symbol": "BTC/USD", "side": "buy", "qty": 0.1, "price": 100.0}]}), 1_700_000_000_000_000_001)
    rec.handle(json.dumps({"method": "subscribe", "success": True, "result": {"channel": "book"}}), 1_700_000_000_000_000_002)
    rec.writer.close()
    f = next(tmp_path.rglob("*.tsv.gz"))
    lines = gzip.open(f, "rt").read().splitlines()
    assert len(lines) == 2 and rec.stats["trade_msgs"] == 1
    assert lines[0].split("\t")[0] == "1700000000000000001"


def test_writer_rotates_by_utc_hour_and_survives_reopen(tmp_path):
    w = RotatingWriter(tmp_path, "r1")
    h = 3600 * 10**9
    base = 1_700_000_000 * 10**9 // h * h
    w.write(base + 1, "a"); w.write(base + 2, "b"); w.write(base + h + 1, "c")
    assert len(w.closed_paths) == 1 and w.current_path is not None
    w.close()
    names = sorted(p.name for p in tmp_path.rglob("*.tsv.gz"))
    assert len(names) == 2
    first = next(p for p in tmp_path.rglob("*.tsv.gz") if p == w.closed_paths[0])
    assert gzip.open(first, "rt").read().splitlines() == [f"{base + 1}\ta", f"{base + 2}\tb"]


def test_end_to_end_against_a_local_fake_feed_with_reconnect(tmp_path):
    bids, asks = [("99.5", "1.5"), ("99.0", "0.001")], [("100.0", "2"), ("100.5", "0.25")]
    snap = book_msg("BTC/USD", bids, asks, "snapshot", good_checksum(bids, asks))
    connections = []

    async def feed(ws):
        connections.append(1)
        subs = [json.loads(await ws.recv()) for _ in range(2)]
        assert {s["params"]["channel"] for s in subs} == {"book", "trade"}
        await ws.send(json.dumps({"method": "subscribe", "success": True, "result": {"channel": "book"}}))
        await ws.send(json.dumps({"channel": "heartbeat"}))
        await ws.send(snap)
        await ws.send(json.dumps({"channel": "trade", "type": "update", "data": [{"symbol": "BTC/USD", "side": "sell", "qty": 0.5, "price": 99.5}]}))
        await asyncio.sleep(0.2)
        await ws.close()  # force a reconnect

    async def go():
        async with websockets.serve(feed, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            r = Recorder({"BTC/USD": SymbolInfo("BTC/USD", 1, 8)}, RotatingWriter(tmp_path, "e2e"), depth=10, url=f"ws://127.0.0.1:{port}")
            await r.run(duration=4.5)
            return r

    r = asyncio.run(go())
    assert len(connections) >= 2 and r.stats["reconnects"] >= 1
    assert r.stats["checksum_mismatches"] == 0 and r.stats["checksum_checks"] >= 2
    text = "\n".join(gzip.open(p, "rt").read() for p in tmp_path.rglob("*.tsv.gz"))
    assert '"_event":"connect"' in text and '"_event":"disconnect"' in text and '"channel": "trade"' in text
    assert r.summary(tmp_path)["files"] >= 1


def test_recompress_to_xz_is_lossless_and_smaller(tmp_path):
    import lzma

    from recorder.uploader import recompress_xz

    w = RotatingWriter(tmp_path, "x")
    base = 1_700_000_000 * 10**9
    for i in range(3000):
        w.write(base + i, '{"channel":"book","type":"update","data":[{"symbol":"BTC/USD","bids":[{"price":100.0,"qty":1.5}],"asks":[],"checksum":123}]}')
    w.close()
    gz = w.closed_paths[0]
    original = gzip.open(gz, "rb").read()
    xz = recompress_xz(gz)
    assert not gz.exists() and xz.name.endswith(".tsv.xz") and not list(tmp_path.rglob("*.tmp"))
    assert lzma.open(xz, "rb").read() == original


def test_uploader_recompresses_then_uploads_and_deletes(tmp_path, monkeypatch):
    import lzma

    from recorder.uploader import Uploader

    class FakeClient:
        def __init__(self):
            self.sent = []

        def upload_file(self, filename, bucket, key):
            self.sent.append((key, lzma.open(filename, "rb").read()))

    up = Uploader.__new__(Uploader)
    up.root, up.bucket, up.prefix, up.client = tmp_path, "b", "kraken-l2", FakeClient()
    w = RotatingWriter(tmp_path, "u")
    w.write(1_700_000_000 * 10**9, "hello")
    w.close()
    assert up.sweep(skip=set()) == 1
    assert up.client.sent[0][0].startswith("kraken-l2/2023/") and up.client.sent[0][0].endswith(".tsv.xz")
    assert up.client.sent[0][1].endswith(b"hello\n")
    assert not list(tmp_path.rglob("*.tsv.*"))
