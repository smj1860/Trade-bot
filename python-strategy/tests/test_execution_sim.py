import gzip
import json
import lzma
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution_sim.engine import (  # noqa: E402
    NS, Cross, Episode, Fees, Post, SimConfig, episode_cost, simulate,
)
from execution_sim.fetch import download  # noqa: E402
from execution_sim.policies import MarketNow, PostAndChase, PostThenCross, default_policies  # noqa: E402
from execution_sim.replay import ASK, BID, BookEvent, GapEvent, Replayer, SymbolBook, TradeEvent, find_recordings  # noqa: E402
from execution_sim.report import full_report, paired, summarize  # noqa: E402

SYM = "BTC/USD"
FEES = Fees(0.004, 0.008)


def lv(levels):
    return [{"price": p, "qty": q} for p, q in levels]


def book_msg(bids, asks, typ="update", sym=SYM):
    return {"channel": "book", "type": typ, "data": [{"symbol": sym, "bids": lv(bids), "asks": lv(asks), "checksum": 1, "timestamp": "x"}]}


def trade_msg(side, price, qty, sym=SYM):
    return {"channel": "trade", "type": "update", "data": [{"symbol": sym, "side": side, "price": price, "qty": qty, "ord_type": "limit"}]}


def write_rec(path, items, opener=gzip.open):
    """items: (seconds, dict) -> recorder-format file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "wb") as f:
        for t, obj in items:
            f.write(f"{int(t * NS)}\t{json.dumps(obj)}\n".encode())


SNAP = book_msg([(100.0, 5.0), (99.9, 5.0), (99.8, 5.0)], [(100.2, 5.0), (100.3, 5.0), (100.4, 5.0)], "snapshot")
CFG = SimConfig(tick=0.1, price_decimals=1)


def fresh_book():
    b = SymbolBook(25)
    b.apply([(100.0, 5.0), (99.9, 5.0), (99.8, 5.0)], [(100.2, 5.0), (100.3, 5.0), (100.4, 5.0)], True)
    return b


class Hold:
    name = "hold"

    def act(self, s):
        return None


def episode(side="buy", cfg=CFG, notional=1000.0, book=None):
    return Episode(0, side, notional, 0, 600 * NS, Hold(), cfg, book or fresh_book())


# ---- replay ----------------------------------------------------------------------------

def test_replay_rebuilds_book_trims_depth_and_flags_gaps(tmp_path):
    f = tmp_path / "2026/10/07/01-run.tsv.gz"
    write_rec(f, [
        (0, {"_event": "connect", "depth": 2}),
        (1, book_msg([(100.0, 1), (99.9, 1), (99.8, 1)], [(100.2, 1), (100.3, 1)], "snapshot")),
        (2, book_msg([(100.0, 0)], [(100.1, 2)])),
        (2.5, book_msg([(1.0, 1)], [(2.0, 1)], sym="ETH/USD")),  # other symbol: ignored
        (3, trade_msg("sell", 99.9, 0.5)),
        (4, {"_event": "disconnect", "reason": "x"}),
        (5, book_msg([(50.0, 1)], [(51.0, 1)])),  # update while invalid: ignored
        (6, {"_event": "connect", "depth": 2}),
        (7, book_msg([(100.0, 3)], [(100.2, 3)], "snapshot")),
    ])
    rep = Replayer([tmp_path], SYM)
    seen = []
    for ev in rep.events():
        seen.append(type(ev).__name__)
        if isinstance(ev, BookEvent) and ev.t_ns == 2 * NS:
            assert rep.book.levels(BID) == [(99.9, 1)] and rep.book.levels(ASK) == [(100.1, 2), (100.2, 1)]  # depth 2: 100.3 trimmed
            assert ev.changes == [(BID, 100.0, 1, 0), (ASK, 100.1, 0.0, 2)]
    assert seen == ["BookEvent", "BookEvent", "TradeEvent", "GapEvent", "BookEvent"]
    assert rep.book.valid and rep.book.levels(BID) == [(100.0, 3)]


def test_replay_reads_gz_and_xz_in_time_order_and_checksum_mismatch_is_a_gap(tmp_path):
    write_rec(tmp_path / "2026/10/07/02-20261007T020000Z-1.tsv.xz", [(20, SNAP), (21, {"_event": "checksum_mismatch", "symbol": SYM})], lzma.open)
    write_rec(tmp_path / "2026/10/07/01-20261007T010000Z-1.tsv.gz", [(10, SNAP)])
    files = find_recordings([tmp_path])
    assert [f.name[:2] for f in files] == ["01", "02"]
    evs = list(Replayer([tmp_path], SYM).events())
    assert [type(e).__name__ for e in evs] == ["BookEvent", "BookEvent", "GapEvent"]
    assert evs[-1].reason == "checksum_mismatch"


def test_replay_ignores_truncated_last_line(tmp_path):
    f = tmp_path / "2026/10/07/01-r.tsv.gz"
    write_rec(f, [(1, SNAP)])
    with gzip.open(f, "ab") as g:
        g.write(b'20000000000\t{"channel":"book","type":"upd')
    assert len(list(Replayer([tmp_path], SYM).events())) == 1


# ---- queue and fills ---------------------------------------------------------------------

def test_join_touch_queues_behind_displayed_size_and_fills_after_it_trades():
    ep = episode()
    ep._post(100.0, 0, ep_book := fresh_book())
    o = ep.order
    assert o.queue_ahead == 5.0 and ep.posts == 1
    ep.on_trade(TradeEvent(1, False, 100.0, 3.0))
    assert ep.fills == [] and o.queue_ahead == 2.0
    ep.on_trade(TradeEvent(2, False, 100.0, 4.0))  # 2 clear the queue, 2 fill us
    assert len(ep.fills) == 1 and ep.fills[0].qty == pytest.approx(2.0) and ep.fills[0].maker and ep.fills[0].price == 100.0
    assert ep.remaining == pytest.approx(ep.qty_total - 2.0)


def test_buyer_aggressor_or_better_price_does_not_fill_a_bid():
    ep = episode()
    ep._post(100.0, 0, fresh_book())
    ep.on_trade(TradeEvent(1, True, 100.0, 50.0))  # a buyer lifting asks cannot fill our bid
    ep.on_trade(TradeEvent(2, False, 100.1, 50.0))  # sell above our price: not our level
    assert ep.fills == []


def test_trade_through_our_price_fills_everything_left():
    ep = episode()
    ep._post(100.0, 0, fresh_book())
    ep.on_trade(TradeEvent(1, False, 99.9, 0.1))
    assert ep.done and ep.fills[0].qty == pytest.approx(ep.qty_total) and ep.completed_ns == 1


def test_sell_side_mirrors():
    ep = episode("sell")
    ep._post(100.2, 0, fresh_book())
    assert ep.order.queue_ahead == 5.0
    ep.on_trade(TradeEvent(1, True, 100.3, 0.1))  # buyer paid 100.3: our ask at 100.2 must have been taken
    assert ep.done and ep.fills[0].price == 100.2


def test_inside_the_spread_has_no_queue_and_crossing_is_rejected():
    ep = episode()
    book = fresh_book()
    ep._post(100.1, 0, book)
    assert ep.order.queue_ahead == 0.0
    ep2 = episode()
    ep2._post(100.2, 0, book)  # equals best ask: would take
    assert ep2.order is None and ep2.rejected == 1 and ep2.last_rejected


def test_reposting_same_price_keeps_queue_position():
    ep = episode()
    book = fresh_book()
    ep._post(100.0, 0, book)
    ep.on_trade(TradeEvent(1, False, 100.0, 4.0))
    ep._post(100.0, 2, book)
    assert ep.order.queue_ahead == 1.0 and ep.posts == 1


def test_queue_cannot_exceed_displayed_size_and_cancel_credit_is_opt_in():
    book = fresh_book()
    ep = episode(book=book)
    ep._post(100.0, 0, book)
    ev = BookEvent(1, book.apply([(100.0, 3.0)], [], False), False)  # 2 disappears, no trade explains it
    ep.on_book(ev, book)
    assert ep.order.queue_ahead == 3.0  # capped at what is displayed, no credit by default
    book2 = fresh_book()
    ep2 = episode(cfg=SimConfig(tick=0.1, cancel_credit=1.0), book=book2)
    ep2._post(100.0, 0, book2)
    ep2.on_trade(TradeEvent(1, False, 100.0, 1.0))  # explains 1 of the coming 2 drop
    ev2 = BookEvent(2, book2.apply([(100.0, 2.0)], [], False), False)  # displayed 5 -> 2: 3 gone, 1 traded, 2 unexplained
    ep2.on_book(ev2, book2)
    assert ep2.order.queue_ahead == pytest.approx(2.0)  # 5-1(trade)=4, -2 credited = 2, cap 2


# ---- crossing and cost -----------------------------------------------------------------------

def test_cross_walks_the_book_and_costs_half_spread_plus_taker_fee():
    book = fresh_book()
    ep = episode(notional=100.1 * 7, book=book)  # 7 units: 5 @100.2 then 2 @100.3
    ep._cross(10, book)
    f = ep.fills[0]
    assert f.price == pytest.approx((5 * 100.2 + 2 * 100.3) / 7) and not f.maker and ep.done
    res = ep.result()
    total, short, fee = episode_cost(res, FEES)
    assert short == pytest.approx((f.price - 100.1) / 100.1) and fee == pytest.approx(0.008 * f.price / 100.1)
    assert total == pytest.approx(short + fee)


def test_cross_beyond_recorded_depth_is_flagged_and_penalised():
    book = fresh_book()
    ep = episode(notional=100.1 * 40, book=book)  # book only shows 15 units
    ep._cross(10, book)
    assert ep.exhausted and ep.fills[0].qty == pytest.approx(40.0, rel=1e-6) and ep.fills[0].price > 100.3


def test_sell_cost_sign_and_passive_fill_beats_mid():
    ep = episode("sell")
    ep._post(100.2, 0, fresh_book())
    ep.on_trade(TradeEvent(1, True, 100.3, 1.0))
    total, short, fee = episode_cost(ep.result(), FEES)
    assert short == pytest.approx(-(100.2 - 100.1) / 100.1)  # sold above mid: negative cost
    assert fee == pytest.approx(0.004 * 100.2 / 100.1)


# ---- end to end -------------------------------------------------------------------------------

def scripted(extra, end=700):
    items = [(0, {"_event": "connect", "depth": 25}), (0.1, SNAP)]
    t = 1.0
    while t < end:
        items.append((t, book_msg([(99.8, 5.0 + (int(t) % 2))], [])))  # keeps time moving, far from the touch
        t += 1.0
    return sorted(items + extra, key=lambda x: x[0])


def run(tmp_path, extra, policies, cfg=CFG, end=700, deadline=60, every=10_000):
    write_rec(tmp_path / "2026/10/07/01-r.tsv.gz", scripted(extra, end))
    rep = Replayer([tmp_path], SYM)
    return simulate(rep.events(), rep.book, policies, cfg, 400.0, every * NS, deadline * NS)


def test_market_pays_half_spread_and_a_passive_fill_saves_it(tmp_path):
    # one spec at t=0.1; a seller hits the bid at 100.0 (queue 5 -> 5.5 clears) at t=8
    res = run(tmp_path, [(8, trade_msg("sell", 100.0, 20.0))], {"market": MarketNow, "post": lambda: PostThenCross(0, 30)})
    rows = {r["policy"]: r for r in summarize(res, FEES)}
    half = (100.2 - 100.1) / 100.1 * 1e4
    buy_cost = half + 0.008 * 1e4 * 100.2 / 100.1  # half spread + taker fee on the ask
    sell_cost = half + 0.008 * 1e4 * 100.0 / 100.1  # half spread + taker fee on the bid
    assert rows["market"]["cost"][0] == pytest.approx((buy_cost + sell_cost) / 2, rel=1e-3)
    assert rows["post"]["maker_share"] == pytest.approx(0.5)  # the buy rested and filled, the sell crossed
    post_buy = [r for r in res if r.policy == "post" and r.side == "buy"][0]
    assert post_buy.fills[0].maker and post_buy.fills[0].price == 100.0
    assert post_buy.completed_ns == pytest.approx(8 * NS, abs=NS)
    sell_cost = episode_cost([r for r in res if r.policy == "post" and r.side == "sell"][0], FEES)[0]
    mkt_sell = episode_cost([r for r in res if r.policy == "market" and r.side == "sell"][0], FEES)[0]
    assert sell_cost == pytest.approx(mkt_sell, rel=0.01)  # unfilled passive sell ends up crossing at the same prices
    buy_post_cost = episode_cost(post_buy, FEES)[0]
    assert buy_post_cost == pytest.approx((100.0 - 100.1) / 100.1 + 0.004 * 100.0 / 100.1)  # bought below mid at the maker fee


def test_unfilled_passive_order_crosses_after_wait_and_has_extra_adverse_move(tmp_path):
    # price runs away upward before we cross: after 30s the ask has moved to 101.0
    moves = [(20, book_msg([(100.9, 5.0)], [(100.2, 0.0), (100.3, 0.0), (101.0, 5.0)]))]
    res = run(tmp_path, moves, {"market": MarketNow, "post": lambda: PostThenCross(0, 30)})
    buy_post = [r for r in res if r.policy == "post" and r.side == "buy"][0]
    buy_mkt = [r for r in res if r.policy == "market" and r.side == "buy"][0]
    assert not buy_post.fills[-1].maker and buy_post.fills[-1].price > buy_mkt.fills[0].price
    assert episode_cost(buy_post, FEES)[0] > episode_cost(buy_mkt, FEES)[0]


def test_deadline_forces_completion(tmp_path):
    res = run(tmp_path, [], {"idle": lambda: Hold()}, deadline=30)
    assert all(r.complete and not r.fills[0].maker for r in res)
    assert all((r.completed_ns - r.start_ns) / NS == pytest.approx(30.1, abs=2.5) for r in res)  # events arrive once a second here


def test_gap_aborts_in_flight_episodes_and_report_drops_them(tmp_path):
    res = run(tmp_path, [(5, {"_event": "disconnect"}), (6, {"_event": "connect", "depth": 25}), (6.5, SNAP)], {"market": MarketNow, "post": lambda: PostThenCross(0, 30)}, every=300)
    keep, kept, dropped = paired(res)
    assert kept >= 1 and all(d["post"].complete for d in keep.values())
    # the passive episode started at ~0.1 and was still resting at the gap
    assert any(r.aborted and r.policy == "post" for r in res)
    assert dropped >= 1


def test_markout_measures_price_after_a_maker_fill(tmp_path):
    items = [(8, trade_msg("sell", 100.0, 20.0)),
             (50, book_msg([(100.0, 0.0), (99.9, 0.0)], [(100.2, 0.0), (100.3, 0.0), (100.4, 0.0), (99.9, 5.0)]))]  # bid 99.8 / ask 99.9 from t=50
    res = run(tmp_path, items, {"market": MarketNow, "post": lambda: PostThenCross(0, 30)})
    row = {r["policy"]: r for r in summarize(res, FEES, side="buy")}["post"]
    m60, n = row["markout"][60]
    assert n == 1 and m60 == pytest.approx((99.85 - 100.0) / 100.0 * 1e4, rel=1e-3)  # bought at 100, mid fell to 99.85 by t=68


def test_chase_reposts_when_the_touch_moves_away(tmp_path):
    items = [(15, book_msg([(100.1, 3.0)], [])),  # someone improves the bid, putting us behind
             (40, trade_msg("sell", 100.1, 20.0))]  # then a seller hits the new touch: only a re-posted order can fill
    res = run(tmp_path, items, {"chase": lambda: PostAndChase(0, 5, 6, 0.2)}, deadline=100)
    buy = [r for r in res if r.side == "buy"][0]
    assert buy.posts == 2 and buy.fills[-1].maker and buy.fills[-1].price == 100.1
    still = run(tmp_path, items, {"post": lambda: PostThenCross(0, 90)}, deadline=100)
    assert [r for r in still if r.side == "buy"][0].fills[-1].maker is False  # the static order never filled at 100.0


def test_report_text_and_fee_scenarios_render(tmp_path):
    res = run(tmp_path, [(8, trade_msg("sell", 100.0, 20.0))], default_policies())
    text = full_report(res, [("t1", Fees(0.004, 0.008)), ("t3", Fees(0.0022, 0.0038))])
    assert "post-touch-wait30" in text and "vs market bps" in text and "-- buy only --" in text


# ---- fetch ---------------------------------------------------------------------------------------

class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.downloads = []

    def list_objects_v2(self, **kw):
        keys = [(k, s) for k, s in self.objects.items() if k.startswith(kw["Prefix"])]
        return {"Contents": [{"Key": k, "Size": s} for k, s in keys], "IsTruncated": False}

    def download_file(self, bucket, key, dest):
        Path(dest).write_bytes(b"x" * self.objects[key])
        self.downloads.append(key)


def test_fetch_downloads_matching_files_once(tmp_path):
    s3 = FakeS3({"kraken-l2/2026/10/07/08-a.tsv.xz": 4, "kraken-l2/2026/10/07/notes.txt": 2, "kraken-l2/2026/10/06/23-a.tsv.gz": 3})
    got = download("2026/10/07", tmp_path, client=s3, bucket="b", prefix="kraken-l2")
    assert [p.name for p in got] == ["08-a.tsv.xz"] and (tmp_path / "2026/10/07/08-a.tsv.xz").read_bytes() == b"xxxx"
    download("2026/10/07", tmp_path, client=s3, bucket="b", prefix="kraken-l2")
    assert s3.downloads == ["kraken-l2/2026/10/07/08-a.tsv.xz"]  # second call skipped the unchanged file
