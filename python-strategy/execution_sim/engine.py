"""Execution episodes over a replayed book.

An *episode* is one parent order: buy or sell ``notional`` of a symbol within
``deadline`` seconds, starting at an arrival time. A policy decides, about
once a second, to post a limit order, cancel, or cross the spread; whatever is
unfilled at the deadline is crossed ("must complete"). Cost is measured
against the mid price at arrival (implementation shortfall) plus fees.

What the model assumes, and where it is optimistic or pessimistic (all of it
is counterfactual: our orders never changed the recorded book):

* Queue position. An order joining a price level starts behind
  ``queue_factor`` x the displayed size there (1.0 = everything displayed is
  ahead of us, the conservative default). It moves up only when trades print at
  its price, plus ``cancel_credit`` x the part of any displayed-size drop that
  trades do not explain (0.0 = conservative: assume cancellations came from
  behind us). It can never be further back than the size displayed.
* Fills. A trade printing through our price fills the whole remaining order. A
  trade at our price fills us once the queue ahead is used up. Nothing else
  fills us.
* Latency. Every action takes effect ``latency`` after the decision, against the
  book as it is then. Fills can happen meanwhile.
* Crossing walks the displayed levels (VWAP); size beyond the recorded depth
  fills at the last level worse by ``exhaust_penalty``.
* Our own size does not move the market and the book beyond the recorded depth
  is invisible. Posting is post-only: a price that would cross is rejected.
* A gap in the recording (disconnect, checksum mismatch) aborts the episode.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .replay import ASK, BID, BookEvent, GapEvent, SymbolBook, TradeEvent

NS = 1_000_000_000


# ---- policy interface -----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Post:
    """Rest a post-only limit at ``price`` for the remaining quantity (replaces any resting order)."""
    price: float


@dataclass(frozen=True, slots=True)
class Cancel:
    pass


@dataclass(frozen=True, slots=True)
class Cross:
    """Take the remaining quantity from the book now."""


Action = Post | Cancel | Cross


@dataclass(slots=True)
class OrderView:
    price: float
    remaining: float
    queue_ahead: float
    age_s: float


@dataclass(slots=True)
class State:
    now_ns: int
    elapsed_s: float
    left_s: float
    left_frac: float
    side: str  # "buy" or "sell"
    remaining_qty: float
    remaining_frac: float
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    mid: float
    spread: float
    tick: float
    arrival_mid: float
    order: OrderView | None
    pending: bool  # an action is in flight (latency)
    last_rejected: bool

    @property
    def touch(self) -> float:
        """Best price on our own side of the book."""
        return self.best_bid if self.side == "buy" else self.best_ask


# ---- config / results -----------------------------------------------------------

@dataclass(slots=True)
class SimConfig:
    tick: float
    price_decimals: int = 2
    latency_ns: int = 100_000_000
    queue_factor: float = 1.0
    cancel_credit: float = 0.0
    decision_interval_ns: int = NS
    exhaust_penalty: float = 0.0005
    markout_horizons_s: tuple[int, ...] = (10, 60)


@dataclass(slots=True)
class Fill:
    t_ns: int
    price: float
    qty: float
    maker: bool


@dataclass(slots=True)
class EpisodeResult:
    spec_id: int
    policy: str
    side: str
    start_ns: int
    notional: float
    qty_total: float
    arrival_mid: float
    fills: list[Fill]
    completed_ns: int | None
    aborted: bool = False
    posts: int = 0
    rejected: int = 0
    exhausted: bool = False
    markouts: dict[int, list[tuple[float, float]]] = field(default_factory=dict)  # horizon_s -> [(qty, signed fraction)]

    @property
    def complete(self) -> bool:
        return self.completed_ns is not None and not self.aborted


@dataclass(frozen=True)
class Fees:
    maker: float
    taker: float


def episode_cost(res: EpisodeResult, fees: Fees) -> tuple[float, float, float]:
    """(total, shortfall, fees) as fractions of notional at arrival."""
    sign = 1.0 if res.side == "buy" else -1.0
    shortfall = sum(sign * (f.price - res.arrival_mid) * f.qty for f in res.fills)
    fee = sum((fees.maker if f.maker else fees.taker) * f.price * f.qty for f in res.fills)
    base = res.qty_total * res.arrival_mid
    return (shortfall + fee) / base, shortfall / base, fee / base


# ---- one episode ------------------------------------------------------------------

class _Order:
    __slots__ = ("price", "remaining", "queue_ahead", "placed_ns", "trade_vol")

    def __init__(self, price: float, remaining: float, queue_ahead: float, placed_ns: int):
        self.price, self.remaining, self.queue_ahead, self.placed_ns = price, remaining, queue_ahead, placed_ns
        self.trade_vol = 0.0  # traded at our price since the displayed size last changed


class Episode:
    def __init__(self, spec_id: int, side: str, notional: float, start_ns: int, deadline_ns: int,
                 policy, cfg: SimConfig, book: SymbolBook):
        self.spec_id, self.side, self.notional = spec_id, side, notional
        self.start_ns, self.deadline_ns = start_ns, deadline_ns
        self.policy, self.cfg = policy, cfg
        self.own = BID if side == "buy" else ASK
        self.opp = ASK if side == "buy" else BID
        bb, ba = book.best(BID), book.best(ASK)
        self.arrival_mid = (bb[0] + ba[0]) / 2.0
        self.qty_total = notional / self.arrival_mid
        self.remaining = self.qty_total
        self.fills: list[Fill] = []
        self.order: _Order | None = None
        self.pending: list[tuple[int, Action]] = []
        self.next_decision_ns = start_ns
        self.forced = False
        self.done = False
        self.aborted = False
        self.completed_ns: int | None = None
        self.posts = self.rejected = 0
        self.last_rejected = False
        self.exhausted = False
        self.markouts: dict[int, list[tuple[float, float]]] = {h: [] for h in cfg.markout_horizons_s}

    # -- events ---------------------------------------------------------------
    def on_trade(self, ev: TradeEvent) -> None:
        o = self.order
        if o is None or self.done:
            return
        if self.side == "buy":
            hits = not ev.aggressor_buy  # a seller hitting bids
            through, level = ev.price < o.price, ev.price == o.price
        else:
            hits = ev.aggressor_buy
            through, level = ev.price > o.price, ev.price == o.price
        if not hits:
            return
        if through:
            self._fill_maker(ev.t_ns, o.remaining)
        elif level:
            o.trade_vol += ev.qty
            if ev.qty <= o.queue_ahead:
                o.queue_ahead -= ev.qty
            else:
                overflow = ev.qty - o.queue_ahead
                o.queue_ahead = 0.0
                self._fill_maker(ev.t_ns, min(o.remaining, overflow))

    def on_book(self, ev: BookEvent, book: SymbolBook) -> None:
        o = self.order
        if o is None or self.done:
            return
        for side, price, old, new in ev.changes:
            if side == self.own and price == o.price:
                if new < old and not ev.snapshot:
                    unexplained = max(0.0, (old - new) - o.trade_vol)
                    o.queue_ahead -= self.cfg.cancel_credit * unexplained
                o.trade_vol = 0.0
        o.queue_ahead = max(0.0, min(o.queue_ahead, book.qty_at(self.own, o.price)))

    def _fill_maker(self, t_ns: int, qty: float) -> None:
        o = self.order
        qty = min(qty, self.remaining)
        if qty <= 0:
            return
        self.fills.append(Fill(t_ns, o.price, qty, True))
        o.remaining -= qty
        self.remaining -= qty
        if self.remaining <= self.qty_total * 1e-9:
            self.remaining = 0.0
            self.order = None
            self._finish(t_ns)

    def _finish(self, t_ns: int) -> None:
        self.done = True
        self.completed_ns = t_ns
        self.order = None
        self.pending.clear()

    # -- actions ----------------------------------------------------------------
    def _apply(self, action: Action, now: int, book: SymbolBook) -> None:
        if isinstance(action, Post):
            self._post(action.price, now, book)
        elif isinstance(action, Cancel):
            self.order = None
        elif isinstance(action, Cross):
            self.order = None
            self._cross(now, book)

    def _post(self, price: float, now: int, book: SymbolBook) -> None:
        bb, ba = book.best(BID), book.best(ASK)
        crossing = (self.side == "buy" and price >= ba[0]) or (self.side == "sell" and price <= bb[0])
        if crossing or price <= 0:
            self.rejected += 1
            self.last_rejected = True
            return
        self.last_rejected = False
        if self.order is not None and self.order.price == price:
            return  # already resting there; re-posting would only lose queue position
        best_own = bb[0] if self.side == "buy" else ba[0]
        better = price > best_own if self.side == "buy" else price < best_own
        queue = 0.0 if better else book.qty_at(self.own, price) * self.cfg.queue_factor
        self.order = _Order(price, self.remaining, queue, now)
        self.posts += 1

    def _cross(self, now: int, book: SymbolBook) -> None:
        need = self.remaining
        spent = 0.0
        got = 0.0
        for price, qty in book.levels(self.opp):
            take = min(need - got, qty)
            spent += take * price
            got += take
            if got >= need:
                break
        if got < need:  # beyond the recorded depth
            self.exhausted = True
            last = book.levels(self.opp)[-1][0]
            worse = last * (1 + self.cfg.exhaust_penalty) if self.side == "buy" else last * (1 - self.cfg.exhaust_penalty)
            spent += (need - got) * worse
            got = need
        self.fills.append(Fill(now, spent / got, got, False))
        self.remaining = 0.0
        self._finish(now)

    # -- time -------------------------------------------------------------------------
    def step(self, now: int, book: SymbolBook) -> None:
        if self.done:
            return
        while self.pending and self.pending[0][0] <= now:
            _, action = self.pending.pop(0)
            self._apply(action, now, book)
            if self.done:
                return
        if self.forced:
            return
        if now >= self.deadline_ns:
            self.forced = True
            self.pending.append((now + self.cfg.latency_ns, Cross()))
            return
        if now >= self.next_decision_ns:
            self.next_decision_ns = now + self.cfg.decision_interval_ns
            action = self.policy.act(self._state(now, book))
            if action is not None:
                self.pending.append((now + self.cfg.latency_ns, action))

    def _state(self, now: int, book: SymbolBook) -> State:
        bb, ba = book.best(BID), book.best(ASK)
        total = max(self.deadline_ns - self.start_ns, 1)
        o = self.order
        return State(
            now_ns=now, elapsed_s=(now - self.start_ns) / NS, left_s=max(0.0, (self.deadline_ns - now) / NS),
            left_frac=max(0.0, (self.deadline_ns - now) / total), side=self.side,
            remaining_qty=self.remaining, remaining_frac=self.remaining / self.qty_total,
            best_bid=bb[0], best_ask=ba[0], bid_qty=bb[1], ask_qty=ba[1],
            mid=(bb[0] + ba[0]) / 2.0, spread=ba[0] - bb[0], tick=self.cfg.tick, arrival_mid=self.arrival_mid,
            order=OrderView(o.price, o.remaining, o.queue_ahead, (now - o.placed_ns) / NS) if o else None,
            pending=bool(self.pending), last_rejected=self.last_rejected,
        )

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            self.spec_id, self.policy.name, self.side, self.start_ns, self.notional, self.qty_total,
            self.arrival_mid, self.fills, self.completed_ns, self.aborted, self.posts, self.rejected,
            self.exhausted, self.markouts,
        )


# ---- driver --------------------------------------------------------------------------

PolicyFactory = Callable[[], object]


def simulate(events: Iterable, book: SymbolBook, policies: dict[str, PolicyFactory], cfg: SimConfig,
             notional: float, every_ns: int, deadline_ns: int, sides: tuple[str, ...] = ("buy", "sell")) -> list[EpisodeResult]:
    """One pass over ``events`` (from ``Replayer.events``, sharing ``book``). At
    each ``every_ns`` boundary a spec is created per side, and every policy runs
    on identical copies of it so results can be compared pairwise. Episodes
    still running when the data ends are dropped."""
    active: list[Episode] = []
    results: list[EpisodeResult] = []
    heap: list = []
    counter = itertools.count()
    next_start: int | None = None
    spec_id = 0

    def retire(ep: Episode) -> None:
        res = ep.result()
        results.append(res)
        if ep.done and not ep.aborted:
            for f in ep.fills:
                if not f.maker:
                    continue
                for h in cfg.markout_horizons_s:
                    heapq.heappush(heap, (f.t_ns + h * NS, next(counter), res, h, f))

    for ev in events:
        now = ev.t_ns
        if isinstance(ev, GapEvent):
            for ep in active:
                ep.aborted = True
                ep.done = True
                retire(ep)
            active = []
            next_start = None  # restart the schedule after the gap
            continue
        if isinstance(ev, TradeEvent):
            for ep in active:
                ep.on_trade(ev)
        elif isinstance(ev, BookEvent):
            for ep in active:
                ep.on_book(ev, book)
        if not book.ready():
            continue
        if next_start is None:
            next_start = now
        if now >= next_start:
            for side in sides:
                for name, factory in policies.items():
                    active.append(Episode(spec_id, side, notional, now, now + deadline_ns, _Named(factory(), name), cfg, book))
                spec_id += 1
            next_start += every_ns
            while next_start <= now:
                next_start += every_ns
        for ep in active:
            ep.step(now, book)
        for ep in [e for e in active if e.done]:
            retire(ep)
        active = [ep for ep in active if not ep.done]
        if heap and heap[0][0] <= now:
            bb, ba = book.best(BID), book.best(ASK)
            mid = (bb[0] + ba[0]) / 2.0
            while heap and heap[0][0] <= now:
                _, _, res, h, f = heapq.heappop(heap)
                signed = (mid - f.price) / f.price if res.side == "buy" else (f.price - mid) / f.price
                res.markouts[h].append((f.qty, signed))
    return results


class _Named:
    """Gives a policy instance the name it was registered under."""

    def __init__(self, inner, name: str):
        self._inner, self.name = inner, name

    def act(self, state: State):
        return self._inner.act(state)
