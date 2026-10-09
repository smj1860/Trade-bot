"""Whole-trade replay: entry, target/stop/timeout exits, against the recorded book.

A paper trade (strategy/paper.py) enters at a bar's close and leaves at the
first of: target touched, stop touched, or ``horizon`` bars later. The paper
numbers assume both legs happen exactly at those prices for free apart from a
flat round-trip cost. Here the same trade is replayed against the real
recorded order book, with fills decided by execution_sim.engine's model:

* Entry: an episode (market, or post-and-chase for ``entry_s``) starting at the
  bar close. Barriers are anchored at the arrival mid.
* Target: either a taker exit triggered when a trade prints at/through the
  target price, or a resting post-only limit at the target (maker).
* Stop: always a taker exit, triggered when a trade prints at/through the stop
  price, executed ``latency`` later against the book as it is then.
* Timeout: the remainder is crossed at the vertical barrier, or (``chase_timeout``)
  worked as post-and-chase during the last ``exit_s`` seconds first.

Simplifications, all of which should be kept in mind when reading results:

* Barriers are not monitored during the entry window; the position counts from
  the moment the entry completes, and the timeout is still measured from the
  bar close, as in the paper trade.
* In the final chase window the resting target is replaced by the chase order;
  a price beyond the target simply fills the chase at the touch.
* Stops trigger on trade prints, not on quotes.
* A recording gap anywhere in the trade's life excludes the trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import NS, Cross, Episode, Fees, Fill, Post, SimConfig
from .policies import MarketNow, PostAndChase
from .replay import ASK, BID, BookEvent, GapEvent, MultiReplayer, TradeEvent

BIG = 1 << 62
ENTRY_S = 600
EXIT_S = 600
MAX_START_LAG_S = 30


class _Hold:
    name = "hold"

    def act(self, state):
        return None


@dataclass(frozen=True, slots=True)
class TradeSpec:
    uid: int
    symbol: str  # as recorded, e.g. BTC/USD
    direction: int  # 1 long, -1 short
    t0_ns: int  # bar close: when the paper trade enters
    barrier: float
    hold_ns: int  # horizon: timeout is t0 + hold


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    passive_entry: bool
    maker_target: bool
    chase_timeout: bool

    @property
    def passive(self) -> bool:
        return self.passive_entry or self.maker_target or self.chase_timeout


SCENARIOS = (
    Scenario("all-taker", False, False, False),
    Scenario("entry-passive", True, False, False),
    Scenario("exit-passive", False, True, True),
    Scenario("all-passive", True, True, True),
)


@dataclass(slots=True)
class Outcome:
    uid: int
    scenario: str
    variant: int
    aborted: bool
    long: bool
    ref: float
    qty: float
    reason: str | None
    entry_fills: list[Fill]
    exit_fills: list[Fill]
    end_ns: int = 0


def returns(o: Outcome, fees: Fees) -> tuple[float, float, float]:
    """(gross, fees, net) as fractions of entry notional (qty x arrival mid)."""
    base = o.qty * o.ref
    buy = o.entry_fills if o.long else o.exit_fills
    sell = o.exit_fills if o.long else o.entry_fills
    gross = (sum(f.price * f.qty for f in sell) - sum(f.price * f.qty for f in buy)) / base
    fee = sum((fees.maker if f.maker else fees.taker) * f.price * f.qty for f in o.entry_fills + o.exit_fills) / base
    return gross, fee, gross - fee


class Lifecycle:
    def __init__(self, spec: TradeSpec, scen: Scenario, variant: int, cfg: SimConfig, book, notional: float,
                 entry_s: float = ENTRY_S, exit_s: float = EXIT_S):
        self.spec, self.scen, self.variant, self.cfg, self.book, self.notional = spec, scen, variant, cfg, book, notional
        self.long = spec.direction > 0
        self.entry_side, self.exit_side = ("buy", "sell") if self.long else ("sell", "buy")
        self.timeout_ns = spec.t0_ns + spec.hold_ns
        self.window_ns = self.timeout_ns - int(exit_s * NS)
        self.exit_s = exit_s
        policy = PostAndChase(0, 10, 6, 0.2) if scen.passive_entry else MarketNow()
        self.entry = Episode(0, self.entry_side, notional, spec.t0_ns, spec.t0_ns + int(entry_s * NS), policy, cfg, book)
        self.ref = self.entry.arrival_mid
        self.cur: Episode = self.entry
        self.phase = "entry"
        self.target = self.stop = 0.0
        self.qty = self.entry.qty_total
        self.triggered = False
        self.window_started = False
        self.fallback = False
        self.reason: str | None = None
        self.retired_fills: list[Fill] = []
        self.done = False
        self.aborted = False
        self.wake_ns = 0
        self.reg: tuple | None = None  # (side, price) of the resting order as registered by the driver

    # -- helpers ------------------------------------------------------------------
    def order_key(self) -> tuple | None:
        o = self.cur.order
        return (self.cur.own, o.price) if o is not None else None

    def _trigger(self, reason: str, t_ns: int) -> None:
        self.triggered = True
        self.reason = reason
        self.cur.pending.append((t_ns + self.cfg.latency_ns, Cross()))
        self.wake_ns = min(self.wake_ns, t_ns + self.cfg.latency_ns)

    def _update_wake(self) -> None:
        cur, w = self.cur, BIG
        if cur.pending:
            w = cur.pending[0][0]
        if not cur.forced:
            w = min(w, cur.deadline_ns, cur.next_decision_ns)
        if self.phase == "hold" and self.scen.chase_timeout and not self.window_started and not self.triggered:
            w = min(w, self.window_ns)
        self.wake_ns = w

    # -- events -------------------------------------------------------------------
    def on_trade(self, ev: TradeEvent) -> bool:
        """True if something structural changed (order gone, trigger, completion)."""
        if self.done:
            return False
        cur = self.cur
        before = cur.order
        dirty = False
        if self.phase == "hold" and not self.triggered and not cur.done:
            p = ev.price
            if (p <= self.stop) if self.long else (p >= self.stop):
                self._trigger("stop", ev.t_ns)
                dirty = True
            elif not self.scen.maker_target and ((p >= self.target) if self.long else (p <= self.target)):
                self._trigger("target", ev.t_ns)
                dirty = True
        if before is not None or self.phase == "entry":
            cur.on_trade(ev)
            if cur.done:
                self.wake_ns = 0
                dirty = True
            elif cur.order is not before:
                dirty = True
        return dirty

    def on_book(self, ev: BookEvent) -> None:
        if not self.done:
            self.cur.on_book(ev, self.book)

    def abort(self) -> None:
        self.aborted = True
        self.done = True

    # -- time ---------------------------------------------------------------------
    def step(self, now: int) -> None:
        if self.done or now < self.wake_ns:
            return
        book, cur = self.book, self.cur
        if self.phase == "hold" and self.scen.chase_timeout and not self.window_started and not self.triggered and now >= self.window_ns:
            self._start_window(now)
            cur = self.cur
        cur.step(now, book)
        if self.phase == "hold" and self.scen.maker_target and cur is not self.entry and not self.window_started \
                and not self.triggered and not self.fallback and not cur.done and cur.rejected and cur.order is None \
                and not cur.pending:
            # the market was already beyond the target when we tried to rest there: take it
            self.fallback = True
            self.reason = "target"
            cur.pending.append((now + self.cfg.latency_ns, Cross()))
        if cur.done:
            if self.phase == "entry":
                self._entered(now)
            else:
                self._finish()
        if not self.done:
            self._update_wake()

    def _entered(self, now: int) -> None:
        self.phase = "hold"
        b = self.spec.barrier
        if self.long:
            self.target, self.stop = self.ref * (1 + b), self.ref * (1 - b)
        else:
            self.target, self.stop = self.ref * (1 - b), self.ref * (1 + b)
        ex = Episode(1, self.exit_side, self.notional, now, self.timeout_ns, _Hold(), self.cfg, self.book, qty=self.qty)
        ex.next_decision_ns = BIG
        if self.scen.maker_target:
            ex.pending.append((now + self.cfg.latency_ns, Post(self.target)))
        self.cur = ex

    def _start_window(self, now: int) -> None:
        ex1 = self.cur
        ex1.order = None
        ex1.pending.clear()
        ex1.done = True
        self.retired_fills += ex1.fills
        self.window_started = True
        rem = ex1.remaining
        if rem <= self.qty * 1e-9:
            self._finish()
            return
        self.cur = Episode(2, self.exit_side, self.notional, now, self.timeout_ns, PostAndChase(0, 10, 6, 0.2),
                           self.cfg, self.book, qty=rem)

    def _finish(self) -> None:
        self.done = True
        if self.reason is None:
            if self.window_started or self.cur.completed_ns is None or self.cur.completed_ns >= self.timeout_ns:
                self.reason = "timeout"
            else:
                self.reason = "target"

    def outcome(self) -> Outcome:
        return Outcome(self.spec.uid, self.scen.name, self.variant, self.aborted, self.long, self.ref, self.qty,
                       self.reason, list(self.entry.fills), self.retired_fills + (list(self.cur.fills) if self.phase == "hold" else []),
                       self.cur.completed_ns or 0)


# ---- driver ----------------------------------------------------------------------------

@dataclass
class RunStats:
    specs: int = 0
    started: int = 0
    no_data_at_entry: int = 0
    lifecycles: int = 0
    aborted: int = 0
    unfinished: int = 0
    notes: list[str] = field(default_factory=list)


def run_lifecycles(specs: list[TradeSpec], rep: MultiReplayer, cfgs: dict[str, SimConfig],
                   variants: list[tuple[float, float]], scenarios=SCENARIOS, notional: float = 1000.0,
                   entry_s: float = ENTRY_S, exit_s: float = EXIT_S) -> tuple[list[Outcome], RunStats]:
    """One pass over the recordings. ``variants`` are (queue_factor, cancel_credit)
    pairs; scenarios with no passive leg are run once (variant 0) since the
    assumptions cannot affect them. ``cfgs`` maps symbol -> SimConfig (tick etc.)."""
    stats = RunStats(specs=len(specs))
    if not specs:
        return [], stats
    starts: dict[str, list[TradeSpec]] = {s: [] for s in rep.books}
    for sp in specs:
        starts[sp.symbol].append(sp)
    for lst in starts.values():
        lst.sort(key=lambda x: x.t0_ns)
    ptr = {s: 0 for s in starts}
    live: dict[str, list[Lifecycle]] = {s: [] for s in starts}
    watch: dict[str, dict[tuple, list[Lifecycle]]] = {s: {} for s in starts}
    sym_wake: dict[str, int] = {s: BIG for s in starts}
    outcomes: list[Outcome] = []
    t_lo = min(sp.t0_ns for sp in specs)
    t_hi = max(sp.t0_ns + sp.hold_ns for sp in specs) + int((exit_s + 60) * NS)
    lag = MAX_START_LAG_S * NS

    def sync(sym: str, lc: Lifecycle) -> None:
        new = lc.order_key()
        if new == lc.reg:
            return
        idx = watch[sym]
        if lc.reg is not None:
            lst = idx.get(lc.reg)
            if lst is not None:
                if lc in lst:
                    lst.remove(lc)
                if not lst:
                    del idx[lc.reg]
        if new is not None:
            idx.setdefault(new, []).append(lc)
        lc.reg = new

    def drop(sym: str, lc: Lifecycle) -> None:
        key = lc.reg
        if key is not None:
            lst = watch[sym].get(key)
            if lst is not None and lc in lst:
                lst.remove(lc)
                if not lst:
                    del watch[sym][key]
            lc.reg = None
        if lc.aborted:
            stats.aborted += 1
        else:
            outcomes.append(lc.outcome())

    for sym, ev in rep.events(start_ns=t_lo - NS, end_ns=t_hi):
        lcs = live[sym]
        if isinstance(ev, GapEvent):
            for lc in lcs:
                lc.abort()
                drop(sym, lc)
            live[sym] = []
            sym_wake[sym] = BIG
            continue
        now = ev.t_ns
        book = rep.books[sym]
        if isinstance(ev, TradeEvent):
            for lc in lcs:
                if lc.on_trade(ev):
                    sync(sym, lc)
                    if lc.wake_ns < sym_wake[sym]:
                        sym_wake[sym] = lc.wake_ns
        else:
            idx = watch[sym]
            if idx:
                for c in ev.changes:
                    lst = idx.get((c[0], c[1]))
                    if lst:
                        for lc in lst:
                            lc.on_book(ev)
        st = starts[sym]
        i = ptr[sym]
        if i < len(st) and st[i].t0_ns <= now and book.ready():
            while i < len(st) and st[i].t0_ns <= now:
                sp = st[i]
                i += 1
                if now - sp.t0_ns > lag:
                    stats.no_data_at_entry += 1
                    continue
                stats.started += 1
                for scen in scenarios:
                    for vi in range(len(variants) if scen.passive else 1):
                        qf, cc = variants[vi]
                        c = cfgs[sym]
                        cfg = SimConfig(tick=c.tick, price_decimals=c.price_decimals, latency_ns=c.latency_ns,
                                        queue_factor=qf, cancel_credit=cc, decision_interval_ns=c.decision_interval_ns,
                                        exhaust_penalty=c.exhaust_penalty, markout_horizons_s=())
                        lc = Lifecycle(sp, scen, vi, cfg, book, notional, entry_s, exit_s)
                        lcs.append(lc)
                        stats.lifecycles += 1
                sym_wake[sym] = 0
            ptr[sym] = i
        if now >= sym_wake[sym]:
            finished = False
            wake = BIG
            for lc in lcs:
                lc.step(now)
                if lc.done:
                    finished = True
                    continue
                sync(sym, lc)
                if lc.wake_ns < wake:
                    wake = lc.wake_ns
            if finished:
                keep = []
                for lc in lcs:
                    if lc.done:
                        drop(sym, lc)
                    else:
                        keep.append(lc)
                live[sym] = keep
            sym_wake[sym] = wake
    for sym, lcs in live.items():
        stats.unfinished += len(lcs)
    return outcomes, stats
