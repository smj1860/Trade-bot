"""Baseline execution policies. Each instance is used for exactly one episode.

A policy gets a ``State`` about once a second and returns ``Post(price)``,
``Cancel()``, ``Cross()`` or ``None`` (hold). The engine crosses whatever is
left at the deadline, so every policy completes the order.
"""

from __future__ import annotations

from .engine import Action, Cross, Post, State


def post_price(s: State, inside_ticks: int, decimals: int = 8) -> float:
    """Where to rest: the touch on our own side, or up to ``inside_ticks``
    ticks better while staying strictly inside the spread (never crossing)."""
    room = int(round(s.spread / s.tick)) - 1  # whole ticks available between bid and ask
    k = max(0, min(inside_ticks, room))
    price = s.touch + k * s.tick if s.side == "buy" else s.touch - k * s.tick
    return round(price, decimals)


class MarketNow:
    """Cross immediately. The reference every other policy is compared with."""

    def act(self, s: State) -> Action | None:
        return Cross()


class PostThenCross:
    """Rest one order (at the touch, or ``inside_ticks`` better) and take the rest after ``wait_s``."""

    def __init__(self, inside_ticks: int = 0, wait_s: float = 60.0):
        self.inside_ticks, self.wait_s = inside_ticks, wait_s
        self._posted = False

    def act(self, s: State) -> Action | None:
        if s.last_rejected:
            return Cross()
        if s.elapsed_s >= self.wait_s:
            return Cross()
        if not self._posted and not s.pending:
            self._posted = True
            return Post(post_price(s, self.inside_ticks))
        return None


class PostAndChase:
    """Rest at the touch and re-post when the market moves away, then cross
    when ``cross_left_frac`` of the time is left. Re-posting loses queue position."""

    def __init__(self, inside_ticks: int = 0, reprice_after_s: float = 10.0, max_posts: int = 6, cross_left_frac: float = 0.2):
        self.inside_ticks, self.reprice_after_s = inside_ticks, reprice_after_s
        self.max_posts, self.cross_left_frac = max_posts, cross_left_frac
        self.posts = 0

    def act(self, s: State) -> Action | None:
        if s.last_rejected or s.left_frac <= self.cross_left_frac:
            return Cross()
        if s.pending:
            return None
        if s.order is None:
            self.posts += 1
            return Post(post_price(s, self.inside_ticks))
        behind = s.order.price < s.best_bid if s.side == "buy" else s.order.price > s.best_ask
        if behind and s.order.age_s >= self.reprice_after_s and self.posts < self.max_posts:
            self.posts += 1
            return Post(post_price(s, self.inside_ticks))
        return None


def default_policies() -> dict:
    return {
        "market": MarketNow,
        "post-touch-wait30": lambda: PostThenCross(0, 30),
        "post-touch-wait120": lambda: PostThenCross(0, 120),
        "post-inside-wait30": lambda: PostThenCross(1, 30),
        "post-inside-wait120": lambda: PostThenCross(1, 120),
        "chase-touch": lambda: PostAndChase(0, 10, 6, 0.2),
        "chase-inside": lambda: PostAndChase(1, 10, 6, 0.2),
    }
