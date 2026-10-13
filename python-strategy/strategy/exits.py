"""Exit rules for an open position: scaled take-profit tiers, a fee-inclusive
stop that holds through dips while the trend is intact, and a hard stop.

Pure functions only (no gRPC, no I/O) so every rule is unit-testable; engine.py
applies them. All percentages are NET of fees: a 1.5% target nets 1.5% after the
entry and exit fees, a 2% stop loses 2% after fees. With maker fee `f` on both
legs, a long entered at `e` that exits at `p` returns

    net = (p * (1 - f) - e * (1 + f)) / (e * (1 + f))

and a short returns the mirror image, so the price for a given net is

    long : p = e * (1 + fe) * (1 + net) / (1 - fx)
    short: p = e * (1 - fe) * (1 - net) / (1 + fx)

(`fe` entry fee, `fx` exit fee). Note how much of a net 2% stop the fees
consume: at 0.40% per leg only about 1.2% of it is price movement.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, Decimal
from typing import Optional, Sequence

from strategy.policy import Side


def target_price(side: Side, entry: Decimal, net: Decimal, fee_entry: Decimal, fee_exit: Decimal, decimals: int) -> Decimal:
    """Exit price for a NET profit of `net` on a position opened as `side`; rounded away from entry."""
    q = Decimal(1).scaleb(-decimals)
    if side == "BUY":
        return (entry * (1 + fee_entry) * (1 + net) / (1 - fee_exit)).quantize(q, rounding=ROUND_CEILING)
    return (entry * (1 - fee_entry) * (1 - net) / (1 + fee_exit)).quantize(q, rounding=ROUND_FLOOR)


def stop_price(side: Side, entry: Decimal, net_loss: Decimal, fee_entry: Decimal, fee_exit: Decimal) -> Decimal:
    """Price at which closing the position loses `net_loss` (e.g. 0.02) after fees. Unrounded: it is only compared against."""
    if side == "BUY":
        return entry * (1 + fee_entry) * (1 - net_loss) / (1 - fee_exit)
    return entry * (1 - fee_entry) * (1 + net_loss) / (1 + fee_exit)


def trend_intact(side: Side, ema_long_ratio: float) -> bool:
    """True while price sits on the right side of the long EMA for this position's direction.
    0.0 is the 'not enough history' default, which counts as NOT intact (no reason to hold a loser)."""
    return ema_long_ratio > 0 if side == "BUY" else ema_long_ratio < 0


def trend_up_for(side: Side, ema_long_ratio: float) -> bool:
    """Alias with the entry-time meaning: tiers are used only when the trend already agrees with the trade."""
    return trend_intact(side, ema_long_ratio)


def tier_plan(side: Side, entry: Decimal, qty: Decimal, uptrend: bool, tiers: Sequence[Sequence[float]],
              single_net: Decimal, fee_entry: Decimal, fee_exit: Decimal, price_decimals: int,
              qty_decimals: int) -> list[tuple[Decimal, Decimal, int]]:
    """[(price, qty, tier_index)] for resting exits.

    With the trend agreeing with the trade: one order per tier, sized as a fraction of `qty`
    (e.g. 30% at 1.5% net, 20% at 2.5% net, ...); whatever fraction is not assigned is the
    runner and gets no resting order. Otherwise: the whole `qty` at `single_net`, tier index -1.
    Quantities round down to `qty_decimals`; a tier that rounds to zero is dropped."""
    if not uptrend or not tiers:
        return [(target_price(side, entry, single_net, fee_entry, fee_exit, price_decimals), qty, -1)]
    qq = Decimal(1).scaleb(-qty_decimals)
    plan = []
    for i, (net, frac) in enumerate(tiers):
        q = (qty * Decimal(str(frac))).quantize(qq, rounding=ROUND_DOWN)
        if q > 0:
            plan.append((target_price(side, entry, Decimal(str(net)), fee_entry, fee_exit, price_decimals), q, i))
    return plan


def exit_reason(side: Side, avg_entry: Decimal, mid: Decimal, ema_long_ratio: float, tiers_filled: int,
                stop_net: Decimal, hard_net: Decimal, hold_in_trend: bool,
                fee_entry: Decimal, fee_exit: Decimal) -> Optional[str]:
    """Why the whole remaining position should be closed now, or None to hold.

    hard_stop            price is past the hard (net) stop: close regardless of trend.
    stop_trend_broken    price is past the normal (net) stop AND the trend no longer supports the
                         position. A dip past the stop while the trend is intact returns None.
    runner_trend_exit    at least one profit tier has filled and the trend has turned against the
                         position: bank the runner.
    With hold_in_trend False the normal stop applies on price alone."""
    long = side == "BUY"
    soft = stop_price(side, avg_entry, stop_net, fee_entry, fee_exit)
    hard = stop_price(side, avg_entry, hard_net, fee_entry, fee_exit)
    broken = not trend_intact(side, ema_long_ratio)
    if (mid <= hard) if long else (mid >= hard):
        return "hard_stop"
    if (mid <= soft) if long else (mid >= soft):
        if broken or not hold_in_trend:
            return "stop_trend_broken" if hold_in_trend else "stop"
        return None
    if tiers_filled > 0 and broken:
        return "runner_trend_exit"
    return None
