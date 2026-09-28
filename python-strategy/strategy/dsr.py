"""Deflated Sharpe Ratio (DSR) — the statistical-significance gate for the
sequential-data-snooping problem the institutional audit flagged
(claude/institutional-audit-2026-09-27.md): eleven-plus rounds of "try a
feature/label/horizon, look at the same held-out accuracy or net P&L,
iterate" is a classic multiple-comparisons problem. A result that looks
good after the Nth attempt is more likely to be noise than a result that
looked good on the first attempt, and neither raw accuracy nor raw net P&L
says which one you're looking at.

This module implements Bailey & Lopez de Prado's Probabilistic Sharpe Ratio
(PSR) and Deflated Sharpe Ratio (DSR) ("The Sharpe Ratio Efficient
Frontier", 2012 and "The Deflated Sharpe Ratio", 2014). In one sentence:
DSR asks "what's the probability the true Sharpe ratio is actually
positive, once you account for the fact that this is the best of N
attempts, not the only attempt?" — a single number that replaces "does
this beat the baseline" with "would this still look good if it had been
the only thing tried."

This is deliberately a pure-math module with no I/O, no database access,
and no dependency on scikit-learn/joblib — everything here operates on a
plain list of per-trade returns (or summary statistics already computed
from one), so it's usable from scripts/evaluate_holdout.py and testable in
isolation. Uses Python's stdlib statistics.NormalDist for the normal CDF
and its inverse rather than adding a scipy dependency purely for two
functions.

Terminology used throughout:
  SR       Sharpe ratio: mean per-trade return / stdev of per-trade return
           (NOT annualized — this pipeline's "returns" are already
           per-trade net P&L fractions after fees, see train_model.py's
           net_pnl(), so there's no meaningful trading-days-per-year
           annualization factor to apply; DSR only cares about relative
           comparisons anyway, so leaving it unannualized changes nothing
           about the pass/fail verdict).
  T        Number of return observations (trades) the SR was computed
           from.
  N        Number of independent trials (configurations) that were tried
           before this one — the same "eleven rounds" the audit is
           talking about. This has to be supplied honestly; DSR cannot
           discover it from the data. See evaluate_holdout.py's
           --num-trials.
  skew     Sample skewness of the per-trade returns (0.0 for a symmetric
           distribution).
  kurtosis Sample (non-excess, i.e. normal == 3.0) kurtosis of the
           per-trade returns.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

_NORMAL = NormalDist(mu=0.0, sigma=1.0)

# Euler-Mascheroni constant, used by expected_max_sharpe's closed-form
# approximation of E[max of N Sharpe-ratio estimates] (Bailey & Lopez de
# Prado 2014, eq. 10).
_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Mean / sample-stdev of `returns`. 0.0 if fewer than 2 observations
    or the sample stdev is 0 (a degenerate, all-identical series) rather
    than raising — a training/evaluation run with too few trades to say
    anything statistically meaningful should report "no signal", not
    crash."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev == 0.0:
        return 0.0
    return mean / stdev


def skewness(returns: Sequence[float]) -> float:
    """Sample skewness (third standardized moment, Fisher-Pearson
    coefficient with the common bias-adjustment omitted — PSR's own
    derivation uses the plain moment-based estimator, not the
    bias-corrected G1 variant). 0.0 for fewer than 3 observations or a
    zero-variance series."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    if variance == 0.0:
        return 0.0
    stdev = math.sqrt(variance)
    third_moment = sum((r - mean) ** 3 for r in returns) / n
    return third_moment / (stdev**3)


def kurtosis(returns: Sequence[float]) -> float:
    """Sample kurtosis (fourth standardized moment, NOT excess kurtosis —
    a normal distribution scores 3.0 here, matching the convention PSR's
    denominator term `(kurtosis - 1) / 4` expects, per Bailey & Lopez de
    Prado). 3.0 (the normal-distribution value, i.e. "assume no excess
    kurtosis") for fewer than 4 observations or a zero-variance series,
    rather than an undefined 0.0 — a degenerate input shouldn't quietly
    make the PSR denominator smaller (and the resulting PSR falsely more
    confident) than the honest "we don't know, assume normal" answer."""
    n = len(returns)
    if n < 4:
        return 3.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    if variance == 0.0:
        return 3.0
    fourth_moment = sum((r - mean) ** 4 for r in returns) / n
    return fourth_moment / (variance**2)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """PSR(benchmark_sr): the probability that the strategy's TRUE Sharpe
    ratio exceeds `benchmark_sr`, given an observed Sharpe ratio
    `observed_sr` estimated from `n_obs` per-trade returns with sample
    `skew` and `kurt` (non-excess, normal == 3.0).

    Formula (Bailey & Lopez de Prado 2012, eq. 7):
        PSR = Phi( (observed_sr - benchmark_sr) * sqrt(n_obs - 1)
                   / sqrt(1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2) )
    where Phi is the standard normal CDF. Skew/excess-kurtosis correct for
    the observed return distribution not actually being normal (fat tails
    or asymmetric wins/losses, both routine for a fee-thresholded trading
    label), which would otherwise bias a plain t-test.

    Returns 0.5 (maximally uninformative) if n_obs < 2 or the denominator
    is non-positive (a degenerate skew/kurtosis/SR combination) rather
    than raising or dividing by zero — this can happen for a very small
    or very skewed sample, and "we can't tell" is the honest answer, not
    an error."""
    if n_obs < 2:
        return 0.5
    denominator_sq = 1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr**2
    if denominator_sq <= 0.0:
        return 0.5
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denominator_sq)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, variance_of_sr_estimates: float) -> float:
    """E[max(SR_1, ..., SR_n_trials)] under the null hypothesis that every
    trial's true Sharpe ratio is 0 (no real skill, only noise) — the
    "how good would the best of N pure-luck attempts look" benchmark that
    deflated_sharpe_ratio tests the observed Sharpe ratio against, instead
    of testing it against a flat 0.

    Closed-form approximation (Bailey & Lopez de Prado 2014, eq. 10):
        E[max SR_n] ≈ sqrt(V) * ((1 - γ) * Φ^-1(1 - 1/N) + γ * Φ^-1(1 - 1/(N*e)))
    where γ is the Euler-Mascheroni constant, Φ^-1 is the inverse standard
    normal CDF, and V (`variance_of_sr_estimates`) is the variance of the
    N trials' individual Sharpe-ratio estimates (see
    deflated_sharpe_ratio's docstring for how this pipeline approximates
    V when the individual trial Sharpe ratios weren't all logged).

    Returns 0.0 for n_trials <= 1 — with only one trial, there is no
    multiple-comparisons inflation to correct for, so the benchmark
    collapses to "no deflation" rather than the formula's ill-defined
    Φ^-1(0) at N=1."""
    if n_trials <= 1:
        return 0.0
    if variance_of_sr_estimates <= 0.0:
        return 0.0
    inv_cdf_a = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    inv_cdf_b = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(variance_of_sr_estimates) * (
        (1.0 - _EULER_MASCHERONI) * inv_cdf_a + _EULER_MASCHERONI * inv_cdf_b
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    variance_of_sr_estimates: float | None = None,
) -> float:
    """The probability that the observed strategy's true Sharpe ratio
    exceeds the expected maximum Sharpe ratio achievable by pure chance
    across `n_trials` independent attempts — i.e. PSR benchmarked against
    expected_max_sharpe(n_trials, ...) instead of against 0. This is what
    the institutional audit means by "add a PBO or deflated Sharpe check
    before calling any future result a 'win'": passing a plain
    accuracy/net-P&L comparison after the Nth sweep round proves nothing
    on its own, because the Nth-best of N noisy attempts looks good even
    when none of them has real skill. DSR folds that multiple-comparisons
    correction into a single number.

    `n_trials` must be supplied honestly — it's the count of distinct
    configurations (feature sets, horizons, label schemes, thresholds)
    evaluated against the SAME sealed-off holdout window before this
    particular result, not just the count of times this exact
    configuration was run. See scripts/evaluate_holdout.py's --num-trials
    docstring for the discipline this requires: the count has to be kept
    honestly across the whole research program, not reset per session.

    `variance_of_sr_estimates`, if not supplied, defaults to
    1 / max(n_obs - 1, 1) — the sampling variance of a SINGLE Sharpe-ratio
    estimate under the null of a zero true Sharpe ratio (a standard
    simplifying assumption from Bailey & Lopez de Prado's own worked
    examples, used whenever the individual Sharpe-ratio estimates of the
    other N-1 trials weren't all individually recorded, which is the
    normal case here — most of this project's prior rounds reported
    accuracy/net P&L, not a logged per-round Sharpe ratio). Passing an
    explicit value only helps if the actual per-trial Sharpe-ratio
    variance across the real N trials is known to differ meaningfully
    from this default.

    Interpretation: DSR >= 0.95 is the customary "statistically
    significant at the 95% level, after correcting for N trials" bar
    (evaluate_holdout.py uses this as its default pass threshold, matching
    conventional PSR/DSR practice) — anything below it means "this could
    plausibly be the best of N noisy attempts," which is exactly the
    failure mode the audit is warning about."""
    if variance_of_sr_estimates is None:
        variance_of_sr_estimates = 1.0 / max(n_obs - 1, 1)
    benchmark_sr = expected_max_sharpe(n_trials, variance_of_sr_estimates)
    return probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew, kurt=kurt)
