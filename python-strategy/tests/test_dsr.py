import math

from strategy.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    kurtosis,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    skewness,
)


def test_sharpe_ratio_of_constant_series_is_zero_not_a_crash():
    # Zero variance -> would divide by zero; must degrade to 0.0 instead.
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_ratio_of_too_few_observations_is_zero():
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([]) == 0.0


def test_sharpe_ratio_matches_hand_computed_value():
    returns = [0.02, -0.01, 0.03, 0.0, 0.01]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    expected = mean / math.sqrt(variance)
    assert sharpe_ratio(returns) == expected


def test_skewness_of_symmetric_series_is_near_zero():
    returns = [-0.02, -0.01, 0.0, 0.01, 0.02]
    assert abs(skewness(returns)) < 1e-9


def test_skewness_of_right_skewed_series_is_positive():
    returns = [-0.01, -0.01, -0.01, -0.01, 0.20]  # one big win, several small losses
    assert skewness(returns) > 0


def test_kurtosis_defaults_to_normal_value_for_too_few_observations():
    assert kurtosis([0.01, 0.02, 0.03]) == 3.0


def test_kurtosis_of_normal_ish_series_is_near_three():
    # A reasonably large, roughly-normal-looking series should land close
    # to (not exactly, it's a small finite sample) the normal value of 3.0.
    import random

    random.seed(42)
    returns = [random.gauss(0, 1) for _ in range(5000)]
    assert 2.5 < kurtosis(returns) < 3.5


def test_psr_at_benchmark_equal_to_observed_is_half():
    # Phi(0) == 0.5: if the observed SR exactly equals the benchmark, the
    # probability of exceeding it is a coin flip.
    assert probabilistic_sharpe_ratio(observed_sr=0.5, benchmark_sr=0.5, n_obs=100) == 0.5


def test_psr_increases_with_more_observations_for_the_same_edge():
    # The same observed edge over the benchmark should become more
    # statistically confident (higher PSR) with more supporting trades.
    psr_small = probabilistic_sharpe_ratio(observed_sr=0.3, benchmark_sr=0.0, n_obs=30)
    psr_large = probabilistic_sharpe_ratio(observed_sr=0.3, benchmark_sr=0.0, n_obs=300)
    assert psr_large > psr_small


def test_psr_below_two_observations_is_uninformative():
    assert probabilistic_sharpe_ratio(observed_sr=0.9, benchmark_sr=0.0, n_obs=1) == 0.5


def test_psr_handles_degenerate_denominator_gracefully():
    # A large positive skew combined with a large observed_sr can drive
    # the denominator (1 - skew*sr + (kurt-1)/4*sr^2) non-positive; must
    # return 0.5, not raise or produce NaN.
    result = probabilistic_sharpe_ratio(observed_sr=10.0, benchmark_sr=0.0, n_obs=50, skew=5.0, kurt=1.0)
    assert result == 0.5


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    # No multiple-comparisons correction needed with only one attempt.
    assert expected_max_sharpe(n_trials=1, variance_of_sr_estimates=0.01) == 0.0
    assert expected_max_sharpe(n_trials=0, variance_of_sr_estimates=0.01) == 0.0


def test_expected_max_sharpe_increases_with_more_trials():
    # More independent attempts -> the expected best-of-N Sharpe ratio
    # under pure luck rises — this is the whole multiple-comparisons point.
    small = expected_max_sharpe(n_trials=5, variance_of_sr_estimates=0.05)
    large = expected_max_sharpe(n_trials=100, variance_of_sr_estimates=0.05)
    assert large > small > 0.0


def test_expected_max_sharpe_scales_with_variance():
    low_var = expected_max_sharpe(n_trials=20, variance_of_sr_estimates=0.01)
    high_var = expected_max_sharpe(n_trials=20, variance_of_sr_estimates=0.04)
    assert high_var > low_var


def test_deflated_sharpe_ratio_is_lower_than_plain_psr_for_many_trials():
    # The whole point of deflation: the same observed SR should look less
    # significant once benchmarked against "best of N" instead of "beats
    # zero" — this is the exact multiple-comparisons correction the audit
    # asked for.
    plain_psr = probabilistic_sharpe_ratio(observed_sr=0.4, benchmark_sr=0.0, n_obs=200)
    dsr_few_trials = deflated_sharpe_ratio(observed_sr=0.4, n_trials=2, n_obs=200)
    dsr_many_trials = deflated_sharpe_ratio(observed_sr=0.4, n_trials=11, n_obs=200)
    assert plain_psr > dsr_few_trials > dsr_many_trials


def test_deflated_sharpe_ratio_with_one_trial_equals_plain_psr_against_zero():
    # n_trials=1 disables the deflation benchmark entirely (see
    # expected_max_sharpe), so DSR should reduce to testing against 0.
    dsr = deflated_sharpe_ratio(observed_sr=0.3, n_trials=1, n_obs=150)
    plain_psr = probabilistic_sharpe_ratio(observed_sr=0.3, benchmark_sr=0.0, n_obs=150)
    assert dsr == plain_psr


def test_deflated_sharpe_ratio_of_a_weak_edge_after_many_trials_is_low():
    # This is the exact failure mode the audit flagged: a marginal edge
    # (small positive Sharpe) that only looks interesting because it's the
    # best of eleven-plus attempts should NOT pass a real significance
    # bar. Regression-guards the headline behavior this module exists for.
    dsr = deflated_sharpe_ratio(observed_sr=0.15, n_trials=11, n_obs=250)
    assert dsr < 0.95
