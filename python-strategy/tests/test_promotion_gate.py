"""
Unit tests for promotion_gate.py's pure statistics: deflated Sharpe ratio,
minimum track record length, and PBO via CSCV. Uses constructed toy
return series with known expected behavior rather than real market data,
same approach test_train_model.py-style scripts in this repo already use
for their own pure-logic functions.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from promotion_gate import (
    DEFAULT_N_SLICES,
    deflated_sharpe_ratio,
    evaluate_gate,
    expected_max_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def test_sharpe_ratio_is_none_for_fewer_than_two_observations():
    assert sharpe_ratio([1.0]) is None
    assert sharpe_ratio([]) is None


def test_sharpe_ratio_is_none_for_zero_variance():
    assert sharpe_ratio([1.0, 1.0, 1.0]) is None


def test_sharpe_ratio_is_positive_for_a_consistently_profitable_series():
    assert sharpe_ratio([1.0, 2.0, 1.5, 1.8, 1.2]) > 0


def test_sharpe_ratio_is_negative_for_a_consistently_losing_series():
    assert sharpe_ratio([-1.0, -2.0, -1.5, -1.8, -1.2]) < 0


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    assert expected_max_sharpe(sharpe_variance=1.0, n_trials=1) == 0.0


def test_expected_max_sharpe_is_zero_for_zero_variance():
    assert expected_max_sharpe(sharpe_variance=0.0, n_trials=100) == 0.0


def test_expected_max_sharpe_increases_with_more_trials():
    sr0_10 = expected_max_sharpe(sharpe_variance=0.25, n_trials=10)
    sr0_100 = expected_max_sharpe(sharpe_variance=0.25, n_trials=100)
    assert 0.0 < sr0_10 < sr0_100


def test_deflated_sharpe_ratio_flags_missing_trial_data_with_a_note():
    returns = [1.0, 2.0, 1.5, 1.8, 1.2, 0.9, 1.6, 2.1, 1.3, 1.7]
    result = deflated_sharpe_ratio(returns, trial_sharpes=None)
    assert result.n_trials == 1
    assert result.expected_max_sharpe_by_chance == 0.0
    assert "no multi-trial Sharpe distribution" in result.note
    assert result.deflated_sharpe_ratio is not None


def test_deflated_sharpe_ratio_is_lower_with_more_competing_trials():
    returns = [1.0, 2.0, 1.5, 1.8, 1.2, 0.9, 1.6, 2.1, 1.3, 1.7] * 3
    few_trials = [0.3, 0.4, 0.5]
    many_trials = [0.1, 0.5, -0.2, 0.8, -0.4, 0.6, 0.9, -0.1, 0.3, 0.7, -0.3, 0.4, 0.2, -0.5, 0.6, 0.8, -0.2, 0.5, 0.1, 0.9]

    dsr_few = deflated_sharpe_ratio(returns, trial_sharpes=few_trials).deflated_sharpe_ratio
    dsr_many = deflated_sharpe_ratio(returns, trial_sharpes=many_trials).deflated_sharpe_ratio
    assert dsr_many <= dsr_few


def test_deflated_sharpe_ratio_is_none_for_an_undefined_sharpe():
    result = deflated_sharpe_ratio([1.0], trial_sharpes=[0.1, 0.2, 0.3])
    assert result.deflated_sharpe_ratio is None
    assert "too short" in result.note


def test_minimum_track_record_length_is_infinite_when_sharpe_does_not_beat_target():
    returns = [-1.0, -2.0, -0.5, -1.5, -1.0]
    assert minimum_track_record_length(returns, target_sharpe=0.0) == math.inf


def test_minimum_track_record_length_is_none_for_an_undefined_sharpe():
    assert minimum_track_record_length([1.0]) is None


def test_minimum_track_record_length_is_finite_and_positive_for_a_real_edge():
    returns = [1.0, 2.0, 1.5, 1.8, 1.2, 0.9, 1.6, 2.1, 1.3, 1.7]
    mintrl = minimum_track_record_length(returns, target_sharpe=0.0)
    assert mintrl is not None and mintrl > 0 and math.isfinite(mintrl)


def test_minimum_track_record_length_shrinks_with_a_lower_confidence_requirement():
    returns = [1.0, 2.0, 1.5, 1.8, 1.2, 0.9, 1.6, 2.1, 1.3, 1.7]
    strict = minimum_track_record_length(returns, confidence=0.99)
    loose = minimum_track_record_length(returns, confidence=0.80)
    assert loose < strict


def _make_trial(good: bool, n_periods: int = 40, seed: int = 0) -> list[float]:
    import random

    rng = random.Random(seed)
    if good:
        return [rng.gauss(0.5, 1.0) for _ in range(n_periods)]
    return [rng.gauss(0.0, 1.0) for _ in range(n_periods)]


def test_pbo_requires_at_least_two_trials():
    result = probability_of_backtest_overfitting([[1.0, 2.0, 3.0, 4.0]])
    assert result.pbo is None
    assert "at least 2 trials" in result.note


def test_pbo_rejects_odd_or_too_small_n_slices():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0] * 8, [2.0] * 8], n_slices=3)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0] * 8, [2.0] * 8], n_slices=2)


def test_pbo_rejects_mismatched_trial_lengths():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0]], n_slices=4)


def test_pbo_reports_not_computed_with_fewer_periods_than_slices():
    result = probability_of_backtest_overfitting([[1.0, 2.0], [3.0, 4.0]], n_slices=4)
    assert result.pbo is None
    assert "fewer than n_slices" in result.note


def test_pbo_is_low_for_one_genuinely_and_consistently_better_trial():
    # One trial is drawn from a distinctly better distribution throughout
    # every period (not just in-sample) -- the "genuinely skilled" case,
    # where picking the IS winner should keep winning OOS too.
    trials = [_make_trial(good=True, seed=1), _make_trial(good=False, seed=2), _make_trial(good=False, seed=3), _make_trial(good=False, seed=4)]
    result = probability_of_backtest_overfitting(trials, n_slices=8)
    assert result.pbo is not None
    assert result.pbo < 0.5


def test_pbo_is_near_half_for_trials_with_no_real_difference():
    # All trials drawn from the identical distribution -- whichever looks
    # best in-sample is essentially a coin flip out-of-sample, so PBO
    # should land close to 0.5 (not exactly, given finite-sample noise).
    trials = [_make_trial(good=False, seed=s) for s in range(6)]
    result = probability_of_backtest_overfitting(trials, n_slices=8)
    assert result.pbo is not None
    assert 0.25 <= result.pbo <= 0.85


def test_evaluate_gate_fails_without_enough_trial_data_and_short_history():
    # A short, noisy return series with no trial context should not pass.
    returns = [0.1, -0.05, 0.02, -0.03, 0.01]
    verdict = evaluate_gate(returns)
    assert not verdict.passed
    assert len(verdict.reasons) > 0


def test_evaluate_gate_notes_pbo_not_computed_when_no_trial_returns_given():
    returns = [1.0, 2.0, 1.5, 1.8, 1.2, 0.9, 1.6, 2.1, 1.3, 1.7] * 5
    verdict = evaluate_gate(returns, trial_sharpes=[0.1, 0.2, 0.15])
    assert any("PBO not computed" in r for r in verdict.reasons)


def test_evaluate_gate_can_pass_with_a_strong_long_consistent_track_record():
    # A long, strongly and consistently profitable series with a tight
    # trial-Sharpe distribution around it should be able to clear every
    # check -- this is the "obviously good enough" sanity check that the
    # gate isn't simply impossible to pass.
    import random

    rng = random.Random(42)
    returns = [rng.gauss(1.0, 0.5) for _ in range(500)]
    trial_sharpes = [rng.gauss(1.8, 0.1) for _ in range(20)]
    verdict = evaluate_gate(returns, trial_sharpes=trial_sharpes, min_dsr=0.95)
    assert verdict.passed, verdict.reasons


def test_evaluate_gate_default_n_slices_matches_module_default():
    assert DEFAULT_N_SLICES % 2 == 0 and DEFAULT_N_SLICES >= 4
