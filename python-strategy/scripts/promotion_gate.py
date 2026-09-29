#!/usr/bin/env python3
"""
Formal model promotion gate — institutional audit Phase 3.2 ("a formal
promotion gate for any model going live: deflated Sharpe, PBO, minimum
track-record length, wired as a release gate").

Why this exists: scripts/train_model.py's walk-forward validation already
answers "did this model beat naive baselines on held-out data," but that
question alone is exactly the kind of question repeated sweeping (this
session ran a 0-5% profit-margin sweep, then a 14-symbol per-symbol-vs-
pooled comparison — 30+ configurations in total) will eventually answer
"yes" for by chance alone, even for a genuinely worthless model, if enough
configurations are tried. This module implements the three standard
statistical corrections for exactly that risk, from the literature that
originated them:

  Deflated Sharpe Ratio (DSR)
      Bailey, D. and Lopez de Prado, M. (2014), "The Deflated Sharpe
      Ratio: Correcting for Selection Bias, Backtest Overfitting, and
      Non-Normality," Journal of Portfolio Management 40(5).
      Answers: given how many configurations were actually tried (and how
      spread out their Sharpe ratios were), how likely is it that this
      candidate's Sharpe ratio reflects real skill rather than the best
      of N noisy draws? Returns a probability in [0, 1]; the gate requires
      it above a threshold (default 0.95) to pass.

  Minimum Track Record Length (MinTRL)
      Same paper. Answers: given this candidate's own observed Sharpe
      ratio (and its return distribution's skew/kurtosis), how many return
      observations would be needed before that Sharpe is statistically
      distinguishable, at a given confidence, from a target (default 0 —
      "better than doing nothing")? The gate requires the candidate's
      *actual* observed track record length to already meet or exceed
      this number — i.e., there's enough history to trust the number at
      all, independent of whether the number itself looks good.

  Probability of Backtest Overfitting (PBO), via Combinatorially
  Symmetric Cross-Validation (CSCV)
      Bailey, D., Borwein, J., Lopez de Prado, M., and Zhu, Q.J. (2015),
      "The Probability of Backtest Overfitting," Journal of Computational
      Finance 20(4). Answers: across every way of splitting the sweep's
      trial history into an in-sample/out-of-sample half, how often would
      picking "whichever configuration looked best in-sample" have
      produced a configuration that was actually below-median out-of-
      sample? A high PBO means the sweep's own selection process is
      unreliable, independent of any single configuration's own numbers.
      Only computable when multiple trial configurations' return series
      over the *same* time periods are available (--pbo-returns-file);
      the gate reports "not computed" rather than skipping the field
      silently when that isn't available.

None of these three checks requires knowing an account's total capital —
they're all computed directly from a return (or per-trade net-P&L)
series, exactly what scripts/train_model.py's simulated net P&L already
produces per trade (see PROMOTION_GATE_INTEGRATION below).

Usage (standalone):
    python3 scripts/promotion_gate.py --returns-file model_returns.json \
        --trial-sharpes-file sweep_sharpes.json

    # model_returns.json: JSON list of per-trade (or per-period) net P&L
    #   floats for the ONE candidate model being evaluated for promotion.
    # sweep_sharpes.json: JSON list of Sharpe ratios from every
    #   configuration actually tried in the sweep that produced this
    #   candidate (including the candidate's own) — this is the "how many
    #   trials, how spread out" input the deflated Sharpe needs. Omit to
    #   fall back to N=1 (no deflation applied — clearly flagged in the
    #   output, since skipping this understates overfitting risk).

Usage (wired into train_model.py):
    python3 scripts/train_model.py --symbol all --folds 1 --gate \
        --gate-trial-sharpes-file sweep_sharpes.json --gate-enforce
    # See train_model.py's --gate/--gate-enforce/--gate-* flags — this
    # runs the same evaluate_gate() below against the production split's
    # own simulated net-P&L series right before the model would be saved,
    # and --gate-enforce refuses to save (exit 1) on a FAIL verdict.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from statistics import NormalDist

_NORMAL = NormalDist()
EULER_MASCHERONI = 0.5772156649015329

# Defaults chosen to be reasonably strict without being untestable on a
# realistic sample size — see each threshold's use in evaluate_gate().
DEFAULT_MIN_DSR = 0.95
DEFAULT_TARGET_SHARPE = 0.0
DEFAULT_TRL_CONFIDENCE = 0.95
DEFAULT_MAX_PBO = 0.5
DEFAULT_N_SLICES = 8  # CSCV slice count for probability_of_backtest_overfitting


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sample_variance(xs: list[float]) -> float:
    """Sample variance (n-1 denominator). Callers must ensure len(xs) >= 2."""
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def sharpe_ratio(returns: list[float]) -> float | None:
    """Non-annualized Sharpe ratio (mean / sample stdev) of a return
    series — deliberately not annualized here, since every formula in this
    module (DSR, MinTRL) is defined in terms of the per-observation Sharpe
    over T observations, not a calendar-annualized figure; annualize only
    for human-facing display, separately, after these checks. `None` for
    fewer than 2 observations or zero variance (nothing to divide by)."""
    if len(returns) < 2:
        return None
    variance = _sample_variance(returns)
    if variance == 0:
        return None
    return _mean(returns) / math.sqrt(variance)


def _skewness(returns: list[float]) -> float:
    """Sample skewness (third standardized moment, population-moment
    convention — g1, not the bias-corrected G1). Zero for a symmetric
    distribution; matches the convention Bailey & Lopez de Prado (2014)
    use for gamma_3 in the DSR/MinTRL formulas."""
    n = len(returns)
    m = _mean(returns)
    variance = sum((x - m) ** 2 for x in returns) / n
    if variance == 0:
        return 0.0
    std = math.sqrt(variance)
    return (sum((x - m) ** 3 for x in returns) / n) / (std**3)


def _kurtosis(returns: list[float]) -> float:
    """Sample kurtosis (fourth standardized moment, NOT excess kurtosis —
    a normal distribution has kurtosis 3.0 under this convention, matching
    gamma_4 in Bailey & Lopez de Prado (2014)'s formulas, which subtract 1
    (not 3) from it directly)."""
    n = len(returns)
    m = _mean(returns)
    variance = sum((x - m) ** 2 for x in returns) / n
    if variance == 0:
        return 3.0  # a degenerate (zero-variance) series is treated as normal-shaped
    std = math.sqrt(variance)
    return (sum((x - m) ** 4 for x in returns) / n) / (std**4)


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """SR0 in Bailey & Lopez de Prado (2014): the Sharpe ratio you'd expect
    the *best* of `n_trials` independent, equally-skilled-at-zero
    strategies to show by chance alone, given how spread out
    (`sharpe_variance`) their Sharpe ratios are. This is the benchmark the
    deflated Sharpe ratio tests the candidate against, instead of testing
    against a flat 0 — the more configurations were tried, the higher this
    bar climbs. Returns 0.0 (no correction) for n_trials <= 1 or zero
    variance, since there's no multiple-testing effect to correct for."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


@dataclass
class DeflatedSharpeResult:
    observed_sharpe: float
    expected_max_sharpe_by_chance: float
    n_trials: int
    n_observations: int
    deflated_sharpe_ratio: float | None  # probability in [0, 1]; None if undefined (degenerate inputs)
    note: str = ""


def deflated_sharpe_ratio(candidate_returns: list[float], trial_sharpes: list[float] | None = None) -> DeflatedSharpeResult:
    """Computes the Deflated Sharpe Ratio for one candidate's return
    series, given the Sharpe ratios of every configuration tried in the
    sweep that produced it (`trial_sharpes`, including the candidate's
    own — this is what sets both `n_trials` and how spread out the sweep's
    results were). Passing `None` or a single-element `trial_sharpes`
    means no other configurations are known to have been tried, which
    disables the multiple-testing correction entirely (SR0 = 0) — the
    result is flagged with a note rather than silently understating
    overfitting risk."""
    sr_hat = sharpe_ratio(candidate_returns)
    t = len(candidate_returns)
    if sr_hat is None:
        return DeflatedSharpeResult(
            observed_sharpe=0.0, expected_max_sharpe_by_chance=0.0, n_trials=len(trial_sharpes or [1]),
            n_observations=t, deflated_sharpe_ratio=None,
            note="candidate return series too short or has zero variance — Sharpe ratio undefined.",
        )

    note = ""
    if not trial_sharpes or len(trial_sharpes) < 2:
        n_trials = 1
        sr0 = 0.0
        note = (
            "no multi-trial Sharpe distribution supplied (or fewer than 2 trials) — multiple-testing "
            "correction disabled (SR0=0); this UNDERSTATES overfitting risk if more than one "
            "configuration was actually tried. Pass --trial-sharpes-file with every trial's Sharpe "
            "ratio for a real deflation."
        )
    else:
        n_trials = len(trial_sharpes)
        sr0 = expected_max_sharpe(_sample_variance(trial_sharpes), n_trials)

    g3 = _skewness(candidate_returns)
    g4 = _kurtosis(candidate_returns)
    if t < 2:
        sigma_sr = None
    else:
        variance_term = 1 - g3 * sr_hat + ((g4 - 1) / 4) * sr_hat**2
        sigma_sr = math.sqrt(variance_term / (t - 1)) if variance_term > 0 else None

    if sigma_sr is None or sigma_sr == 0:
        dsr = None
        note = (note + " " if note else "") + "sigma_SR undefined or zero — deflated Sharpe ratio could not be computed."
    else:
        dsr = _NORMAL.cdf((sr_hat - sr0) / sigma_sr)

    return DeflatedSharpeResult(
        observed_sharpe=sr_hat,
        expected_max_sharpe_by_chance=sr0,
        n_trials=n_trials,
        n_observations=t,
        deflated_sharpe_ratio=dsr,
        note=note,
    )


def minimum_track_record_length(
    candidate_returns: list[float], target_sharpe: float = DEFAULT_TARGET_SHARPE, confidence: float = DEFAULT_TRL_CONFIDENCE
) -> float | None:
    """MinTRL (Bailey & Lopez de Prado, 2014): the minimum number of return
    observations needed for this candidate's OWN observed Sharpe ratio to
    be statistically distinguishable, at `confidence`, from `target_sharpe`
    (default 0 — "no better than doing nothing"). Returns `math.inf` if
    the observed Sharpe doesn't even exceed the target (no amount of
    additional history would help — the strategy needs to actually be
    better, not just more measured), and `None` if the Sharpe ratio itself
    is undefined (see sharpe_ratio)."""
    sr_hat = sharpe_ratio(candidate_returns)
    if sr_hat is None:
        return None
    if sr_hat <= target_sharpe:
        return math.inf
    g3 = _skewness(candidate_returns)
    g4 = _kurtosis(candidate_returns)
    z = _NORMAL.inv_cdf(confidence)
    variance_term = 1 - g3 * sr_hat + ((g4 - 1) / 4) * sr_hat**2
    return 1 + variance_term * (z / (sr_hat - target_sharpe)) ** 2


def _slice_bounds(n_periods: int, n_slices: int) -> list[tuple[int, int]]:
    """Splits `n_periods` chronological periods into `n_slices` contiguous,
    as-equal-as-possible groups (the CSCV combinatorial slicing unit) —
    any remainder periods are distributed one-per-slice starting from the
    first slice, rather than dumped entirely into the last one."""
    base, remainder = divmod(n_periods, n_slices)
    bounds = []
    start = 0
    for s in range(n_slices):
        size = base + (1 if s < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


@dataclass
class PboResult:
    n_trials: int
    n_slices: int
    n_combinations: int
    pbo: float | None
    note: str = ""


def probability_of_backtest_overfitting(trial_returns: list[list[float]], n_slices: int = DEFAULT_N_SLICES) -> PboResult:
    """CSCV-based Probability of Backtest Overfitting (Bailey, Borwein,
    Lopez de Prado & Zhu, 2015). `trial_returns` is one equal-length
    per-period return series per configuration tried in the sweep (same
    time periods across every row — this is a stricter data requirement
    than deflated_sharpe_ratio's trial_sharpes, which only needs each
    trial's already-summarized Sharpe ratio).

    For every way of splitting the `n_slices` chronological slices into
    two equal halves (in-sample IS, out-of-sample OOS): finds whichever
    trial had the best IS Sharpe, then checks that same trial's OOS rank
    among all trials. PBO is the fraction of splits where that trial's OOS
    performance fell in the bottom half — i.e., where picking "whatever
    looked best in-sample" would, out-of-sample, have picked a below-
    median configuration purely from overfitting the sample split, not
    real skill.

    `n_slices` must be even and at least 4 (need at least 2 slices on each
    side to form a meaningful IS/OOS split). Returns pbo=None (with a
    note) if fewer than 2 trials are given, or fewer periods than
    n_slices — there's nothing to compare, or no way to form even one
    non-empty slice."""
    n_trials = len(trial_returns)
    if n_trials < 2:
        return PboResult(n_trials=n_trials, n_slices=n_slices, n_combinations=0, pbo=None, note="need at least 2 trials to compute PBO.")
    if n_slices < 4 or n_slices % 2 != 0:
        raise ValueError(f"n_slices must be even and >= 4 (got {n_slices}).")

    n_periods = len(trial_returns[0])
    if any(len(r) != n_periods for r in trial_returns):
        raise ValueError("every trial's return series must be the same length (same time periods).")
    if n_periods < n_slices:
        return PboResult(
            n_trials=n_trials, n_slices=n_slices, n_combinations=0, pbo=None,
            note=f"only {n_periods} periods available, fewer than n_slices={n_slices}.",
        )

    bounds = _slice_bounds(n_periods, n_slices)
    slice_indices = range(n_slices)
    half = n_slices // 2

    below_median_count = 0
    n_combinations = 0
    for is_slices in itertools.combinations(slice_indices, half):
        is_slices = set(is_slices)
        oos_slices = [s for s in slice_indices if s not in is_slices]

        def _concat(trial: list[float], slices) -> list[float]:
            out = []
            for s in slices:
                start, end = bounds[s]
                out.extend(trial[start:end])
            return out

        is_sharpes = [sharpe_ratio(_concat(trial, is_slices)) for trial in trial_returns]
        # A trial with an undefined IS Sharpe (e.g. zero variance in this
        # particular slice combination) can't be a candidate "IS winner"
        # for this split.
        candidates = [(sr, idx) for idx, sr in enumerate(is_sharpes) if sr is not None]
        if not candidates:
            continue
        _, winner_idx = max(candidates)

        oos_sharpes = [sharpe_ratio(_concat(trial, oos_slices)) for trial in trial_returns]
        defined_oos = [(idx, sr) for idx, sr in enumerate(oos_sharpes) if sr is not None]
        if len(defined_oos) < 2 or oos_sharpes[winner_idx] is None:
            continue

        winner_oos = oos_sharpes[winner_idx]
        rank = sum(1 for idx, sr in defined_oos if sr <= winner_oos)  # 1-indexed rank from the bottom
        relative_rank = rank / len(defined_oos)
        n_combinations += 1
        if relative_rank <= 0.5:
            below_median_count += 1

    if n_combinations == 0:
        return PboResult(n_trials=n_trials, n_slices=n_slices, n_combinations=0, pbo=None, note="no combination produced a comparable IS/OOS split.")
    return PboResult(n_trials=n_trials, n_slices=n_slices, n_combinations=n_combinations, pbo=below_median_count / n_combinations)


@dataclass
class GateVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    dsr: DeflatedSharpeResult | None = None
    min_trl: float | None = None
    pbo: PboResult | None = None


def evaluate_gate(
    candidate_returns: list[float],
    trial_sharpes: list[float] | None = None,
    trial_returns: list[list[float]] | None = None,
    min_dsr: float = DEFAULT_MIN_DSR,
    target_sharpe: float = DEFAULT_TARGET_SHARPE,
    trl_confidence: float = DEFAULT_TRL_CONFIDENCE,
    max_pbo: float = DEFAULT_MAX_PBO,
    n_slices: int = DEFAULT_N_SLICES,
) -> GateVerdict:
    """Runs every check this module implements against one candidate's
    return series and combines them into a single PASS/FAIL verdict — this
    is the function train_model.py's --gate flag calls. A model is
    promoted (passed=True) only when:
      1. Deflated Sharpe Ratio >= min_dsr (default 0.95) — the candidate's
         edge is unlikely to be a lucky draw from however many
         configurations were actually tried.
      2. The candidate's own observed track record (n_observations) is
         >= its MinTRL — there's enough history for check #1's number to
         itself be trustworthy, not just a favorable early read.
      3. PBO <= max_pbo (default 0.5), WHEN COMPUTABLE — if trial_returns
         wasn't supplied, this check is skipped (not failed) and noted as
         not computed, since PBO requires data this project doesn't always
         have on hand (see probability_of_backtest_overfitting's
         docstring for why that's a stricter requirement than DSR's).
    Every reason a check failed (or couldn't run) is collected into
    `reasons` regardless of whether the overall verdict already failed, so
    an operator sees the complete picture in one run rather than fixing
    issues one at a time."""
    reasons: list[str] = []

    dsr_result = deflated_sharpe_ratio(candidate_returns, trial_sharpes)
    if dsr_result.deflated_sharpe_ratio is None:
        reasons.append(f"Deflated Sharpe Ratio could not be computed: {dsr_result.note}")
        dsr_ok = False
    else:
        dsr_ok = dsr_result.deflated_sharpe_ratio >= min_dsr
        if not dsr_ok:
            reasons.append(
                f"Deflated Sharpe Ratio {dsr_result.deflated_sharpe_ratio:.3f} is below the required {min_dsr:.3f} "
                f"(observed Sharpe {dsr_result.observed_sharpe:.3f} vs. expected-by-chance "
                f"{dsr_result.expected_max_sharpe_by_chance:.3f} across {dsr_result.n_trials} trial(s))."
            )
        if dsr_result.note:
            reasons.append(f"note: {dsr_result.note}")

    min_trl = minimum_track_record_length(candidate_returns, target_sharpe, trl_confidence)
    n_obs = len(candidate_returns)
    if min_trl is None:
        reasons.append("Minimum Track Record Length could not be computed (Sharpe ratio undefined).")
        trl_ok = False
    elif math.isinf(min_trl):
        reasons.append(
            f"observed Sharpe does not exceed the target Sharpe ({target_sharpe}) — no amount of "
            "additional history would make this candidate pass; it needs to actually be better, not "
            "just more measured."
        )
        trl_ok = False
    else:
        trl_ok = n_obs >= min_trl
        if not trl_ok:
            reasons.append(
                f"track record has {n_obs} observations, below the required MinTRL of {min_trl:.1f} at "
                f"{trl_confidence:.0%} confidence."
            )

    pbo_result: PboResult | None = None
    pbo_ok = True  # not failing the gate when PBO simply wasn't computable — see docstring
    if trial_returns is not None:
        pbo_result = probability_of_backtest_overfitting(trial_returns, n_slices)
        if pbo_result.pbo is None:
            reasons.append(f"PBO could not be computed: {pbo_result.note}")
        else:
            pbo_ok = pbo_result.pbo <= max_pbo
            if not pbo_ok:
                reasons.append(
                    f"Probability of Backtest Overfitting {pbo_result.pbo:.2f} exceeds the maximum "
                    f"allowed {max_pbo:.2f} across {pbo_result.n_trials} trials / {pbo_result.n_combinations} "
                    "IS/OOS combinations."
                )
    else:
        reasons.append("PBO not computed — no --trial-returns-file supplied (this does not, by itself, fail the gate).")

    return GateVerdict(passed=dsr_ok and trl_ok and pbo_ok, reasons=reasons, dsr=dsr_result, min_trl=min_trl, pbo=pbo_result)


def _print_verdict(verdict: GateVerdict) -> None:
    banner = "PASS" if verdict.passed else "FAIL"
    print(f"=== promotion gate: {banner} ===", file=sys.stderr)
    if verdict.dsr:
        dsr_str = f"{verdict.dsr.deflated_sharpe_ratio:.4f}" if verdict.dsr.deflated_sharpe_ratio is not None else "undefined"
        print(
            f"  Deflated Sharpe Ratio: {dsr_str} (observed Sharpe {verdict.dsr.observed_sharpe:.4f}, "
            f"{verdict.dsr.n_trials} trial(s), {verdict.dsr.n_observations} observations)",
            file=sys.stderr,
        )
    if verdict.min_trl is not None:
        trl_str = "inf (never)" if math.isinf(verdict.min_trl) else f"{verdict.min_trl:.1f}"
        print(f"  Minimum Track Record Length: {trl_str}", file=sys.stderr)
    if verdict.pbo:
        pbo_str = f"{verdict.pbo.pbo:.3f}" if verdict.pbo.pbo is not None else "not computed"
        print(f"  Probability of Backtest Overfitting: {pbo_str}", file=sys.stderr)
    for reason in verdict.reasons:
        print(f"  - {reason}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--returns-file", required=True, help="JSON file: list of the candidate model's per-trade/per-period net P&L floats.")
    parser.add_argument("--trial-sharpes-file", default=None, help="JSON file: list of Sharpe ratios from every configuration tried in the sweep (including the candidate's own).")
    parser.add_argument("--trial-returns-file", default=None, help="JSON file: list of equal-length per-period return series, one per trial configuration, for PBO.")
    parser.add_argument("--min-dsr", type=float, default=DEFAULT_MIN_DSR)
    parser.add_argument("--target-sharpe", type=float, default=DEFAULT_TARGET_SHARPE)
    parser.add_argument("--trl-confidence", type=float, default=DEFAULT_TRL_CONFIDENCE)
    parser.add_argument("--max-pbo", type=float, default=DEFAULT_MAX_PBO)
    parser.add_argument("--n-slices", type=int, default=DEFAULT_N_SLICES)
    parser.add_argument("--json-out", default=None, help="Optional path to also write the verdict as JSON.")
    args = parser.parse_args()

    with open(args.returns_file) as f:
        candidate_returns = json.load(f)
    trial_sharpes = None
    if args.trial_sharpes_file:
        with open(args.trial_sharpes_file) as f:
            trial_sharpes = json.load(f)
    trial_returns = None
    if args.trial_returns_file:
        with open(args.trial_returns_file) as f:
            trial_returns = json.load(f)

    verdict = evaluate_gate(
        candidate_returns,
        trial_sharpes=trial_sharpes,
        trial_returns=trial_returns,
        min_dsr=args.min_dsr,
        target_sharpe=args.target_sharpe,
        trl_confidence=args.trl_confidence,
        max_pbo=args.max_pbo,
        n_slices=args.n_slices,
    )
    _print_verdict(verdict)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "passed": verdict.passed,
                    "reasons": verdict.reasons,
                    "deflated_sharpe_ratio": verdict.dsr.deflated_sharpe_ratio if verdict.dsr else None,
                    "observed_sharpe": verdict.dsr.observed_sharpe if verdict.dsr else None,
                    "min_track_record_length": None if verdict.min_trl is None or math.isinf(verdict.min_trl) else verdict.min_trl,
                    "pbo": verdict.pbo.pbo if verdict.pbo else None,
                },
                f,
                indent=2,
            )

    sys.exit(0 if verdict.passed else 1)


if __name__ == "__main__":
    main()
