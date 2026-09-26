"""
ModelWrapper: the interface between "a set of features" and "a trading
signal in [-1, 1]" (positive = bullish, negative = bearish, magnitude =
conviction). Every concrete model — rule-based today, a trained
scikit-learn or PyTorch model later — implements this same interface, so
swapping one in for another is a one-line config change
(`strategy.model.kind`), never a strategy-code change.

Signal generation is deliberately kept separate from the decision policy
(strategy/policy.py) that turns a signal into an order: a model here only
ever answers "what do I think", never "what should we do about it" —
sizing and thresholds are a separate, auditable step.

RuleBasedModel is the only one of these actually exercised end-to-end in
this project so far. SklearnModelWrapper and TorchModelWrapper are real,
working implementations of the same interface, but there is no trained
model or historical data in this project to load into them — they're
tested here against a toy model built on the fly (see tests/test_models.py),
not against anything meaningful for actual trading. Don't read their
presence as "an ML model was validated for this strategy" — it wasn't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from strategy.features import Features


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ModelWrapper(ABC):
    """`predict` takes a Features snapshot and returns a signal in
    [-1, 1]. Implementations must clip their own output into that range —
    callers are entitled to assume it's already bounded."""

    @abstractmethod
    def predict(self, features: Features) -> float:
        raise NotImplementedError


class RuleBasedModel(ModelWrapper):
    """Combines order-book imbalance and short-term momentum into a single
    signal via a fixed weighted sum. No training, no historical
    calibration — it's an explicit, auditable rule, which is exactly why
    it's the baseline: every number in the decision is visible in this
    file and in config, not learned from data nobody has yet."""

    def __init__(self, imbalance_weight: float, momentum_weight: float) -> None:
        self.imbalance_weight = imbalance_weight
        self.momentum_weight = momentum_weight

    def predict(self, features: Features) -> float:
        # Momentum is a fractional price change (e.g. 0.002 = 0.2%), a much
        # smaller-magnitude number than imbalance (already in [-1, 1]).
        # Scale it up so it can actually compete with imbalance in the
        # weighted sum instead of always being drowned out — 100x turns a
        # 1% move over the window into a magnitude-1 contribution.
        scaled_momentum = _clip(features.momentum * 100.0)
        signal = self.imbalance_weight * features.imbalance + self.momentum_weight * scaled_momentum
        return _clip(signal)


class SklearnModelWrapper(ModelWrapper):
    """Loads a scikit-learn estimator via joblib. Supports either a
    classifier with `predict_proba` (mapped from a [0, 1] "probability of
    up" into a [-1, 1] signal) or a regressor whose `predict` output is
    assumed to already be roughly in [-1, 1] and is clipped defensively.

    `feature_order` must list the Features attribute names in the exact
    order the model was trained on — there is no way to infer this from
    the model file itself, so getting it wrong silently feeds the model
    garbage. This is on you to get right when you actually have a model.
    """

    def __init__(self, model_path: str, feature_order: Sequence[str]):
        try:
            import joblib
        except ImportError as e:
            raise ImportError(
                "SklearnModelWrapper requires scikit-learn and joblib — "
                "install them (see requirements-ml.txt) before using strategy.model.kind = 'sklearn'"
            ) from e
        if not feature_order:
            raise ValueError("feature_order must be non-empty for SklearnModelWrapper")
        self._model = joblib.load(model_path)
        self._feature_order = list(feature_order)

    def _feature_vector(self, features: Features) -> list[float]:
        values: Mapping[str, float] = {
            "mid_price": float(features.mid_price),
            "spread": float(features.spread),
            "imbalance": features.imbalance,
            "momentum": features.momentum,
        }
        try:
            return [values[name] for name in self._feature_order]
        except KeyError as e:
            raise ValueError(f"feature_order references an unknown feature: {e}") from e

    def predict(self, features: Features) -> float:
        vector = [self._feature_vector(features)]
        if hasattr(self._model, "predict_proba"):
            proba_up = self._model.predict_proba(vector)[0][-1]
            return _clip((proba_up - 0.5) * 2)
        raw = self._model.predict(vector)[0]
        return _clip(float(raw))


class TorchModelWrapper(ModelWrapper):
    """Loads a TorchScript model (`torch.jit.load`) rather than a raw
    `torch.save`d model, deliberately: a TorchScript module is
    self-contained (no dependency on the original Python class definition
    being importable at load time), which matters for a long-running
    service loading a model that may have been trained somewhere else
    entirely. Assumes a single scalar output per forward pass.
    """

    def __init__(self, model_path: str, feature_order: Sequence[str]):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "TorchModelWrapper requires PyTorch — install it (see requirements-ml.txt) "
                "before using strategy.model.kind = 'torch'"
            ) from e
        if not feature_order:
            raise ValueError("feature_order must be non-empty for TorchModelWrapper")
        self._torch = torch
        self._model = torch.jit.load(model_path)
        self._model.eval()
        self._feature_order = list(feature_order)

    def _feature_vector(self, features: Features) -> list[float]:
        values: Mapping[str, float] = {
            "mid_price": float(features.mid_price),
            "spread": float(features.spread),
            "imbalance": features.imbalance,
            "momentum": features.momentum,
        }
        try:
            return [values[name] for name in self._feature_order]
        except KeyError as e:
            raise ValueError(f"feature_order references an unknown feature: {e}") from e

    def predict(self, features: Features) -> float:
        torch = self._torch
        vector = self._feature_vector(features)
        with torch.no_grad():
            tensor = torch.tensor([vector], dtype=torch.float32)
            output = self._model(tensor)
            return _clip(float(output.reshape(-1)[0].item()))


def build_model(kind: str, *, imbalance_weight: float, momentum_weight: float,
                 model_path: str | None, feature_order: Sequence[str]) -> ModelWrapper:
    """Factory matching strategy.model.kind in strategy_config.toml."""
    if kind == "rule_based":
        return RuleBasedModel(imbalance_weight, momentum_weight)
    if kind == "sklearn":
        if not model_path:
            raise ValueError("strategy.model.model_path is required when kind = 'sklearn'")
        return SklearnModelWrapper(model_path, feature_order)
    if kind == "torch":
        if not model_path:
            raise ValueError("strategy.model.model_path is required when kind = 'torch'")
        return TorchModelWrapper(model_path, feature_order)
    raise ValueError(f"unknown strategy.model.kind: {kind!r} (expected 'rule_based', 'sklearn', or 'torch')")
