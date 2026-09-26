from decimal import Decimal

import pytest

from strategy.features import Features
from strategy.models import RuleBasedModel, build_model


def make_features(imbalance: float = 0.0, momentum: float = 0.0, **bar_features) -> Features:
    return Features(
        symbol="BTC-USD",
        mid_price=Decimal(100),
        spread=Decimal("0.5"),
        imbalance=imbalance,
        momentum=momentum,
        **bar_features,
    )


def test_rule_based_model_positive_imbalance_gives_positive_signal():
    model = RuleBasedModel(imbalance_weight=1.0, momentum_weight=0.0)
    signal = model.predict(make_features(imbalance=0.5))
    assert signal == pytest.approx(0.5)


def test_rule_based_model_negative_imbalance_gives_negative_signal():
    model = RuleBasedModel(imbalance_weight=1.0, momentum_weight=0.0)
    signal = model.predict(make_features(imbalance=-0.5))
    assert signal == pytest.approx(-0.5)


def test_rule_based_model_clips_to_valid_range():
    model = RuleBasedModel(imbalance_weight=1.0, momentum_weight=1.0)
    signal = model.predict(make_features(imbalance=1.0, momentum=1.0))
    assert -1.0 <= signal <= 1.0
    assert signal == 1.0


def test_rule_based_model_combines_both_weighted():
    model = RuleBasedModel(imbalance_weight=0.6, momentum_weight=0.4)
    # momentum scaled by 100 inside the model, then clipped to [-1, 1]
    signal = model.predict(make_features(imbalance=0.5, momentum=0.01))
    expected = 0.6 * 0.5 + 0.4 * 1.0  # 0.01 * 100 = 1.0, already at the clip boundary
    assert signal == pytest.approx(min(expected, 1.0))


def test_build_model_rule_based():
    model = build_model("rule_based", imbalance_weight=0.5, momentum_weight=0.5, model_path=None, feature_order=())
    assert isinstance(model, RuleBasedModel)


def test_build_model_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_kind", imbalance_weight=0.5, momentum_weight=0.5, model_path=None, feature_order=())


def test_build_model_sklearn_requires_model_path():
    with pytest.raises(ValueError):
        build_model("sklearn", imbalance_weight=0.5, momentum_weight=0.5, model_path=None, feature_order=("imbalance",))


def test_sklearn_wrapper_against_a_toy_model(tmp_path):
    """Proves SklearnModelWrapper's plumbing works — feature ordering,
    joblib loading, predict_proba mapping to [-1, 1] — against a toy
    classifier trained on nothing meaningful. This is NOT a validated
    trading model; see models.py's module docstring."""
    sklearn = pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array([[-1.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    toy_model = LogisticRegression().fit(X, y)

    model_path = tmp_path / "toy.joblib"
    joblib.dump(toy_model, model_path)

    from strategy.models import SklearnModelWrapper

    wrapper = SklearnModelWrapper(str(model_path), feature_order=["imbalance", "momentum"])
    signal = wrapper.predict(make_features(imbalance=1.0, momentum=1.0))
    assert -1.0 <= signal <= 1.0
    # Trained so positive features -> class 1 -> should lean positive.
    assert signal > 0


def test_sklearn_wrapper_feature_vector_includes_bar_derived_features():
    """_feature_vector must expose the new bar-derived feature names too,
    so a feature_order referencing them (once a real model is trained on
    historical Kraken candles) doesn't hit the "unknown feature" KeyError
    path."""
    from strategy.models import SklearnModelWrapper

    class _FakeModel:
        def predict(self, vector):
            return [0.0]

    wrapper = SklearnModelWrapper.__new__(SklearnModelWrapper)
    wrapper._model = _FakeModel()
    wrapper._feature_order = ["sma_ratio", "rsi", "realized_vol", "bar_momentum"]

    features = make_features(sma_ratio=0.1, rsi=0.2, realized_vol=0.3, bar_momentum=0.4)
    assert wrapper._feature_vector(features) == [0.1, 0.2, 0.3, 0.4]


def test_sklearn_wrapper_feature_vector_includes_ema_bollinger_ao():
    """Same as above, for the newer indicators (EMA, Bollinger Bands,
    Awesome Oscillator)."""
    from strategy.models import SklearnModelWrapper

    class _FakeModel:
        def predict(self, vector):
            return [0.0]

    wrapper = SklearnModelWrapper.__new__(SklearnModelWrapper)
    wrapper._model = _FakeModel()
    wrapper._feature_order = ["ema_ratio", "bollinger_percent_b", "bollinger_bandwidth", "awesome_oscillator"]

    features = make_features(
        ema_ratio=0.5, bollinger_percent_b=0.6, bollinger_bandwidth=0.7, awesome_oscillator=0.8
    )
    assert wrapper._feature_vector(features) == [0.5, 0.6, 0.7, 0.8]
