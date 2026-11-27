"""Tests for Phase 5 imbalance-handling functions."""

import numpy as np

from src.model_optimization import (
    WeightedLogisticRegression,
    find_best_threshold,
    smote_resample,
    stratified_validation_split,
)


def test_smote_is_training_only_and_balances_fraud_count() -> None:
    features = np.vstack([
        np.array([[0.0, 1.0], [0.1, 1.1], [10.0, 10.0], [11.0, 11.0]]),
    ])
    target = np.array([1, 1, 0, 0])

    resampled_features, resampled_target = smote_resample(features, target, random_seed=42)

    assert len(resampled_features) == len(resampled_target)
    assert int(resampled_target.sum()) == int((resampled_target == 0).sum())
    assert len(features) == 4


def test_stratified_validation_split_contains_both_classes() -> None:
    features = np.arange(40).reshape(20, 2).astype(np.float64)
    target = np.array([0] * 16 + [1] * 4)

    x_train, x_validation, y_train, y_validation = stratified_validation_split(features, target)

    assert set(y_train.tolist()).issubset({0, 1})
    assert set(y_validation.tolist()) == {0, 1}
    assert len(x_train) + len(x_validation) == len(features)


def test_weighted_logistic_outputs_valid_probabilities() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    target = np.array([0, 0, 0, 1, 1, 1])

    model = WeightedLogisticRegression(fraud_weight=5.0, epochs=5).fit(features, target)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (6,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_threshold_search_returns_valid_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, metrics = find_best_threshold(y_true, probabilities)

    assert 0.05 <= threshold <= 0.95
    assert set(["precision", "recall", "f1"]).issubset(metrics)
