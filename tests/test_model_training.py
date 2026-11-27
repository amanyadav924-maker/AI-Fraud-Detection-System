"""Tests for baseline model metrics and implementations."""

import numpy as np

from src.model_training import (
    DecisionTreeBaseline,
    LogisticRegressionBaseline,
    RandomForestBaseline,
    average_precision_score,
    binary_roc_auc,
    confusion_matrix,
    classification_metrics,
)


def test_confusion_matrix_layout() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    matrix = confusion_matrix(y_true, y_pred)

    np.testing.assert_array_equal(matrix, np.array([[1, 1], [1, 1]]))


def test_perfect_classifier_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(y_true, y_pred, probabilities)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_metrics_handle_no_positive_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.zeros(4, dtype=np.int64)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = classification_metrics(y_true, y_pred, probabilities)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_roc_auc_is_1_for_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    assert binary_roc_auc(y_true, probabilities) == 1.0


def test_average_precision_is_1_for_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    assert average_precision_score(y_true, probabilities) == 1.0


def test_logistic_and_decision_tree_return_valid_probabilities() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    target = np.array([0, 0, 0, 1, 1, 1])

    logistic = LogisticRegressionBaseline(epochs=5, batch_size=3).fit(features, target)
    tree = DecisionTreeBaseline(max_depth=3, min_samples_leaf=1).fit(features, target)

    logistic_probabilities = logistic.predict_proba(features)
    tree_probabilities = tree.predict_proba(features)

    assert logistic_probabilities.shape == (6,) 
    assert np.all((logistic_probabilities >= 0.0) & (logistic_probabilities <= 1.0))
    assert np.all((tree_probabilities >= 0.0) & (tree_probabilities <= 1.0))


def test_random_forest_returns_valid_predictions() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    target = np.array([0, 0, 0, 1, 1, 1])

    forest = RandomForestBaseline(n_trees=3, max_depth=3, min_samples_leaf=1)
    forest.fit(features, target)

    predictions = forest.predict(features)

    assert predictions.shape == (6,) 
    assert set(predictions.tolist()).issubset({0, 1})
