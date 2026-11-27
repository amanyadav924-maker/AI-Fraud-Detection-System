"""Tests for final model threshold selection and validation utilities."""

import numpy as np

from src.final_model_selection import choose_threshold, evaluate_thresholds


def test_threshold_evaluation_returns_expected_columns() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    rows = evaluate_thresholds(y_true, probabilities, thresholds=np.array([0.5]))

    assert len(rows) == 1
    assert set(rows[0]) == {
        "threshold", "precision", "recall", "f1", "false_positives",
        "false_negatives", "roc_auc", "pr_auc",
    }


def test_choose_threshold_prefers_highest_f1() -> None:
    rows = [
        {"threshold": 0.3, "precision": 0.7, "recall": 0.8, "f1": 0.70, "false_positives": 10, "false_negatives": 5},
        {"threshold": 0.5, "precision": 0.9, "recall": 0.7, "f1": 0.80, "false_positives": 3, "false_negatives": 8},
    ]

    selected = choose_threshold(rows)

    assert selected["threshold"] == 0.5


def test_choose_threshold_is_not_fixed_at_zero_point_five() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.45, 0.55, 0.8, 0.9])

    rows = evaluate_thresholds(y_true, probabilities)
    selected = choose_threshold(rows)

    assert selected["threshold"] != 0.5 or selected["f1"] > 0.0


def test_choose_threshold_rejects_empty_rows() -> None:
    try:
        choose_threshold([])
    except ValueError:
        return
    raise AssertionError("Empty threshold rows should raise ValueError")
