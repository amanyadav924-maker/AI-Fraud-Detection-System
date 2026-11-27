"""Tests for fraud-detection preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    remove_duplicate_records,
    scale_amount_and_time,
    split_features_and_target,
    split_dataset,
    validate_dataset,
)


def make_small_dataset() -> pd.DataFrame:
    """Create a small dataset containing both target classes."""
    rows = []
    for index in range(20):
        rows.append(
            {
                "Time": float(index),
                "V1": float(index),
                "Amount": float(index * 10),
                "Class": 1 if index < 4 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_validate_dataset_accepts_valid_data() -> None:
    data = make_small_dataset()

    result = validate_dataset(data)

    assert result["missing_values"] == 0
    assert result["duplicate_count"] == 0
    assert result["valid_classes"] == [0, 1]
    assert result["non_numeric_columns"] == []


def test_validate_dataset_rejects_missing_values() -> None:
    data = make_small_dataset()
    data.loc[0, "Amount"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(data)


def test_validate_dataset_rejects_invalid_class_values() -> None:
    data = make_small_dataset()
    data.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="invalid values"):
        validate_dataset(data)


def test_duplicate_removal_does_not_change_original() -> None:
    data = pd.concat([make_small_dataset(), make_small_dataset()], ignore_index=True)
    original_rows = len(data)

    cleaned = remove_duplicate_records(data)

    assert len(data) == original_rows
    assert len(cleaned) == 20


def test_features_exclude_target_column() -> None:
    data = make_small_dataset()

    features, target = split_features_and_target(data)

    assert "Class" not in features.columns
    assert target.name == "Class"
    assert list(features.columns) == ["Time", "V1", "Amount"]


def test_stratified_split_is_reproducible_and_preserves_classes() -> None:
    features, target = split_features_and_target(make_small_dataset())

    first_split = split_dataset(features, target)
    second_split = split_dataset(features, target)

    x_train_1, x_test_1, y_train_1, y_test_1 = first_split
    x_train_2, x_test_2, y_train_2, y_test_2 = second_split

    pd.testing.assert_frame_equal(x_train_1, x_train_2)
    pd.testing.assert_frame_equal(x_test_1, x_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

    assert set(y_train_1.unique()) == {0, 1}
    assert set(y_test_1.unique()) == {0, 1}


def test_scaler_uses_training_statistics_only() -> None:
    data = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    x_train = data.iloc[:4].copy()
    x_test = data.iloc[4:].copy()

    x_train_scaled, x_test_scaled, scaler = scale_amount_and_time(x_train, x_test)

    # Training data is centered and scaled, while test data uses training parameters.
    assert np.isclose(x_train_scaled["Amount"].mean(), 0.0)
    assert np.isclose(x_train_scaled["Time"].mean(), 0.0)
    assert np.allclose(scaler.mean_, x_train[["Amount", "Time"]].mean().to_numpy())

    # V1 must remain unchanged.
    assert x_train_scaled["V1"].equals(x_train["V1"])
    assert x_test_scaled["V1"].equals(x_test["V1"])
