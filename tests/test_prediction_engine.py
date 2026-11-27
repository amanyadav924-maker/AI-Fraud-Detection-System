"""Tests for the fraud prediction engine."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.prediction_engine import (
    PredictionInputError,
    PredictionModelError,
    load_final_artifact,
    predict_transaction,
    risk_category,
    validate_transaction,
)


class DummyModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, features):
        return np.array([self.probability])


class DummyScaler:
    def transform(self, frame):
        transformed = frame.copy()
        transformed[["Amount", "Time"]] = 0.0
        return transformed


def write_dummy_artifact(tmp_path: Path, probability: float) -> tuple[Path, list[str], Path]:
    feature_names = ["Time", "V1", "V2", "Amount"]
    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(DummyModel(probability), model_path)
    joblib.dump(DummyScaler(), scaler_path)

    artifact = {
        "model": DummyModel(probability),
        "decision_threshold": 0.40,
        "feature_names": feature_names,
        "preprocessing_scaler": scaler_path,
    }
    joblib.dump(artifact, model_path)
    return model_path, feature_names, scaler_path


def test_validate_transaction_accepts_mapping(tmp_path: Path) -> None:
    names = ["Time", "V1", "Amount"]
    frame = validate_transaction({"Amount": 20.0, "Time": 100.0, "V1": -1.0}, names)

    assert frame.columns.tolist() == names
    assert frame.to_numpy().tolist() == [[100.0, -1.0, 20.0]]


def test_validate_transaction_rejects_missing_feature() -> None:
    with pytest.raises(PredictionInputError, match="Missing features"):
        validate_transaction({"Time": 1.0}, ["Time", "Amount"])


def test_validate_transaction_rejects_non_numeric_values() -> None:
    with pytest.raises(PredictionInputError, match="numeric"):
        validate_transaction(["bad", 10.0], ["Time", "Amount"])


def test_validate_transaction_rejects_wrong_feature_count() -> None:
    with pytest.raises(PredictionInputError, match="Expected 2"):
        validate_transaction([10.0], ["Time", "Amount"])


def test_risk_categories_have_clear_boundaries() -> None:
    assert risk_category(19.99) == "Low"
    assert risk_category(20.0) == "Medium"
    assert risk_category(59.99) == "Medium"
    assert risk_category(60.0) == "High"


def test_valid_fraud_prediction_and_threshold(tmp_path: Path) -> None:
    model_path, _, _ = write_dummy_artifact(tmp_path, 0.72)

    result = predict_transaction({"Time": 100.0, "V1": -1.0, "V2": 2.0, "Amount": 20.0}, model_path)

    assert result["predicted_class"] == 1
    assert result["predicted_label"] == "Fraudulent"
    assert result["fraud_probability"] == 0.72
    assert result["risk_percentage"] == 72.0
    assert result["risk_category"] == "High"


def test_legitimate_prediction_below_threshold(tmp_path: Path) -> None:
    model_path, _, _ = write_dummy_artifact(tmp_path, 0.20)

    result = predict_transaction([100.0, -1.0, 2.0, 20.0], model_path)

    assert result["predicted_class"] == 0
    assert result["predicted_label"] == "Legitimate"


def test_missing_model_file_is_reported_safely(tmp_path: Path) -> None:
    with pytest.raises(PredictionModelError, match="not found"):
        load_final_artifact(tmp_path / "missing.joblib")


def test_invalid_model_artifact_is_reported_safely(tmp_path: Path) -> None:
    model_path = tmp_path / "invalid.joblib"
    joblib.dump({"model": DummyModel(0.5)}, model_path)

    with pytest.raises(PredictionModelError, match="missing keys"):
        load_final_artifact(model_path)


def test_deployed_model_artifact_uses_relative_scaler_reference() -> None:
    artifact = load_final_artifact()

    assert artifact["preprocessing_scaler"] == "data/processed/amount_time_scaler.joblib"
