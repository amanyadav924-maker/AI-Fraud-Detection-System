"""Tests for the fraud prediction engine."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.dashboard import load_deployment_test_row
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


@pytest.mark.unit
@pytest.mark.case(
    case_id="TC-UNIT-001",
    description="Order named transaction features to match the model schema",
    expected_result="The returned row follows the requested feature order.",
)
def test_validate_transaction_accepts_mapping(tmp_path: Path) -> None:
    names = ["Time", "V1", "Amount"]
    frame = validate_transaction({"Amount": 20.0, "Time": 100.0, "V1": -1.0}, names)

    assert frame.columns.tolist() == names
    assert frame.to_numpy().tolist() == [[100.0, -1.0, 20.0]]


@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-MISSING",
    description="Reject a transaction that omits a required feature",
    expected_result="PredictionInputError identifies the missing feature.",
)
def test_validate_transaction_rejects_missing_feature() -> None:
    with pytest.raises(PredictionInputError, match="Missing features"):
        validate_transaction({"Time": 1.0}, ["Time", "Amount"])


@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-NONNUMERIC",
    description="Reject non-numeric transaction values",
    expected_result="PredictionInputError reports that all feature values must be numeric.",
)
def test_validate_transaction_rejects_non_numeric_values() -> None:
    with pytest.raises(PredictionInputError, match="numeric"):
        validate_transaction(["bad", 10.0], ["Time", "Amount"])


@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-DIMENSION",
    description="Reject input with the wrong number of features",
    expected_result="PredictionInputError reports the expected and received dimensions.",
)
def test_validate_transaction_rejects_wrong_feature_count() -> None:
    with pytest.raises(PredictionInputError, match="Expected 2"):
        validate_transaction([10.0], ["Time", "Amount"])


@pytest.mark.parametrize(
    "amount",
    [-0.01, -100.0, np.nan, np.inf, "not-a-number"],
    ids=[
        "TC-BOUNDARY-AMOUNT-NEGATIVE-FRACTION",
        "TC-BOUNDARY-AMOUNT-NEGATIVE-LARGE",
        "TC-BOUNDARY-AMOUNT-NAN",
        "TC-BOUNDARY-AMOUNT-INFINITY",
        "TC-BOUNDARY-AMOUNT-NONNUMERIC",
    ],
)
@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-AMOUNT",
    description="Reject negative, non-finite, and non-numeric transaction amounts",
    expected_result="Each invalid Amount raises PredictionInputError.",
)
def test_validate_transaction_rejects_invalid_amounts(amount: object) -> None:
    with pytest.raises(PredictionInputError):
        validate_transaction({"Time": 100.0, "Amount": amount}, ["Time", "Amount"])


@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-AMOUNT-ZERO",
    description="Accept a zero transaction amount at the valid boundary",
    expected_result="Amount=0 is retained as a finite numeric feature.",
)
def test_validate_transaction_accepts_zero_amount() -> None:
    frame = validate_transaction(
        {"Time": 100.0, "Amount": 0.0},
        ["Time", "Amount"],
    )

    assert frame["Amount"].iloc[0] == 0.0


@pytest.mark.boundary
@pytest.mark.case(
    case_id="TC-BOUNDARY-RISK",
    description="Classify risk at low, medium, and high category boundaries",
    expected_result="Risk categories change at 20% and 60%.",
)
def test_risk_categories_have_clear_boundaries() -> None:
    assert risk_category(19.99) == "Low"
    assert risk_category(20.0) == "Medium"
    assert risk_category(59.99) == "Medium"
    assert risk_category(60.0) == "High"


@pytest.mark.functional
@pytest.mark.case(
    case_id="TC-FUNCTIONAL-FRAUD",
    description="Flag a transaction whose probability exceeds the decision threshold",
    expected_result="The transaction is labeled Fraudulent with the expected risk details.",
)
def test_valid_fraud_prediction_and_threshold(tmp_path: Path) -> None:
    model_path, _, _ = write_dummy_artifact(tmp_path, 0.72)

    result = predict_transaction({"Time": 100.0, "V1": -1.0, "V2": 2.0, "Amount": 20.0}, model_path)

    assert result["predicted_class"] == 1
    assert result["predicted_label"] == "Fraudulent"
    assert result["fraud_probability"] == 0.72
    assert result["risk_percentage"] == 72.0
    assert result["risk_category"] == "High"


@pytest.mark.functional
@pytest.mark.case(
    case_id="TC-FUNCTIONAL-LEGITIMATE",
    description="Leave a transaction below the decision threshold unflagged",
    expected_result="The transaction is labeled Legitimate.",
)
def test_legitimate_prediction_below_threshold(tmp_path: Path) -> None:
    model_path, _, _ = write_dummy_artifact(tmp_path, 0.20)

    result = predict_transaction([100.0, -1.0, 2.0, 20.0], model_path)

    assert result["predicted_class"] == 0
    assert result["predicted_label"] == "Legitimate"


@pytest.mark.integration
@pytest.mark.case(
    case_id="TC-INTEGRATION-MODEL-MISSING",
    description="Report a missing model artifact",
    expected_result="PredictionModelError identifies the missing model file.",
)
def test_missing_model_file_is_reported_safely(tmp_path: Path) -> None:
    with pytest.raises(PredictionModelError, match="not found"):
        load_final_artifact(tmp_path / "missing.joblib")


@pytest.mark.integration
@pytest.mark.case(
    case_id="TC-INTEGRATION-MODEL-INVALID",
    description="Reject an artifact missing required model metadata",
    expected_result="PredictionModelError identifies missing artifact keys.",
)
def test_invalid_model_artifact_is_reported_safely(tmp_path: Path) -> None:
    model_path = tmp_path / "invalid.joblib"
    joblib.dump({"model": DummyModel(0.5)}, model_path)

    with pytest.raises(PredictionModelError, match="missing keys"):
        load_final_artifact(model_path)


@pytest.mark.integration
@pytest.mark.case(
    case_id="TC-INTEGRATION-DEPLOYED-PREDICTION",
    description="Load the saved model and scaler and predict a deployment sample",
    expected_result="Prediction returns a valid class, probability, risk category, and explanation.",
)
def test_saved_model_predicts_deployment_sample() -> None:
    artifact = load_final_artifact()
    sample = load_deployment_test_row(0)

    result = predict_transaction(sample)

    assert result["predicted_label"] in {"Fraudulent", "Legitimate"}
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert result["explanation"]
    assert len(artifact["feature_names"]) == len(sample)


@pytest.mark.integration
@pytest.mark.case(
    case_id="TC-INTEGRATION-PORTABLE-ARTIFACT",
    description="Check the deployed artifact uses its portable scaler reference",
    expected_result="The scaler is referenced by a project-relative POSIX path.",
)
def test_deployed_model_artifact_uses_relative_scaler_reference() -> None:
    artifact = load_final_artifact()

    assert artifact["preprocessing_scaler"] == "data/processed/amount_time_scaler.joblib"
