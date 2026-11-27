"""Reusable prediction engine for the finalized fraud model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_fraud_model.joblib"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class PredictionInputError(ValueError):
    """Raised when transaction input cannot be converted safely."""


class PredictionModelError(RuntimeError):
    """Raised when the model artifact cannot be loaded or used."""


def risk_category(risk_percentage: float) -> str:
    """Map a 0-100 risk percentage to a clearly defined risk category."""
    if risk_percentage < 20.0:
        return "Low"
    if risk_percentage < 60.0:
        return "Medium"
    return "High"


def _load_feature_names() -> list[str]:
    """Load the exact feature order saved by the final model selection phase."""
    x_train_path = DEFAULT_PROCESSED_DIR / "X_train.csv"
    if not x_train_path.exists():
        raise PredictionModelError(f"Preprocessing output not found: {x_train_path}")
    return pd.read_csv(x_train_path, nrows=0).columns.tolist()


def load_final_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load the final model, threshold, feature names, and preprocessing metadata."""
    path = Path(model_path)
    if not path.exists():
        raise PredictionModelError(f"Final model file not found: {path}")

    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise PredictionModelError(f"Could not load the final model: {error}") from error

    required_keys = {"model", "decision_threshold", "feature_names", "preprocessing_scaler"}
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise PredictionModelError(f"Model artifact is missing keys: {sorted(missing_keys)}")

    return artifact


def validate_transaction(
    transaction: Mapping[str, object] | Sequence[object],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Validate and order a transaction exactly as the model was trained."""
    if isinstance(transaction, Mapping):
        missing = [name for name in feature_names if name not in transaction]
        if missing:
            raise PredictionInputError(f"Missing features: {', '.join(missing)}")

        unexpected = [name for name in transaction if name not in set(feature_names)]
        if unexpected:
            raise PredictionInputError(f"Unexpected features: {', '.join(unexpected)}")

        ordered_values = [transaction[name] for name in feature_names]
    else:
        if isinstance(transaction, (str, bytes)):
            raise PredictionInputError("Transaction must be a feature mapping or numeric sequence.")
        ordered_values = list(transaction)

    if len(ordered_values) != len(feature_names):
        raise PredictionInputError(
            f"Expected {len(feature_names)} feature values, received {len(ordered_values)}."
        )

    try:
        numeric_values = np.asarray(ordered_values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PredictionInputError("All feature values must be numeric.") from error

    if not np.all(np.isfinite(numeric_values)):
        raise PredictionInputError("Feature values must be finite numbers; NaN and infinity are invalid.")

    return pd.DataFrame([numeric_values], columns=list(feature_names))


def preprocess_transaction(
    transaction_frame: pd.DataFrame,
    preprocessing_scaler,
) -> pd.DataFrame:
    """Apply the exact Amount and Time standardization used during training."""
    required_columns = {"Amount", "Time"}
    if not required_columns.issubset(transaction_frame.columns):
        raise PredictionInputError("Processed transaction data must contain Amount and Time.")

    try:
        transformed = preprocessing_scaler.transform(transaction_frame)
    except Exception as error:
        raise PredictionModelError(f"Could not preprocess transaction: {error}") from error

    return transformed


def predict_transaction(
    transaction: Mapping[str, object] | Sequence[object],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Validate, preprocess, and classify one transaction without retraining."""
    artifact = load_final_artifact(model_path)
    model = artifact["model"]

    threshold = float(artifact["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise PredictionModelError("The saved decision threshold must be between 0 and 1.")

    feature_names = list(artifact["feature_names"])
    transaction_frame = validate_transaction(transaction, feature_names)

    scaler_reference = artifact["preprocessing_scaler"]
    if isinstance(scaler_reference, Path):
        scaler_path = scaler_reference
    else:
        scaler_path = Path(str(scaler_reference))

    if not scaler_path.is_absolute():
        scaler_path = PROJECT_ROOT / scaler_path

    if not scaler_path.exists():
        raise PredictionModelError(f"Preprocessing scaler not found: {scaler_path}")

    try:
        preprocessing_scaler = joblib.load(scaler_path)
    except Exception as error:
        raise PredictionModelError(f"Could not load preprocessing scaler: {error}") from error

    processed_frame = preprocess_transaction(transaction_frame, preprocessing_scaler)

    try:
        probability = float(model.predict_proba(processed_frame.to_numpy())[0])
    except Exception as error:
        raise PredictionModelError(f"The loaded model could not produce a prediction: {error}") from error

    probability = float(np.clip(probability, 0.0, 1.0))
    predicted_class = 1 if probability >= threshold else 0
    risk_percentage = probability * 100.0

    if predicted_class == 1:
        label = "Fraudulent"
        explanation = (
            f"The estimated fraud probability is {risk_percentage:.2f}% and is at or above "
            f"the selected decision threshold of {threshold:.2f}. This transaction is flagged for review."
        )
    else:
        label = "Legitimate"
        explanation = (
            f"The estimated fraud probability is {risk_percentage:.2f}% and is below "
            f"the selected decision threshold of {threshold:.2f}. The model does not flag this transaction."
        )

    return {
        "predicted_class": predicted_class,
        "predicted_label": label,
        "fraud_probability": probability,
        "risk_percentage": risk_percentage,
        "risk_category": risk_category(risk_percentage),
        "decision_threshold": threshold,
        "explanation": explanation,
    }
