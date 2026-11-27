"""Simple command-line interface for the final fraud prediction engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction_engine import (
    PredictionInputError,
    PredictionModelError,
    predict_transaction,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud for one transaction.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--json", help="Path to a JSON object containing all 30 feature values.")
    input_group.add_argument("--test-row", type=int, help="Row index from data/processed/X_test.csv for a clearly labeled test example.")

    parser.add_argument("--model", default="models/final_fraud_model.joblib", help="Path to the final model artifact.")
    return parser.parse_args()


def get_test_row(row_index: int) -> dict:
    test_path = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
    if not test_path.exists():
        raise PredictionInputError(f"Test data not found: {test_path}")

    test_rows = pd.read_csv(test_path)
    if row_index < 0 or row_index >= len(test_rows):
        raise PredictionInputError(
            f"Test row index must be between 0 and {len(test_rows) - 1}; received {row_index}."
        )

    # Values in X_test.csv are already processed. Convert Amount and Time back to
    # their original scale so the prediction engine can apply the same preprocessing.
    row = test_rows.iloc[row_index].to_dict()
    scaler_path = PROJECT_ROOT / "data" / "processed" / "amount_time_scaler.joblib"
    if not scaler_path.exists():
        raise PredictionModelError("The preprocessing scaler required by test-row mode is missing.")

    scaler = joblib.load(scaler_path)
    row["Amount"] = float(row["Amount"] * scaler.scale_[0] + scaler.mean_[0])
    row["Time"] = float(row["Time"] * scaler.scale_[1] + scaler.mean_[1])

    return {name: float(value) for name, value in row.items()}


def load_json_transaction(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise PredictionInputError(f"JSON transaction file not found: {path}")

    try:
        transaction = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PredictionInputError(f"Invalid JSON: {error}") from error

    if not isinstance(transaction, dict):
        raise PredictionInputError("The JSON root must be an object containing feature names and values.")

    return transaction


def main() -> None:
    args = parse_arguments()

    try:
        if args.test_row is not None:
            print("Using a row from the held-out test data. This is test data, not an actual banking transaction.")
            transaction = get_test_row(args.test_row)
            preprocessed_test_values = True
        else:
            transaction = load_json_transaction(args.json)
            preprocessed_test_values = False

        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        result = predict_transaction(transaction, model_path=model_path)

        print("Fraud Prediction Result")
        print("-" * 40)
        print(f"Predicted class: {result['predicted_label']} (Class {result['predicted_class']})")
        print(f"Fraud probability: {result['fraud_probability']:.6f}")
        print(f"Risk percentage: {result['risk_percentage']:.2f}%")
        print(f"Risk category: {result['risk_category']}")
        print(f"Decision threshold: {result['decision_threshold']:.2f}")
        print(f"Explanation: {result['explanation']}")

        if preprocessed_test_values:
            print("Note: This value came from already processed test data; it was not an actual transaction.")

    except (PredictionInputError, PredictionModelError) as error:
        print(f"Prediction error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
