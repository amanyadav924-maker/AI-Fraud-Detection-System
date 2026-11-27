"""Reusable data and report helpers for the Streamlit fraud dashboard."""

from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
FINAL_REPORT_PATH = PROJECT_ROOT / "reports" / "final_model_evaluation.txt"
DEPLOYMENT_SUMMARY_PATH = PROJECT_ROOT / "data" / "deployment" / "summary.json"
DEPLOYMENT_TEST_SAMPLES_PATH = PROJECT_ROOT / "data" / "deployment" / "test_samples.csv"


def empty_dataset_summary() -> dict:
    """Return a safe default when dashboard data cannot be loaded."""
    return {
        "total": 0,
        "legitimate": 0,
        "fraudulent": 0,
        "fraud_percentage": 0.0,
    }


def load_deployment_summary() -> dict:
    """Load small, actual aggregate statistics included for cloud deployments."""
    if not DEPLOYMENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Deployment summary not found: {DEPLOYMENT_SUMMARY_PATH}")
    return json.loads(DEPLOYMENT_SUMMARY_PATH.read_text(encoding="utf-8"))


def load_deployment_test_row(row_index: int) -> dict:
    """Load one actual held-out test sample from the small cloud-safe bundle."""
    if not DEPLOYMENT_TEST_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Deployment test samples not found: {DEPLOYMENT_TEST_SAMPLES_PATH}")

    samples = pd.read_csv(DEPLOYMENT_TEST_SAMPLES_PATH)
    if row_index < 0 or row_index >= len(samples):
        raise ValueError(f"Test row index must be between 0 and {len(samples) - 1}.")

    return samples.iloc[row_index].drop(labels=["SampleClass"]).to_dict()


def load_dataset() -> pd.DataFrame:
    """Load the public/local credit-card dataset read-only."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def dataset_summary(data: pd.DataFrame) -> dict:
    """Calculate overview counts and fraud percentage."""
    if "Class" not in data.columns:
        raise ValueError("Dataset is missing the Class column.")

    counts = data["Class"].value_counts()
    legitimate = int(counts.get(0, 0))
    fraudulent = int(counts.get(1, 0))
    total = int(len(data))
    percentage = fraudulent / total * 100 if total else 0.0

    return {
        "total": total,
        "legitimate": legitimate,
        "fraudulent": fraudulent,
        "fraud_percentage": percentage,
    }


def filter_transactions(
    data: pd.DataFrame,
    class_filter: str = "All",
    minimum_amount: float = 0.0,
    maximum_amount: float | None = None,
) -> pd.DataFrame:
    """Filter the explorer by class and inclusive Amount range."""
    filtered = data.copy()

    if "Class" not in filtered.columns:
        raise ValueError("Dataset is missing the Class column.")
    if "Amount" not in filtered.columns:
        raise ValueError("Dataset is missing the Amount column.")

    if class_filter == "Legitimate":
        filtered = filtered[filtered["Class"] == 0]
    elif class_filter == "Fraudulent":
        filtered = filtered[filtered["Class"] == 1]

    filtered = filtered[filtered["Amount"] >= minimum_amount]
    if maximum_amount is not None:
        filtered = filtered[filtered["Amount"] <= maximum_amount]

    return filtered


def parse_final_evaluation(report_path: Path = FINAL_REPORT_PATH) -> dict:
    """Parse actual metrics and confusion-matrix values from the saved Phase 6 report."""
    if not report_path.exists():
        raise FileNotFoundError(f"Final evaluation report not found: {report_path}")

    text = report_path.read_text(encoding="utf-8")

    def find_metric(label: str) -> float:
        match = re.search(rf"^{re.escape(label)}:\s*([0-9.]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Metric '{label}' was not found in the final report.")
        return float(match.group(1))

    confusion_match = re.search(r"Confusion Matrix \(TN, FP; FN, TP\): \[\[(\d+), (\d+)\], \[(\d+), (\d+)\]\]", text)
    if not confusion_match:
        raise ValueError("Confusion matrix was not found in the final report.")

    tn, fp, fn, tp = (int(value) for value in confusion_match.groups())

    return {
        "precision": find_metric("Precision"),
        "recall": find_metric("Recall"),
        "f1": find_metric("F1-score"),
        "accuracy": find_metric("Accuracy"),
        "roc_auc": find_metric("ROC-AUC"),
        "pr_auc": find_metric("Precision-Recall AUC"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_test_row(row_index: int) -> dict:
    """Load and return one held-out test row as a clearly identified test sample."""
    if not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(f"Processed test data not found: {PROCESSED_TEST_PATH}")

    test_data = pd.read_csv(PROCESSED_TEST_PATH)
    if row_index < 0 or row_index >= len(test_data):
        raise ValueError(f"Test row index must be between 0 and {len(test_data) - 1}.")

    return test_data.iloc[row_index].to_dict()


def inverse_scale_test_row(row: dict, scaler) -> dict:
    """Convert already-scaled Time and Amount test values back to engine input scale."""
    converted = dict(row)
    converted["Amount"] = float(row["Amount"] * scaler.scale_[0] + scaler.mean_[0])
    converted["Time"] = float(row["Time"] * scaler.scale_[1] + scaler.mean_[1])
    return converted
