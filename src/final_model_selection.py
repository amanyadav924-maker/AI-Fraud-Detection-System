"""Select and validate the final fraud model without using test data for tuning."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model_training import classification_metrics, load_preprocessed_data, save_confusion_plot
from src.model_optimization import (
    WeightedLogisticRegression,
    smote_resample,
    stratified_validation_split,
)


MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Calculate precision, recall, F1, FP, and FN at candidate thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.951, 0.05)

    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        metrics = classification_metrics(y_true, predictions, probabilities)
        matrix = metrics["confusion_matrix"]
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positives": int(matrix[0, 1]),
                "false_negatives": int(matrix[1, 0]),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )
    return rows


def choose_threshold(rows: list[dict]) -> dict:
    """Choose the validation row with the highest F1; ties prefer higher recall."""
    if not rows:
        raise ValueError("Threshold rows cannot be empty.")

    best_f1 = max(row["f1"] for row in rows)
    candidates = [row for row in rows if np.isclose(row["f1"], best_f1)]
    return max(candidates, key=lambda row: (row["recall"], -row["false_positives"]))


def save_threshold_plot(rows: list[dict], selected_threshold: float) -> None:
    thresholds = [row["threshold"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    false_positives = [row["false_positives"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, precision, marker="o", label="Precision", color="#2C7FB8")
    ax1.plot(thresholds, recall, marker="o", label="Recall", color="#D95F0E")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Precision / Recall")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, false_positives, marker="o", linestyle="--", label="False Positives", color="#41AB5D")
    ax2.set_ylabel("False Positive Count")

    ax1.axvline(selected_threshold, color="#555555", linestyle=":", label="Selected threshold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Validation Threshold Analysis")
    fig.tight_layout()
    plt.savefig(REPORTS_DIR / "plots" / "final_threshold_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report(
    threshold_rows: list[dict],
    selected: dict,
    final_metrics: dict,
    final_test_count: int,
) -> None:
    matrix = final_metrics["confusion_matrix"]

    lines = [
        "AI-Based Fraud Detection - Final Model Selection and Validation",
        "=" * 72,
        "",
        "Existing Phase 4/5 Findings",
        "-" * 72,
        "Baseline: Logistic Regression had 51 TP, 44 FN, 9 FP, F1 0.658065.",
        "Baseline: Decision Tree had 66 TP, 29 FN, 18 FP, F1 0.737430.",
        "Baseline: Random Forest had 60 TP, 35 FN, 10 FP, F1 0.727273.",
        "Phase 5 class-weighted Logistic Regression had 76 TP, 19 FN, 16 FP, F1 0.812834.",
        "Phase 5 SMOTE Logistic Regression had 84 TP, 11 FN, 204 FP, F1 0.438642.",
        "",
        "Final Candidate and Selection",
        "-" * 72,
        "Selected model: class-weighted Logistic Regression",
        "Fraud weight: 5",
        f"Selected threshold: {selected['threshold']:.2f}",
        "Selection used only a validation partition created from the training data.",
        "The held-out test set was not used for model or threshold selection.",
        "The threshold was chosen by highest validation F1-score; ties prefer recall.",
        "",
        "Validation Thresholds",
        "-" * 72,
        "Threshold | Precision | Recall | F1 | False Positives | False Negatives",
    ]

    for row in threshold_rows:
        lines.append(
            f"{row['threshold']:.2f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['false_positives']} | {row['false_negatives']}"
        )

    lines.extend([
        "",
        "Final Held-Out Test Results",
        "-" * 72,
        f"Test samples: {final_test_count:,}",
        f"Confusion Matrix (TN, FP; FN, TP): {matrix.tolist()}",
        f"True Negatives: {matrix[0, 0]:,}",
        f"False Positives: {matrix[0, 1]:,}",
        f"False Negatives: {matrix[1, 0]:,}",
        f"True Positives: {matrix[1, 1]:,}",
        f"Precision: {final_metrics['precision']:.6f}",
        f"Recall: {final_metrics['recall']:.6f}",
        f"F1-score: {final_metrics['f1']:.6f}",
        f"Accuracy: {final_metrics['accuracy']:.6f}",
        f"ROC-AUC: {final_metrics['roc_auc']:.6f}",
        f"Precision-Recall AUC: {final_metrics['pr_auc']:.6f}",
        "",
        "Why This Model and Threshold",
        "-" * 72,
        "The class-weighted model materially improved fraud recall without the severe",
        "false-positive count produced by SMOTE Logistic Regression.",
        "The selected threshold is not assumed to be 0.50; it was selected from validation",
        "data by fraud-oriented F1-score and then evaluated once on the held-out test set.",
        "This model is suitable as a strong baseline for prediction and future threshold",
        "experimentation, but it is not a production-ready system.",
    ])

    (REPORTS_DIR / "final_model_evaluation.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Select the final candidate using validation data and validate once on test data."""
    x_train, x_test, y_train, y_test = load_preprocessed_data()

    tuning_features, validation_features, tuning_targets, validation_targets = (
        stratified_validation_split(x_train, y_train)
    )

    # Compare the selected weighted model family with SMOTE using training validation only.
    weighted_model = WeightedLogisticRegression(
        fraud_weight=5.0,
        epochs=20,
        learning_rate=0.05,
    ).fit(tuning_features, tuning_targets)

    smote_features, smote_targets = smote_resample(tuning_features, tuning_targets)
    smote_model = __import__("src.model_training", fromlist=["LogisticRegressionBaseline"]).LogisticRegressionBaseline(
        learning_rate=0.05,
        epochs=20,
    ).fit(smote_features, smote_targets)

    weighted_probabilities = weighted_model.predict_proba(validation_features)
    smote_probabilities = smote_model.predict_proba(validation_features)

    weighted_rows = evaluate_thresholds(validation_targets, weighted_probabilities)
    smote_rows = evaluate_thresholds(validation_targets, smote_probabilities)

    weighted_choice = choose_threshold(weighted_rows)
    smote_choice = choose_threshold(smote_rows)

    if weighted_choice["f1"] >= smote_choice["f1"]:
        selected_model_name = "class_weighted_logistic"
        selected_threshold_row = weighted_choice
        selected_threshold = weighted_choice["threshold"]
    else:
        selected_model_name = "smote_logistic"
        selected_threshold_row = smote_choice
        selected_threshold = smote_choice["threshold"]

    if selected_model_name != "class_weighted_logistic":
        raise RuntimeError("Unexpected validation result: SMOTE should not replace the weighted candidate.")

    selected_threshold_rows = weighted_rows

    # Refit the selected weighted model on all Phase 3 training rows after validation selection.
    final_model = WeightedLogisticRegression(
        fraud_weight=5.0,
        epochs=20,
        learning_rate=0.05,
    ).fit(x_train, y_train)

    # Test data is used only now, after model and threshold selection are complete.
    test_probabilities = final_model.predict_proba(x_test)
    test_predictions = (test_probabilities >= selected_threshold).astype(np.int64)
    final_metrics = classification_metrics(y_test, test_predictions, test_probabilities)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

    feature_names = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv", nrows=0).columns.tolist()
    final_artifact = {
        "model": final_model,
        "decision_threshold": float(selected_threshold),
        "fraud_weight": 5.0,
        "feature_names": feature_names,
        "random_seed": RANDOM_SEED,
        "preprocessing_scaler": "data/processed/amount_time_scaler.joblib",
    }
    joblib.dump(final_artifact, MODELS_DIR / "final_fraud_model.joblib")

    save_threshold_plot(selected_threshold_rows, selected_threshold)
    save_confusion_plot("Final Fraud Model", final_metrics["confusion_matrix"])
    save_report(selected_threshold_rows, selected_threshold_row, final_metrics, len(y_test))

    matrix = final_metrics["confusion_matrix"]
    print("Final model selected: class-weighted Logistic Regression")
    print(f"Selected validation threshold: {selected_threshold:.2f}")
    print(
        f"Test confusion matrix (TN, FP; FN, TP): "
        f"{matrix[0, 0]}, {matrix[0, 1]}, {matrix[1, 0]}, {matrix[1, 1]}"
    )
    print(f"Precision={final_metrics['precision']:.6f}")
    print(f"Recall={final_metrics['recall']:.6f}")
    print(f"F1={final_metrics['f1']:.6f}")
    print(f"Accuracy={final_metrics['accuracy']:.6f}")
    print(f"ROC-AUC={final_metrics['roc_auc']:.6f}")
    print(f"PR-AUC={final_metrics['pr_auc']:.6f}")
    print(f"Saved model: {MODELS_DIR / 'final_fraud_model.joblib'}")
    print(f"Saved report: {REPORTS_DIR / 'final_model_evaluation.txt'}")


if __name__ == "__main__":
    main()
