"""Experiment with class weights and SMOTE using the existing Phase 3/4 data."""

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

from src.model_training import (
    LogisticRegressionBaseline,
    classification_metrics,
    load_preprocessed_data,
    save_confusion_plot,
)


MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42


class WeightedLogisticRegression(LogisticRegressionBaseline):
    """Logistic regression that applies a configurable weight to fraud samples."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 20,
        batch_size: int = 1024,
        fraud_weight: float = 1.0,
    ):
        super().__init__(learning_rate, epochs, batch_size)
        self.fraud_weight = fraud_weight

    def fit(self, features: np.ndarray, target: np.ndarray) -> "WeightedLogisticRegression":
        rng = np.random.default_rng(RANDOM_SEED)
        features = features.astype(np.float64)
        target = target.astype(np.float64)
        self.weights = np.zeros(features.shape[1], dtype=np.float64)
        self.bias = 0.0
        sample_weights = np.where(target == 1.0, self.fraud_weight, 1.0)

        for _ in range(self.epochs):
            indices = rng.permutation(len(target))
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                x_batch = features[batch]
                y_batch = target[batch]
                weight_batch = sample_weights[batch]

                probabilities = self._sigmoid(x_batch @ self.weights + self.bias)
                error = (probabilities - y_batch) * weight_batch

                weight_gradient = x_batch.T @ error / weight_batch.sum()
                bias_gradient = error.sum() / weight_batch.sum()

                self.weights -= self.learning_rate * weight_gradient
                self.bias -= self.learning_rate * bias_gradient

        return self


def smote_resample(
    features: np.ndarray,
    target: np.ndarray,
    random_seed: int = RANDOM_SEED,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fraud examples by linearly interpolating each fraud sample with neighbors.

    SMOTE is applied only when this function receives training data. It never receives test data.
    """
    rng = np.random.default_rng(random_seed)
    fraud_features = features[target == 1]
    legitimate_features = features[target == 0]

    if len(fraud_features) == 0:
        return features.copy(), target.copy()

    desired_fraud_count = len(legitimate_features)
    generated_count = desired_fraud_count - len(fraud_features)

    if generated_count <= 0:
        return features.copy(), target.copy()

    fraud_features = fraud_features.astype(np.float64)
    difference = fraud_features[:, None, :] - fraud_features[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=2))
    np.fill_diagonal(distances, np.inf)

    neighbor_count = min(k_neighbors, len(fraud_features) - 1)
    if neighbor_count <= 0:
        neighbor_count = 1

    nearest_indices = np.argsort(distances, axis=1)[:, :neighbor_count]
    synthetic_features = np.empty((generated_count, features.shape[1]), dtype=np.float64)

    for index in range(generated_count):
        base_index = index % len(fraud_features)
        neighbor_number = rng.integers(0, nearest_indices.shape[1])
        neighbor_index = nearest_indices[base_index, neighbor_number]
        fraction = rng.random()
        synthetic_features[index] = fraud_features[base_index] + fraction * (
            fraud_features[neighbor_index] - fraud_features[base_index]
        )

    combined_features = np.vstack([features, synthetic_features])
    synthetic_targets = np.ones(generated_count, dtype=np.int64)
    combined_targets = np.concatenate([target, synthetic_targets])

    shuffle_indices = rng.permutation(len(combined_targets))
    return combined_features[shuffle_indices], combined_targets[shuffle_indices]


def stratified_validation_split(
    features: np.ndarray,
    target: np.ndarray,
    validation_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a validation subset from training data only, preserving both classes."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_value in (0, 1):
        indices = np.flatnonzero(target == class_value)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        features[np.array(train_indices)],
        features[np.array(validation_indices)],
        target[np.array(train_indices)],
        target[np.array(validation_indices)],
    )


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict]:
    """Choose a threshold using only validation F1-score."""
    candidate_thresholds = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_metrics: dict | None = None

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = classification_metrics(y_true, predictions, probabilities)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("No threshold was evaluated.")
    return best_threshold, best_metrics


def load_baseline_results() -> dict:
    """Read the baseline report as a short summary rather than retraining baseline models."""
    report_path = REPORTS_DIR / "baseline_model_report.txt"
    if not report_path.exists():
        raise FileNotFoundError("Missing Phase 4 report: reports/baseline_model_report.txt")

    return {
        "Logistic Regression (baseline)": report_path,
        "Decision Tree (baseline)": report_path,
        "Random Forest (baseline)": report_path,
    }


def save_comparison_plot(results: dict[str, dict]) -> None:
    names = list(results)
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for axis, metric in zip(axes, metrics):
        values = [results[name][metric] for name in names]
        bars = axis.bar(range(len(names)), values, color="#2C7FB8")
        axis.set_xticks(range(len(names)), [name.replace(" ", "\n") for name in names], fontsize=8)
        axis.set_title(metric.upper().replace("_", "-"))
        axis.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Phase 5 Fraud Detection Experiments")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "plots" / "phase5_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report(results: dict[str, dict], thresholds: dict[str, float]) -> None:
    """Write the final held-out-test comparison and experiment decisions."""
    best_f1 = max(results, key=lambda name: results[name]["f1"])
    best_pr_auc = max(results, key=lambda name: results[name]["pr_auc"])
    fewest_false_negatives = min(results, key=lambda name: int(results[name]["confusion_matrix"][1, 0]))
    fewest_false_positives = min(results, key=lambda name: int(results[name]["confusion_matrix"][0, 1]))

    lines = [
        "AI-Based Fraud Detection - Phase 5 Imbalance Handling and Optimization",
        "=" * 72,
        "",
        "Experimental setup",
        "-" * 72,
        f"Random seed: {RANDOM_SEED}",
        "The original Phase 3/4 held-out test data was preserved and not resampled.",
        "Validation was created only from the training partition.",
        "Hyperparameters and thresholds were selected using training validation data,",
        "not by repeatedly evaluating the final test set.",
        "",
        "Baseline reference",
        "-" * 72,
        "Baseline Phase 4 results: Logistic Regression 51 TP / 44 FN / 9 FP,",
        "Decision Tree 66 TP / 29 FN / 18 FP, Random Forest 60 TP / 35 FN / 10 FP.",
        "",
        "Experiment Results",
        "-" * 72,
    ]

    for name, result in results.items():
        matrix = result["confusion_matrix"]
        lines.extend([
            name,
            f"  Threshold: {thresholds.get(name, 0.5):.2f}",
            f"  Confusion Matrix (TN, FP; FN, TP): {matrix.tolist()}",
            f"  False Positives: {matrix[0, 1]:,}",
            f"  False Negatives: {matrix[1, 0]:,}",
            f"  Precision: {result['precision']:.6f}",
            f"  Recall: {result['recall']:.6f}",
            f"  F1-score: {result['f1']:.6f}",
            f"  Accuracy: {result['accuracy']:.6f}",
            f"  ROC-AUC: {result['roc_auc']:.6f}",
            f"  Precision-Recall AUC: {result['pr_auc']:.6f}",
            "",
        ])

    lines.extend([
        "Model Selection",
        "-" * 72,
        f"Best F1-score: {best_f1}",
        f"Best PR-AUC: {best_pr_auc}",
        f"Fewest false negatives: {fewest_false_negatives}",
        f"Fewest false positives: {fewest_false_positives}",
        "",
        "Trade-off",
        "-" * 72,
        "Lower thresholds typically detect more fraud but create more false positives.",
        "Higher thresholds reduce false alerts but increase missed fraud cases.",
        "The selected model is chosen primarily by F1-score and PR-AUC, while also",
        "reviewing false negatives because missed fraud can be costly.",
    ])

    (REPORTS_DIR / "phase5_experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run class-weight and training-only SMOTE experiments, then preserve the test set."""
    x_train, x_test, y_train, y_test = load_preprocessed_data()

    # Validation is derived only from the training partition.
    train_for_tuning, validation_features, train_for_tuning_targets, validation_targets = (
        stratified_validation_split(x_train, y_train)
    )

    weight_candidates = [5.0, 10.0, 20.0]
    weighted_validation_results: list[tuple[float, float, WeightedLogisticRegression]] = []

    for fraud_weight in weight_candidates:
        model = WeightedLogisticRegression(fraud_weight=fraud_weight, epochs=20).fit(
            train_for_tuning, train_for_tuning_targets
        )
        probabilities = model.predict_proba(validation_features)
        threshold, metrics = find_best_threshold(validation_targets, probabilities)
        weighted_validation_results.append((metrics["f1"], threshold, model))

    best_weighted_f1, best_weighted_threshold, best_weighted_model_template = max(
        weighted_validation_results, key=lambda item: item[0]
    )

    # Refit class-weighted logistic regression on all training rows after validation selection.
    fraud_weight = best_weighted_model_template.fraud_weight
    weighted_model = WeightedLogisticRegression(fraud_weight=fraud_weight, epochs=20).fit(x_train, y_train)

    smote_features, smote_targets = smote_resample(x_train, y_train)
    smote_model = LogisticRegressionBaseline(learning_rate=0.05, epochs=20).fit(smote_features, smote_targets)

    smote_validation_model = LogisticRegressionBaseline(learning_rate=0.05, epochs=20).fit(
        *smote_resample(train_for_tuning, train_for_tuning_targets)
    )
    smote_validation_probabilities = smote_validation_model.predict_proba(validation_features)
    smote_threshold, _ = find_best_threshold(validation_targets, smote_validation_probabilities)

    results: dict[str, dict] = {}
    thresholds: dict[str, float] = {}

    weighted_probabilities = weighted_model.predict_proba(x_test)
    weighted_predictions = (weighted_probabilities >= best_weighted_threshold).astype(np.int64)
    results["Class-weighted Logistic Regression"] = classification_metrics(
        y_test, weighted_predictions, weighted_probabilities
    )
    thresholds["Class-weighted Logistic Regression"] = best_weighted_threshold

    smote_probabilities = smote_model.predict_proba(x_test)
    smote_predictions = (smote_probabilities >= smote_threshold).astype(np.int64)
    results["SMOTE Logistic Regression"] = classification_metrics(y_test, smote_predictions, smote_probabilities)
    thresholds["SMOTE Logistic Regression"] = smote_threshold

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

    joblib.dump(weighted_model, MODELS_DIR / "class_weighted_logistic_phase5.joblib")
    joblib.dump(smote_model, MODELS_DIR / "smote_logistic_phase5.joblib")

    for name, result in results.items():
        save_confusion_plot(name, result["confusion_matrix"])

    save_comparison_plot(results)
    save_report(results, thresholds)

    print("Phase 5 experiments complete.")
    print(f"Weighted-logistic fraud weight: {fraud_weight}")
    print(f"Weighted-logistic validation threshold: {best_weighted_threshold:.2f}")
    print(f"SMOTE validation threshold: {smote_threshold:.2f}")
    print(f"SMOTE training shape: {smote_features.shape[0]:,} rows x {smote_features.shape[1]} features")

    for name, result in results.items():
        matrix = result["confusion_matrix"]
        print(
            f"{name}: TN={matrix[0, 0]}, FP={matrix[0, 1]}, "
            f"FN={matrix[1, 0]}, TP={matrix[1, 1]}, F1={result['f1']:.6f}, "
            f"PR-AUC={result['pr_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
