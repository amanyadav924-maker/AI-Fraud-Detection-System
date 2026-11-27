"""Train baseline fraud classifiers and evaluate them on the held-out test set.

This module intentionally does not apply SMOTE or any other imbalance correction.
It uses NumPy and pandas so it does not depend on native scikit-learn modules.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42


class LogisticRegressionBaseline:
    """Small logistic-regression implementation using mini-batch gradient descent."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 20, batch_size: int = 1024):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LogisticRegressionBaseline":
        rng = np.random.default_rng(RANDOM_SEED)
        features = features.astype(np.float64)
        target = target.astype(np.float64)
        self.weights = np.zeros(features.shape[1], dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.epochs):
            indices = rng.permutation(len(target))
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                x_batch = features[batch]
                y_batch = target[batch]

                probabilities = self._sigmoid(x_batch @ self.weights + self.bias)
                error = probabilities - y_batch

                weight_gradient = x_batch.T @ error / len(batch)
                bias_gradient = error.mean()

                self.weights -= self.learning_rate * weight_gradient
                self.bias -= self.learning_rate * bias_gradient

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be fitted before predict_proba is called.")
        scores = features.astype(np.float64) @ self.weights + self.bias
        return self._sigmoid(scores)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)


class DecisionNode:
    """A compact node in a custom decision tree."""

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: "DecisionNode | None" = None,
        right: "DecisionNode | None" = None,
        probability: float = 0.0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.probability = probability


class DecisionTreeBaseline:
    """Compact CART-style decision tree using Gini impurity and random feature candidates."""

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_leaf: int = 50,
        max_features: int | None = None,
        random_seed: int = RANDOM_SEED,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.root: DecisionNode | None = None

    @staticmethod
    def _gini(target: np.ndarray) -> float:
        fraud_fraction = target.mean()
        return 2.0 * fraud_fraction * (1.0 - fraud_fraction)

    def _build(self, features: np.ndarray, target: np.ndarray, depth: int, rng: np.random.Generator) -> DecisionNode:
        probability = float(target.mean())

        if depth >= self.max_depth or len(target) < 2 * self.min_samples_leaf:
            return DecisionNode(probability=probability)

        if len(np.unique(target)) == 1:
            return DecisionNode(probability=probability)

        feature_count = features.shape[1]
        candidate_count = feature_count if self.max_features is None else min(self.max_features, feature_count)
        candidates = rng.choice(feature_count, candidate_count, replace=False)

        best_feature: int | None = None
        best_threshold: float | None = None
        best_loss = float("inf")

        for feature in candidates:
            column = features[:, feature]
            if np.all(column == column[0]):
                continue

            thresholds = np.quantile(column, np.linspace(0.05, 0.95, 9))
            thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left = column <= threshold
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                left_gini = self._gini(target[left])
                right_gini = self._gini(target[right])
                weighted_loss = (left.sum() * left_gini + right.sum() * right_gini) / len(target)

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        if best_feature is None:
            return DecisionNode(probability=probability)

        split = features[:, best_feature] <= best_threshold
        left_node = self._build(features[split], target[split], depth + 1, rng)
        right_node = self._build(features[~split], target[~split], depth + 1, rng)
        return DecisionNode(best_feature, best_threshold, left_node, right_node, probability)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "DecisionTreeBaseline":
        rng = np.random.default_rng(self.random_seed)
        self.root = self._build(features.astype(np.float64), target.astype(np.int64), 0, rng)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model must be fitted before predict_proba is called.")

        probabilities = np.zeros(len(features), dtype=np.float64)
        for index, row in enumerate(features):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            probabilities[index] = node.probability
        return probabilities

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)


class RandomForestBaseline:
    """Small random forest composed of bootstrap-sampled decision trees."""

    def __init__(
        self,
        n_trees: int = 12,
        max_depth: int = 8,
        min_samples_leaf: int = 50,
        random_seed: int = RANDOM_SEED,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_seed = random_seed
        self.trees: list[DecisionTreeBaseline] = []

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RandomForestBaseline":
        rng = np.random.default_rng(self.random_seed)
        feature_count = features.shape[1]
        max_features = max(1, int(np.sqrt(feature_count)))

        for tree_number in range(self.n_trees):
            bootstrap_size = min(len(target), 100_000)
            bootstrap_indices = rng.choice(len(target), bootstrap_size, replace=True)
            tree = DecisionTreeBaseline(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_seed=self.random_seed + tree_number,
            )
            tree.fit(features[bootstrap_indices], target[bootstrap_indices])
            self.trees.append(tree)

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Model must be fitted before predict_proba is called.")
        probabilities = np.vstack([tree.predict_proba(features) for tree in self.trees])
        return probabilities.mean(axis=0)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] for binary fraud labels."""
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate common binary classification metrics without scikit-learn."""
    matrix = confusion_matrix(y_true, y_pred)
    true_negatives, false_positives = int(matrix[0, 0]), int(matrix[0, 1])
    false_negatives, true_positives = int(matrix[1, 0]), int(matrix[1, 1])

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positives + true_negatives) / len(y_true)

    roc_auc = binary_roc_auc(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    return {
        "confusion_matrix": matrix,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def binary_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate ROC-AUC using the Mann-Whitney pairwise comparison."""
    positives = probabilities[y_true == 1]
    negatives = probabilities[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0

    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))


def average_precision_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate a simple average precision score from ranked probability values."""
    order = np.argsort(-probabilities)
    y_sorted = y_true[order]
    true_positives = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = true_positives / ranks
    return float(np.sum(precision * y_sorted) / np.sum(y_sorted)) if np.sum(y_sorted) else 0.0


def load_preprocessed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the existing Phase 3 train-test data without recreating the split."""
    x_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").to_numpy(dtype=np.float64)
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv").to_numpy(dtype=np.float64)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(dtype=np.int64)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(dtype=np.int64)
    return x_train, x_test, y_train, y_test


def save_confusion_plot(model_name: str, matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted Legitimate", "Predicted Fraud"])
    ax.set_yticks([0, 1], ["Actual Legitimate", "Actual Fraud"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix - {model_name}")

    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "plots" / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_comparison_plot(results: dict[str, dict]) -> None:
    names = list(results)
    metrics = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for axis, metric in zip(axes, metrics):
        values = [results[name][metric] for name in names]
        bars = axis.bar(names, values, color=["#2C7FB8", "#41AB5D", "#D95F0E"])
        axis.set_title(metric.replace("_", "-").upper())
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Baseline Model Comparison")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "plots" / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report(results: dict[str, dict], train_shape: tuple[int, int], test_shape: tuple[int, int]) -> None:
    """Save actual baseline test metrics and model-selection observations."""
    most_fraud_detected = max(results, key=lambda name: int(results[name]["confusion_matrix"][1, 1]))
    fewest_false_positives = min(results, key=lambda name: int(results[name]["confusion_matrix"][0, 1]))
    best_f1 = max(results, key=lambda name: results[name]["f1"])

    lines = [
        "AI-Based Credit Card Fraud Detection - Baseline Model Training Report",
        "=" * 72,
        "",
        "Training setup",
        "-" * 72,
        f"Training data shape: {train_shape[0]:,} rows x {train_shape[1]} features",
        f"Testing data shape: {test_shape[0]:,} rows x {test_shape[1]} features",
        f"Random seed: {RANDOM_SEED}",
        "Reused the existing Phase 3 processed train-test split.",
        "No SMOTE, oversampling, undersampling, or class balancing was applied.",
        "",
        "Evaluation Results",
        "-" * 72,
    ]

    for name, result in results.items():
        matrix = result["confusion_matrix"]
        lines.extend([
            name,
            f"  Confusion Matrix (TN, FP; FN, TP): {matrix.tolist()}",
            f"  True Negatives: {matrix[0, 0]:,}",
            f"  False Positives: {matrix[0, 1]:,}",
            f"  False Negatives: {matrix[1, 0]:,}",
            f"  True Positives: {matrix[1, 1]:,}",
            f"  Precision: {result['precision']:.6f}",
            f"  Recall: {result['recall']:.6f}",
            f"  F1-score: {result['f1']:.6f}",
            f"  Accuracy: {result['accuracy']:.6f}",
            f"  ROC-AUC: {result['roc_auc']:.6f}",
            f"  Precision-Recall AUC: {result['pr_auc']:.6f}",
            "",
        ])

    lines.extend([
        "Model Comparison",
        "-" * 72,
        f"Most fraudulent transactions detected: {most_fraud_detected}",
        f"Fewest false positives: {fewest_false_positives}",
        f"Best F1-score: {best_f1}",
        "",
        "Notes",
        "-" * 72,
        "Accuracy can be misleading because predicting every transaction as legitimate",
        "would still be highly accurate in a dataset where fraud is rare.",
        "Recall shows the fraction of actual fraud cases detected.",
        "Precision shows the fraction of fraud alerts that are genuinely fraudulent.",
        "ROC-AUC summarizes ranking across thresholds; PR-AUC is especially informative",
        "for highly imbalanced data because it focuses on the fraud-positive class.",
        "The best-F1 model is a baseline candidate for further threshold tuning and",
        "class-imbalance experiments in later phases.",
    ])

    (REPORTS_DIR / "baseline_model_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Train and evaluate all three baseline models using existing processed data."""
    x_train, x_test, y_train, y_test = load_preprocessed_data()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegressionBaseline(learning_rate=0.05, epochs=20, batch_size=1024),
        "Decision Tree": DecisionTreeBaseline(max_depth=10, min_samples_leaf=50, max_features=12),
        "Random Forest": RandomForestBaseline(n_trees=12, max_depth=8, min_samples_leaf=50),
    }

    results: dict[str, dict] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_test)
        predictions = (probabilities >= 0.5).astype(np.int64)
        result = classification_metrics(y_test, predictions, probabilities)
        results[name] = result

        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        save_confusion_plot(name, result["confusion_matrix"])

        matrix = result["confusion_matrix"]
        print(f"  TN={matrix[0, 0]}, FP={matrix[0, 1]}, FN={matrix[1, 0]}, TP={matrix[1, 1]}")
        print(f"  Precision={result['precision']:.6f}, Recall={result['recall']:.6f}, F1={result['f1']:.6f}")

    save_comparison_plot(results)
    save_report(results, x_train.shape, x_test.shape)

    print("Baseline training complete.")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Report saved to: {REPORTS_DIR / 'baseline_model_report.txt'}")


if __name__ == "__main__":
    main()
