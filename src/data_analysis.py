"""Perform exploratory data analysis on the credit card fraud dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.txt"


def print_basic_dataset_information(data: pd.DataFrame) -> None:
    """Print shape, sample rows, column types, and missing-value counts."""
    print("Dataset Shape")
    print("-" * 40)
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}\n")

    print("First 5 Rows")
    print("-" * 40)
    print(data.head())
    print()

    print("Column Names and Data Types")
    print("-" * 40)
    print(data.dtypes.to_string())
    print()

    print("Missing Values by Column")
    print("-" * 40)
    print(data.isna().sum().to_string())
    print()


def print_class_analysis(data: pd.DataFrame) -> tuple[int, int, float, float]:
    """Print class counts and percentages."""
    class_counts = data["Class"].value_counts().sort_index()

    print("Target Distribution (Class)")
    print("-" * 40)
    print(class_counts.to_string())
    print()

    total_transactions = len(data)
    legitimate_count = int(class_counts.get(0, 0))
    fraudulent_count = int(class_counts.get(1, 0))
    legitimate_percentage = legitimate_count / total_transactions * 100
    fraudulent_percentage = fraudulent_count / total_transactions * 100

    print("Legitimate and Fraudulent Transaction Summary")
    print("-" * 40)
    print(f"Legitimate transactions (Class 0): {legitimate_count}")
    print(f"Legitimate percentage: {legitimate_percentage:.4f}%")
    print(f"Fraudulent transactions (Class 1): {fraudulent_count}")
    print(f"Fraudulent percentage: {fraudulent_percentage:.4f}%\n")

    return legitimate_count, fraudulent_count, legitimate_percentage, fraudulent_percentage


def print_amount_and_time_analysis(data: pd.DataFrame) -> None:
    """Print descriptive statistics for Amount and Time."""
    print("Descriptive Statistics for Amount")
    print("-" * 40)
    print(data["Amount"].describe().to_string())
    print()

    print("Descriptive Statistics for Time")
    print("-" * 40)
    print(data["Time"].describe().to_string())
    print()


def create_visualizations(data: pd.DataFrame) -> None:
    """Create and save the EDA plots required for this project phase."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Fraud versus legitimate transaction count.
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=data, x="Class", hue="Class", palette={0: "#2C7FB8", 1: "#D95F0E"}, legend=False)
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Overall transaction amount distribution.
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(data=data, x="Amount", bins=60, kde=False, color="#2C7FB8")
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amount_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Fraud-only transaction amount distribution.
    fraud_data = data.loc[data["Class"] == 1]
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(data=fraud_data, x="Amount", bins=40, kde=False, color="#D95F0E")
    ax.set_title("Fraud Transaction Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Fraud Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_amount_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Transaction time distribution for all classes.
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(data=data, x="Time", bins=50, hue="Class", palette={0: "#2C7FB8", 1: "#D95F0E"}, kde=False)
    ax.set_title("Transaction Time Distribution")
    ax.set_xlabel("Time (seconds from first recorded transaction)")
    ax.set_ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_eda_report(data: pd.DataFrame, counts: tuple[int, int, float, float]) -> None:
    """Save a plain-text summary of the EDA findings."""
    legitimate_count, fraudulent_count, legitimate_percentage, fraudulent_percentage = counts

    missing_values = data.isna().sum()
    duplicate_count = int(data.duplicated().sum())
    amount_stats = data["Amount"].describe()
    time_stats = data["Time"].describe()
    imbalance_ratio = legitimate_count / fraudulent_count

    lines = [
        "AI-Based Credit Card Fraud Detection - Exploratory Data Analysis",
        "=" * 72,
        "",
        "Dataset",
        "-" * 72,
        f"Source: {DATASET_PATH}",
        f"Rows: {data.shape[0]:,}",
        f"Columns: {data.shape[1]}",
        f"Duplicate records: {duplicate_count:,}",
        f"Columns containing missing values: {int((missing_values > 0).sum())}",
        "",
        "Data Types",
        "-" * 72,
        data.dtypes.to_string(),
        "",
        "Missing Values",
        "-" * 72,
        missing_values.to_string(),
        "",
        "Class Distribution",
        "-" * 72,
        f"Legitimate transactions (Class 0): {legitimate_count:,}",
        f"Legitimate percentage: {legitimate_percentage:.4f}%",
        f"Fraudulent transactions (Class 1): {fraudulent_count:,}",
        f"Fraudulent percentage: {fraudulent_percentage:.4f}%",
        f"Legitimate-to-fraud ratio: approximately {imbalance_ratio:.1f}:1",
        "",
        "Class Imbalance Analysis",
        "-" * 72,
        "Fraud is a rare event, making up only a small fraction of the dataset.",
        "This creates a major machine-learning problem because a model could achieve",
        "high accuracy by predicting every transaction as legitimate while detecting",
        "almost no fraud. Accuracy alone is therefore misleading. Recall is important",
        "because it measures how many actual fraudulent transactions are found. Precision",
        "is important because it measures how many predicted fraud cases are genuinely",
        "fraudulent. Evaluation should use precision, recall, F1-score, ROC-AUC, and",
        "confusion matrices rather than accuracy alone.",
        "",
        "Amount Statistics",
        "-" * 72,
        amount_stats.to_string(),
        "",
        "Time Statistics",
        "-" * 72,
        time_stats.to_string(),
        "Time is recorded as seconds elapsed from the first transaction in the dataset.",
        "",
        "Visualizations",
        "-" * 72,
        str(PLOTS_DIR / "class_distribution.png"),
        str(PLOTS_DIR / "amount_distribution.png"),
        str(PLOTS_DIR / "fraud_amount_distribution.png"),
        str(PLOTS_DIR / "time_distribution.png"),
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the credit card fraud exploratory data analysis."""
    data = pd.read_csv(DATASET_PATH)

    print_basic_dataset_information(data)

    counts = print_class_analysis(data)

    print("Duplicate Record Check")
    print("-" * 40)
    print(f"Duplicate records: {int(data.duplicated().sum())}\n")

    print_amount_and_time_analysis(data)

    create_visualizations(data)
    save_eda_report(data, counts)

    print("EDA Outputs")
    print("-" * 40)
    print(f"Plots saved in: {PLOTS_DIR}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
