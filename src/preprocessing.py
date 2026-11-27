"""Validate and prepare the credit card fraud dataset for model training."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "preprocessing_report.txt"
SCALER_PATH = PROCESSED_DIR / "amount_time_scaler.joblib"
RANDOM_SEED = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Amount", "Time"]


class AmountTimeScaler:
    """Standardize Amount and Time using only the training-set statistics."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "AmountTimeScaler":
        values = data[SCALE_COLUMNS].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        # Match StandardScaler: a constant column receives scale 1.0.
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform is called.")

        transformed = data.copy()
        values = transformed[SCALE_COLUMNS].to_numpy(dtype=float)
        transformed[SCALE_COLUMNS] = (values - self.mean_) / self.scale_
        return transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)


def validate_dataset(data: pd.DataFrame) -> dict:
    """Check missing values, duplicates, target values, and feature types."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")

    missing_values = data.isna().sum()
    duplicate_count = int(data.duplicated().sum())
    invalid_class_values = sorted(set(data[TARGET_COLUMN].dropna()) - {0, 1})

    non_numeric_columns = data.drop(columns=[TARGET_COLUMN]).select_dtypes(exclude="number").columns.tolist()

    if invalid_class_values:
        raise ValueError(f"Class contains invalid values: {invalid_class_values}")

    if missing_values.sum() > 0:
        raise ValueError("Dataset contains missing values. Handle these before scaling or splitting.")

    if non_numeric_columns:
        raise ValueError(f"Feature columns must be numeric. Found: {non_numeric_columns}")

    return {
        "missing_values": int(missing_values.sum()),
        "duplicate_count": duplicate_count,
        "valid_classes": sorted(data[TARGET_COLUMN].unique().tolist()),
        "non_numeric_columns": non_numeric_columns,
    }


def remove_duplicate_records(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact duplicate rows removed, leaving the source file unchanged."""
    return data.drop_duplicates().copy()


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from the target and exclude Class from features."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")

    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].copy()
    return features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible stratified training and testing sets."""
    rng = np.random.default_rng(random_seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_value in sorted(target.unique().tolist()):
        class_indices = target.index[target == class_value].to_numpy()
        rng.shuffle(class_indices)
        test_count = int(round(len(class_indices) * test_size))
        test_indices.extend(class_indices[:test_count].tolist())
        train_indices.extend(class_indices[test_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    x_train = features.loc[train_indices].copy()
    x_test = features.loc[test_indices].copy()
    y_train = target.loc[train_indices].copy()
    y_test = target.loc[test_indices].copy()
    return x_train, x_test, y_train, y_test


def scale_amount_and_time(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, AmountTimeScaler]:
    """Fit standardization only on training Amount and Time, then transform both sets."""
    scaler = AmountTimeScaler()

    # Fit only on training data. This prevents test-data statistics from leaking.
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, scaler


def save_preprocessing_outputs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: AmountTimeScaler,
) -> None:
    """Save processed train/test data and the fitted scaler outside the raw data file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    x_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    joblib.dump(scaler, SCALER_PATH)


def count_class_distribution(target: pd.Series) -> dict:
    """Return legitimate and fraudulent counts for a target split."""
    counts = target.value_counts()
    return {
        "legitimate": int(counts.get(0, 0)),
        "fraudulent": int(counts.get(1, 0)),
    }


def save_preprocessing_report(
    data_quality: dict,
    original_rows: int,
    cleaned_rows: int,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Write a plain-text summary of preprocessing decisions and results."""
    train_counts = count_class_distribution(y_train)
    test_counts = count_class_distribution(y_test)

    lines = [
        "AI-Based Credit Card Fraud Detection - Data Preprocessing Report",
        "=" * 72,
        "",
        "Data Validation",
        "-" * 72,
        f"Missing values: {data_quality['missing_values']}",
        f"Duplicate records found: {data_quality['duplicate_count']}",
        f"Validated Class values: {data_quality['valid_classes']}",
        f"Non-numeric feature columns: {data_quality['non_numeric_columns'] or 'None'}",
        "",
        "Cleaning and Raw/Processed Separation",
        "-" * 72,
        f"Rows before duplicate removal: {original_rows:,}",
        f"Rows after duplicate removal: {cleaned_rows:,}",
        f"Duplicate rows removed: {original_rows - cleaned_rows:,}",
        f"Original raw file unchanged: {DATASET_PATH}",
        f"Processed directory: {PROCESSED_DIR}",
        "",
        "Features and Target",
        "-" * 72,
        f"Target column: {TARGET_COLUMN}",
        f"Number of features: {x_train.shape[1]}",
        "Features used:",
        ", ".join(x_train.columns),
        "",
        "Train-Test Split",
        "-" * 72,
        f"Random seed: {RANDOM_SEED}",
        f"Test size: {TEST_SIZE:.0%}",
        "Split type: Stratified train-test split",
        f"Training samples: {len(x_train):,}",
        f"Training legitimate (Class 0): {train_counts['legitimate']:,}",
        f"Training fraudulent (Class 1): {train_counts['fraudulent']:,}",
        f"Testing samples: {len(x_test):,}",
        f"Testing legitimate (Class 0): {test_counts['legitimate']:,}",
        f"Testing fraudulent (Class 1): {test_counts['fraudulent']:,}",
        "",
        "Preprocessing Method",
        "-" * 72,
        "Amount and Time are standardized using zero mean and unit population deviation.",
        "The scaler is fit only on training Amount and Time values.",
        "The fitted training scaler is then applied to test Amount and Time values.",
        "No scaling is applied to the PCA-derived V1-V28 feature columns.",
        "",
        "Class Imbalance Decision",
        "-" * 72,
        "No SMOTE, oversampling, undersampling, or other balancing was applied.",
        "The original class ratio is retained in both splits.",
        "Appropriate next-phase strategies include class-weighted models,",
        "precision/recall/F1-threshold analysis, ROC-AUC, and optionally",
        "train-only resampling techniques evaluated through cross-validation.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Validate, clean, split, scale, and save the fraud dataset."""
    data = pd.read_csv(DATASET_PATH)
    original_rows = len(data)

    data_quality = validate_dataset(data)
    cleaned_data = remove_duplicate_records(data)
    cleaned_rows = len(cleaned_data)

    features, target = split_features_and_target(cleaned_data)
    x_train, x_test, y_train, y_test = split_dataset(features, target)

    x_train, x_test, scaler = scale_amount_and_time(x_train, x_test)
    save_preprocessing_outputs(x_train, x_test, y_train, y_test, scaler)
    save_preprocessing_report(
        data_quality,
        original_rows,
        cleaned_rows,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    print("Data Preprocessing Complete")
    print("-" * 40)
    print(f"Original rows: {original_rows:,}")
    print(f"Rows after duplicate removal: {cleaned_rows:,}")
    print(f"Training samples: {len(x_train):,}")
    print(f"Testing samples: {len(x_test):,}")

    train_counts = count_class_distribution(y_train)
    test_counts = count_class_distribution(y_test)
    print(f"Training Class 0 / 1: {train_counts['legitimate']:,} / {train_counts['fraudulent']:,}")
    print(f"Testing Class 0 / 1: {test_counts['legitimate']:,} / {test_counts['fraudulent']:,}")
    print(f"Features: {x_train.shape[1]}")
    print(f"Processed files saved to: {PROCESSED_DIR}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
