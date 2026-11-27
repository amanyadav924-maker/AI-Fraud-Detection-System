"""Tests for dashboard data helpers and report parsing."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import (
    dataset_summary,
    filter_transactions,
    load_deployment_summary,
    load_deployment_test_row,
    parse_final_evaluation,
)


def make_dashboard_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [1.0, 2.0, 3.0, 4.0],
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "Class": [0, 1, 0, 1],
        }
    )


def test_dataset_summary_is_calculated() -> None:
    summary = dataset_summary(make_dashboard_data())

    assert summary["total"] == 4
    assert summary["legitimate"] == 2
    assert summary["fraudulent"] == 2
    assert summary["fraud_percentage"] == 50.0


def test_filter_transactions_by_class_and_amount() -> None:
    data = make_dashboard_data()

    filtered = filter_transactions(data, "Fraudulent", 15.0, 35.0)

    assert filtered["Class"].tolist() == [1]
    assert filtered["Amount"].tolist() == [20.0]


def test_filter_transactions_accepts_all() -> None:
    filtered = filter_transactions(make_dashboard_data(), "All", 20.0, 40.0)

    assert len(filtered) == 3


def test_deployment_summary_loads_actual_totals() -> None:
    summary = load_deployment_summary()

    assert summary["total"] == 284807
    assert summary["legitimate"] == 284315
    assert summary["fraudulent"] == 492


def test_deployment_test_row_loads_without_raw_dataset() -> None:
    row = load_deployment_test_row(0)

    assert "Time" in row
    assert "Amount" in row
    assert "SampleClass" not in row


@pytest.mark.system
@pytest.mark.case(
    case_id="TC-SYSTEM-NAVIGATION",
    description="Open the Software Testing Center and execute a pytest category",
    expected_result="The center displays actual category results without exceptions.",
)
def test_streamlit_app_loads_without_exceptions() -> None:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    app = AppTest.from_file(app_path, default_timeout=60)
    app.run()

    assert not app.exception
    pages = [
        "Overview",
        "Fraud Analytics",
        "Transaction Prediction",
        "Model Performance",
        "Data Explorer",
        "Software Testing Center",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception

    app.sidebar.radio[0].set_value("Transaction Prediction").run()
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any("MODEL ASSESSMENT" in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Software Testing Center").run()
    assert not app.exception
    assert any("Choose a test category" in item.value for item in app.info)
    app.button[0].click().run()

    assert not app.exception
    assert any(
        metric.label == "Test Cases" and int(metric.value) > 0
        for metric in app.metric
    )
    assert any("completed successfully" in item.value for item in app.success)


def test_final_report_parser_reads_actual_metrics(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "Precision: 0.826087\n"
        "Recall: 0.800000\n"
        "F1-score: 0.812834\n"
        "Accuracy: 0.999383\n"
        "ROC-AUC: 0.988116\n"
        "Precision-Recall AUC: 0.774204\n"
        "Confusion Matrix (TN, FP; FN, TP): [[56635, 16], [19, 76]]\n",
        encoding="utf-8",
    )

    metrics = parse_final_evaluation(report)

    assert metrics["precision"] == 0.826087
    assert metrics["fp"] == 16
    assert metrics["fn"] == 19
