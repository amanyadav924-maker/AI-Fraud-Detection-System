"""Interactive Streamlit dashboard for the AI fraud detection system."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import (
    dataset_summary,
    filter_transactions,
    inverse_scale_test_row,
    load_dataset,
    load_deployment_summary,
    load_deployment_test_row,
    load_test_row,
    parse_final_evaluation,
)
from src.prediction_engine import (
    PredictionInputError,
    PredictionModelError,
    load_final_artifact,
    predict_transaction,
)
from src.ui_components import apply_dashboard_styles, kpi_card, page_header


st.set_page_config(
    page_title="FRAUD // INTEL",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dashboard_styles()


@st.cache_data(ttl=3600)
def get_dashboard_data():
    return load_dataset()


@st.cache_resource
def get_model_artifact():
    return load_final_artifact()


def histogram_counts(values: pd.Series, bins: int = 30) -> pd.DataFrame:
    """Return finite histogram counts with numeric bucket midpoints."""
    numeric_values = values.to_numpy(dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return pd.DataFrame(
            {"Amount band midpoint": pd.Series(dtype=float), "Transactions": pd.Series(dtype=int)}
        )

    categories, edges = pd.cut(
        numeric_values,
        bins=bins,
        include_lowest=True,
        retbins=True,
    )
    counts = np.bincount(categories.codes, minlength=bins)
    midpoints = np.round((edges[:-1] + edges[1:]) / 2, 2)
    return pd.DataFrame(
        {"Amount band midpoint": midpoints, "Transactions": counts}
    )


def histogram_from_aggregates(histogram: dict) -> pd.DataFrame:
    """Plot existing deployment histogram values on their numeric bucket midpoints."""
    edges = np.asarray(histogram["edges"], dtype=float)
    counts = np.asarray(histogram["counts"], dtype=int)
    if len(edges) != len(counts) + 1:
        raise ValueError("Histogram edges must be one longer than its count values.")
    midpoints = np.round((edges[:-1] + edges[1:]) / 2, 2)
    return pd.DataFrame(
        {"Amount band midpoint": midpoints, "Transactions": counts}
    )


def show_overview(data: pd.DataFrame | None, deployment_summary: dict | None = None) -> None:
    page_header(
        "Executive overview",
        "A live view of transaction volume, observed fraud, and value distribution.",
        "REAL-TIME RISK INTELLIGENCE",
    )
    summary = dataset_summary(data) if data is not None else {
        "total": deployment_summary["total"],
        "legitimate": deployment_summary["legitimate"],
        "fraudulent": deployment_summary["fraudulent"],
        "fraud_percentage": deployment_summary["fraud_percentage"],
    }

    st.caption(
        "SOURCE · Full transaction dataset"
        if data is not None
        else "SOURCE · Cloud-safe aggregates from the original dataset"
    )

    metrics = st.columns(4)
    with metrics[0]:
        kpi_card("Total transactions", f"{summary['total']:,}", "◈", "cyan", "Observed records")
    with metrics[1]:
        kpi_card("Legitimate", f"{summary['legitimate']:,}", "✓", "green", "Observed Class 0")
    with metrics[2]:
        kpi_card("Fraudulent", f"{summary['fraudulent']:,}", "!", "red", "Observed Class 1")
    with metrics[3]:
        kpi_card(
            "Fraud rate",
            f"{summary['fraud_percentage']:.4f}%",
            "↗",
            "blue",
            "Share of all transactions",
        )

    left, right = st.columns(2)
    with left.container(border=True):
        left.subheader("Observed transaction classes")
        class_chart = pd.DataFrame(
            {
                "Legitimate": [summary["legitimate"]],
                "Fraudulent": [summary["fraudulent"]],
            },
            index=["Transactions"],
        )
        left.bar_chart(class_chart, color=["#41d6a1", "#ff607c"])
        left.caption("Class labels in the original dataset")

    with right.container(border=True):
        right.subheader("Transaction amount distribution")
        if data is not None:
            amount_series = data["Amount"].clip(upper=data["Amount"].quantile(0.99))
            right.bar_chart(
                histogram_counts(amount_series, 25),
                x="Amount band midpoint",
                y="Transactions",
                color="#45d9f3",
            )
            right.caption("Amounts above the 99th percentile are clipped for readability.")
        else:
            right.bar_chart(
                histogram_from_aggregates(deployment_summary["amount_histogram"]),
                x="Amount band midpoint",
                y="Transactions",
                color="#45d9f3",
            )
            right.caption("Cloud-safe amount aggregates from the original dataset.")


def show_analytics(data: pd.DataFrame | None, deployment_summary: dict | None = None) -> None:
    page_header(
        "Fraud analytics",
        "Explore observed class distribution, transaction values, and temporal patterns.",
        "PATTERN ANALYSIS",
    )
    summary = dataset_summary(data) if data is not None else {
        "total": deployment_summary["total"],
        "legitimate": deployment_summary["legitimate"],
        "fraudulent": deployment_summary["fraudulent"],
        "fraud_percentage": deployment_summary["fraud_percentage"],
    }

    metrics = st.columns(3)
    with metrics[0]:
        kpi_card("Observed fraud rate", f"{summary['fraud_percentage']:.4f}%", "↗", "red")
    with metrics[1]:
        kpi_card(
            "Legitimate : fraud",
            f"{summary['legitimate'] / max(summary['fraudulent'], 1):,.1f}:1",
            "⇄",
            "blue",
        )
    median_amount = data["Amount"].median() if data is not None else deployment_summary["amount_median"]
    with metrics[2]:
        kpi_card("Median amount", f"${median_amount:,.2f}", "$", "cyan")

    cohort_options = ["All transactions", "Fraudulent"]
    if data is not None:
        cohort_options.insert(1, "Legitimate")
    cohort = st.selectbox("Amount distribution cohort", cohort_options)

    chart1, chart2 = st.columns(2)
    with chart1.container(border=True):
        chart1.subheader("Fraud vs. legitimate")
        class_chart = pd.DataFrame(
            {
                "Legitimate": [summary["legitimate"]],
                "Fraudulent": [summary["fraudulent"]],
            },
            index=["Transactions"],
        )
        chart1.bar_chart(class_chart, color=["#41d6a1", "#ff607c"])
        chart1.caption("Actual class counts from the selected data source.")

    with chart2.container(border=True):
        chart2.subheader(f"{cohort} · amount distribution")
        if data is not None:
            class_code = {
                "Legitimate": 0,
                "Fraudulent": 1,
            }.get(cohort)
            cohort_data = (
                data if class_code is None else data.loc[data["Class"] == class_code]
            )
            if cohort_data.empty:
                chart2.info("No transactions match this cohort.")
            else:
                amounts = cohort_data["Amount"]
                amounts = amounts.clip(upper=amounts.quantile(0.99))
                chart2.bar_chart(
                    histogram_counts(amounts, 25),
                    x="Amount band midpoint",
                    y="Transactions",
                    color="#45d9f3",
                )
        else:
            histogram_key = (
                "fraud_amount_histogram"
                if cohort == "Fraudulent"
                else "amount_histogram"
            )
            chart2.bar_chart(
                histogram_from_aggregates(deployment_summary[histogram_key]),
                x="Amount band midpoint",
                y="Transactions",
                color="#45d9f3",
            )
        chart2.caption("The upper 1% is clipped for a readable distribution.")

    chart3, chart4 = st.columns(2)
    with chart3.container(border=True):
        chart3.subheader("Median amount by class")
        if data is not None:
            median_by_class = data.groupby("Class")["Amount"].median().rename(
                index={0: "Legitimate", 1: "Fraudulent"}
            )
            chart3.bar_chart(median_by_class, color="#45d9f3")
        else:
            chart3.metric(
                "Fraud median amount",
                f"${deployment_summary['fraud_amount_median']:,.2f}",
            )
            chart3.caption("Aggregate from the original dataset.")

    with chart4.container(border=True):
        chart4.subheader("Fraud rate by time bucket")
        if data is not None:
            time_sample = data[["Time", "Class"]].sort_values("Time").copy()
            time_sample["Hour"] = (time_sample["Time"] / 3600).astype(int)
            hourly_fraud_rate = (
                time_sample.groupby("Hour")["Class"]
                .mean()
                .rename("Fraud rate")
                .reset_index()
            )
            chart4.line_chart(
                hourly_fraud_rate,
                x="Hour",
                y="Fraud rate",
                color="#ff607c",
            )
        else:
            hourly = deployment_summary["hourly_fraud_rates"]
            hourly_fraud_rate = pd.Series(
                {int(hour): rate for hour, rate in hourly.items()},
                name="Fraud rate",
            ).rename_axis("Hour").reset_index()
            chart4.line_chart(
                hourly_fraud_rate,
                x="Hour",
                y="Fraud rate",
                color="#ff607c",
            )
        chart4.caption("Rates are derived from actual class labels, not model estimates.")


def show_prediction(data: pd.DataFrame | None = None, deployment_summary: dict | None = None) -> None:
    page_header(
        "Transaction prediction",
        "Submit a transaction against the saved model and review its risk estimate.",
        "MODEL-POWERED SCREENING",
    )

    try:
        artifact = get_model_artifact()
    except (PredictionModelError, PredictionInputError) as error:
        st.error(f"Model could not be loaded: {error}")
        return

    feature_names = list(artifact["feature_names"])
    threshold = float(artifact["decision_threshold"])

    st.caption(
        f"ACTIVE MODEL · Class-weighted Logistic Regression · "
        f"Decision threshold {threshold:.2f}"
    )

    test_option = st.toggle("Load an existing held-out test row")
    default_values = {}

    if test_option:
        try:
            max_test_row = 56745 if data is not None else 49
            row_index = st.number_input(
                "Held-out test row index",
                min_value=0,
                max_value=max_test_row,
                value=0,
                step=1,
            )
            if data is not None:
                processed_row = load_test_row(int(row_index))
            else:
                processed_row = load_deployment_test_row(int(row_index))
            scaler_path = PROJECT_ROOT / "data" / "processed" / "amount_time_scaler.joblib"
            scaler = joblib.load(scaler_path)
            default_values = (
                inverse_scale_test_row(processed_row, scaler)
                if data is not None
                else processed_row
            )
            st.caption("This value comes from held-out test data and is not an actual banking transaction.")
        except Exception as error:
            st.error(f"Could not load test row: {error}")

    if not default_values:
        try:
            if data is not None:
                defaults = data[feature_names].median(numeric_only=True).to_dict()
                defaults["Amount"] = float(data["Amount"].median())
                defaults["Time"] = float(data["Time"].median())
            else:
                defaults = {name: 0.0 for name in feature_names}
                defaults["Amount"] = float(deployment_summary["amount_median"])
                defaults["Time"] = 0.0
            default_values = defaults
        except Exception:
            default_values = {name: 0.0 for name in feature_names}

    with st.container(border=True):
        st.subheader("Transaction details")
        st.caption(
            "Enter original-scale values. Transaction amount must be non-negative; "
            "latent features retain the model's saved feature order."
        )
        values = {}
        with st.form("transaction_prediction_form"):
            context_left, context_right = st.columns(2)
            for name, container in (("Amount", context_left), ("Time", context_right)):
                if name in feature_names:
                    value = float(default_values.get(name, 0.0))
                    values[name] = container.number_input(
                        name,
                        value=max(value, 0.0) if name == "Amount" else value,
                        min_value=0.0 if name == "Amount" else None,
                        format="%.2f" if name == "Amount" else "%.2f",
                        help=(
                            "Amount in the original dataset's currency units."
                            if name == "Amount"
                            else "Elapsed transaction time in seconds."
                        ),
                        key=f"feature_{name}",
                    )

            remaining_features = [
                name for name in feature_names if name not in {"Amount", "Time"}
            ]
            if remaining_features:
                with st.expander(
                    f"Model features · {len(remaining_features)} fields",
                    expanded=False,
                ):
                    st.caption(
                        "These anonymized features are required by the trained model. "
                        "Values are not inferred from cardholder identity."
                    )
                    feature_columns = st.columns(2)
                    for index, name in enumerate(remaining_features):
                        default = float(default_values.get(name, 0.0))
                        values[name] = feature_columns[index % 2].number_input(
                            name,
                            value=default,
                            format="%.5f",
                            key=f"feature_{name}",
                        )

            submitted = st.form_submit_button(
                "Analyze transaction",
                type="primary",
                width="stretch",
            )

    if submitted:
        try:
            with st.spinner("Scoring transaction with the saved model..."):
                result = predict_transaction(values)
        except (PredictionInputError, PredictionModelError) as error:
            st.error(f"Analysis could not be completed: {error}")
            return

        risk = result["risk_percentage"]
        predicted_label = result["predicted_label"]

        st.markdown('<div class="eyebrow">MODEL ASSESSMENT</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1.4, 1])
        with result_left.container(border=True):
            if predicted_label == "Fraudulent":
                st.error(f"FRAUD ALERT · {predicted_label}")
            else:
                st.success(f"SCREENING RESULT · {predicted_label}")
            st.write(result["explanation"])
            st.progress(min(max(risk / 100.0, 0.0), 1.0), text="Fraud probability")
        with result_right.container(border=True):
            st.metric("Risk category", result["risk_category"])
            st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
            st.metric("Decision threshold", f"{threshold:.2f}")


def show_performance() -> None:
    page_header(
        "Model performance",
        "Evaluation metrics from the saved final-model report and held-out test set.",
        "MODEL ASSURANCE",
    )

    try:
        metrics = parse_final_evaluation()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    metric_rows = [
        [
            ("Precision", metrics["precision"], "◉", "cyan"),
            ("Recall", metrics["recall"], "↗", "blue"),
            ("F1-score", metrics["f1"], "◆", "green"),
        ],
        [
            ("Accuracy", metrics["accuracy"], "✓", "green"),
            ("ROC-AUC", metrics["roc_auc"], "⌁", "cyan"),
            ("PR-AUC", metrics["pr_auc"], "⌖", "blue"),
        ],
    ]
    for metric_row in metric_rows:
        columns = st.columns(3)
        for column, (label, value, icon, tone) in zip(columns, metric_row):
            with column:
                kpi_card(label, f"{value:.4f}", icon, tone, "Held-out evaluation")

    left, right = st.columns([1.2, 1])
    with left.container(border=True):
        left.subheader("Confusion matrix")
        matrix = pd.DataFrame(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"],
        )
        left.dataframe(matrix, width="stretch")
        left.caption("Counts are from the finalized held-out evaluation.")

    with right.container(border=True):
        right.subheader("Classification profile")
        right.bar_chart(
            pd.Series(
                {
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-score": metrics["f1"],
                    "ROC-AUC": metrics["roc_auc"],
                    "PR-AUC": metrics["pr_auc"],
                },
                name="Score",
            ),
            color="#45d9f3",
        )
        right.caption("Metric values are sourced from the saved evaluation report.")
    error_columns = st.columns(2)
    with error_columns[0]:
        st.metric("False positives", f"{metrics['fp']:,}")
    with error_columns[1]:
        st.metric("False negatives", f"{metrics['fn']:,}")


def show_explorer(data: pd.DataFrame) -> None:
    page_header(
        "Data explorer",
        "Search, filter, page through, and export the original transaction records.",
        "TRANSACTION INVESTIGATION",
    )

    class_filter = st.selectbox("Transaction class", ["All", "Legitimate", "Fraudulent"])
    minimum_amount = float(data["Amount"].min())
    maximum_amount = float(data["Amount"].max())

    amount_range = st.slider(
        "Transaction amount range",
        min_value=minimum_amount,
        max_value=maximum_amount,
        value=(minimum_amount, maximum_amount),
    )

    filtered = filter_transactions(data, class_filter, amount_range[0], amount_range[1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Visible Transactions", f"{len(filtered):,}")
    col2.metric("Visible Fraud", f"{int((filtered['Class'] == 1).sum()):,}")
    mean_amount = filtered["Amount"].mean()
    col3.metric(
        "Visible Mean Amount",
        f"{mean_amount:,.2f}" if pd.notna(mean_amount) else "—",
    )

    search_text = st.text_input("Search row value", help="Filters matching text across every visible column")
    if search_text:
        searchable = filtered.fillna("").astype(str)
        mask = searchable.apply(lambda column: column.str.contains(search_text, case=False, regex=False)).any(axis=1)
        filtered = filtered.loc[mask]

    st.caption(f"{len(filtered):,} records match the current filters.")
    page_size = st.selectbox("Rows per page", [25, 50, 100], index=1)
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    if "explorer_page" not in st.session_state:
        st.session_state["explorer_page"] = 1
    current_page = min(max(int(st.session_state["explorer_page"]), 1), page_count)
    st.session_state["explorer_page"] = current_page

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    if previous_column.button("← Previous", disabled=current_page <= 1):
        st.session_state["explorer_page"] = current_page - 1
        st.rerun()
    page_column.markdown(
        f'<div class="page-count">PAGE {current_page:,} OF {page_count:,}</div>',
        unsafe_allow_html=True,
    )
    if next_column.button("Next →", disabled=current_page >= page_count):
        st.session_state["explorer_page"] = current_page + 1
        st.rerun()

    start = (current_page - 1) * page_size
    visible_rows = filtered.iloc[start : start + page_size]
    st.dataframe(visible_rows, width="stretch", height=480)
    export_signature = (
        class_filter,
        amount_range[0],
        amount_range[1],
        search_text,
    )
    if st.button("Prepare filtered CSV", key="prepare_transaction_export"):
        st.session_state["explorer_export"] = {
            "signature": export_signature,
            "data": filtered.to_csv(index=False).encode("utf-8"),
        }

    export = st.session_state.get("explorer_export")
    if export and export["signature"] == export_signature:
        st.download_button(
            "Download filtered transactions",
            data=export["data"],
            file_name="filtered_transactions.csv",
            mime="text/csv",
            width="stretch",
        )
    elif export:
        st.session_state.pop("explorer_export", None)


def show_software_testing_center() -> None:
    from testing_center import (
        CATEGORY_MARKERS,
        TestExecutionError,
        run_pytest_suite,
        to_csv_report,
        to_json_report,
    )

    page_header(
        "Software testing center",
        "Run the repository's pytest suite and inspect verified execution results.",
        "QUALITY ASSURANCE",
    )
    st.caption(
        "Results are read from pytest-generated JUnit XML. No case outcomes are pre-filled."
    )

    selected_category = st.selectbox("Test category", list(CATEGORY_MARKERS))
    if st.button("Run pytest tests", type="primary", width="stretch"):
        try:
            with st.spinner(f"Running {selected_category.lower()}..."):
                st.session_state["software_testing_result"] = run_pytest_suite(
                    selected_category
                )
        except TestExecutionError as error:
            st.session_state.pop("software_testing_result", None)
            st.error(f"Test execution could not be completed: {error}")

    result = st.session_state.get("software_testing_result")
    if result is None:
        st.info("Choose a test category and run pytest to populate this report.")
        return
    if result.selected_category != selected_category:
        st.info(
            f"The displayed report is for {result.selected_category}. "
            "Run the selected category to refresh it."
        )
        return

    if result.exit_code == 0:
        st.success(f"{result.selected_category} completed successfully.")
    else:
        st.error(
            f"{result.selected_category} completed with pytest exit code "
            f"{result.exit_code}."
        )

    total = len(result.cases)
    summary_columns = st.columns(5)
    summary_columns[0].metric("Test Cases", f"{total:,}")
    summary_columns[1].metric("Passed", f"{result.passed:,}")
    summary_columns[2].metric("Failed", f"{result.failed:,}")
    summary_columns[3].metric("Skipped", f"{result.skipped:,}")
    summary_columns[4].metric("Execution Time", f"{result.duration_seconds:.2f} s")

    st.caption(f"Pytest exit code · {result.exit_code}")

    if result.cases:
        category_names = sorted({case.category for case in result.cases})
        category_breakdown = pd.DataFrame(
            {
                "Passed": [
                    sum(
                        case.category == category and case.status == "PASS"
                        for case in result.cases
                    )
                    for category in category_names
                ],
                "Failed": [
                    sum(
                        case.category == category and case.status == "FAIL"
                        for case in result.cases
                    )
                    for category in category_names
                ],
                "Skipped": [
                    sum(
                        case.category == category and case.status == "SKIP"
                        for case in result.cases
                    )
                    for category in category_names
                ],
            },
            index=category_names,
        )
        with st.container(border=True):
            st.subheader("Execution by category")
            st.bar_chart(
                category_breakdown,
                color=["#41d6a1", "#ff607c", "#91a5bd"],
            )

        results_table = pd.DataFrame(
            [
                {
                    "Test Case ID": case.case_id,
                    "Test Description": case.description,
                    "Expected Result": case.expected_result,
                    "Actual Result": case.actual_result,
                    "Status": case.status,
                    "Duration (s)": round(case.duration_seconds, 4),
                }
                for case in result.cases
            ]
        )
        st.dataframe(results_table, width="stretch", hide_index=True)

        for case in result.cases:
            if case.status == "FAIL":
                with st.expander(f"{case.case_id}: {case.description} — failure details"):
                    st.code(case.failure_details or case.actual_result)
    else:
        st.warning("Pytest produced no individual test-case results.")

    if result.stdout or result.stderr:
        with st.expander("Pytest execution output"):
            if result.stdout:
                st.text(result.stdout)
            if result.stderr:
                st.code(result.stderr)

    csv_column, json_column, junit_column = st.columns(3)
    csv_column.download_button(
        "Download CSV report",
        data=to_csv_report(result),
        file_name="software-testing-report.csv",
        mime="text/csv",
        width="stretch",
    )
    json_column.download_button(
        "Download JSON report",
        data=to_json_report(result),
        file_name="software-testing-report.json",
        mime="application/json",
        width="stretch",
    )
    junit_column.download_button(
        "Download JUnit XML",
        data=result.junit_xml,
        file_name="software-testing-results.xml",
        mime="application/xml",
        width="stretch",
    )


# Sidebar navigation
st.sidebar.markdown(
    (
        '<div class="brand-lockup">'
        '<div class="brand-mark" aria-hidden="true">FI</div>'
        '<div><div class="brand-name">FRAUD // INTEL</div>'
        '<div class="brand-subtitle">Financial crime intelligence</div></div>'
        "</div>"
    ),
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="nav-section-label">INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
navigation_icons = {
    "Overview": "◈",
    "Fraud Analytics": "⌁",
    "Transaction Prediction": "◇",
    "Model Performance": "▥",
    "Data Explorer": "⌕",
    "Software Testing Center": "✓",
}
navigation_pages = list(navigation_icons)
page = st.sidebar.radio(
    "Workspace",
    navigation_pages,
    format_func=lambda label: f"{navigation_icons[label]}  {label}",
    label_visibility="collapsed",
)

dashboard_data = None
deployment_summary = None
try:
    dashboard_data = get_dashboard_data()
except (FileNotFoundError, ValueError):
    try:
        deployment_summary = load_deployment_summary()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.stop()

data_mode = "LOCAL DATASET" if dashboard_data is not None else "CLOUD AGGREGATES"
st.sidebar.markdown(
    (
        '<div class="sidebar-status">'
        '<div class="status-line"><span class="status-dot"></span>'
        "DASHBOARD ONLINE</div>"
        f'<p class="status-detail">Data mode · {data_mode}</p>'
        "</div>"
    ),
    unsafe_allow_html=True,
)
if page != "Data Explorer":
    st.session_state.pop("explorer_export", None)

if page == "Overview":
    show_overview(dashboard_data, deployment_summary)
elif page == "Fraud Analytics":
    show_analytics(dashboard_data, deployment_summary)
elif page == "Transaction Prediction":
    show_prediction(dashboard_data, deployment_summary)
elif page == "Model Performance":
    show_performance()
elif page == "Software Testing Center":
    show_software_testing_center()
else:
    if dashboard_data is None:
        st.warning("The full data explorer is unavailable in cloud mode because the raw public dataset is not committed. Aggregate dashboard views remain available.")
    else:
        show_explorer(dashboard_data)
