"""Interactive Streamlit dashboard for the AI fraud detection system."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
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


st.set_page_config(
    page_title="Fraud Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #07111f 0%, #0b1728 100%); color: #e7edf5; }
section[data-testid="stSidebar"] { background: #091425; border-right: 1px solid #1b2a40; }
[data-testid="stMetric"] { background: #111f32; border: 1px solid #1d2d44; border-radius: 14px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color: #9fb0c4; }
[data-testid="stMetricValue"] { color: #f7fafc; }
.stButton > button { border-radius: 9px; border: 1px solid #263b57; background: #13243a; color: #f7fafc; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #ef4444, #dc2626); border: none; }
div[data-testid="stDataFrame"] { border: 1px solid #1d2d44; border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_dashboard_data():
    return load_dataset()


@st.cache_resource
def get_model_artifact():
    return load_final_artifact()


def histogram_counts(values: pd.Series, bins: int = 30) -> pd.Series:
    """Convert numeric values into Streamlit-compatible bar-chart bins."""
    counts, edges = pd.cut(values, bins=bins, include_lowest=True, retbins=True)
    labels = [f"{edges[index]:.0f}-{edges[index + 1]:.0f}" for index in range(len(edges) - 1)]
    result = counts.value_counts(sort=False)
    return pd.Series(result.to_numpy(), index=labels, name="Count")


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"### {title}")
    st.caption(subtitle)
    st.divider()


def show_overview(data: pd.DataFrame | None, deployment_summary: dict | None = None) -> None:
    page_header("Executive Overview", "Transaction population and finalized model at a glance")
    summary = dataset_summary(data) if data is not None else {
        "total": deployment_summary["total"],
        "legitimate": deployment_summary["legitimate"],
        "fraudulent": deployment_summary["fraudulent"],
        "fraud_percentage": deployment_summary["fraud_percentage"],
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{summary['total']:,}")
    col2.metric("Legitimate", f"{summary['legitimate']:,}")
    col3.metric("Fraudulent", f"{summary['fraudulent']:,}", delta=f"{summary['fraud_percentage']:.4f}%", delta_color="inverse")
    col4.metric("Fraud Percentage", f"{summary['fraud_percentage']:.4f}%")

    left, right = st.columns(2)
    class_chart = pd.Series(
        {"Legitimate": summary["legitimate"], "Fraudulent": summary["fraudulent"]},
        name="Transactions",
    )
    left.subheader("Fraud vs Legitimate")
    left.bar_chart(class_chart, color="#ef4444")

    right.subheader("Transaction Amount Distribution")
    if data is not None:
        amount_series = data["Amount"].clip(upper=data["Amount"].quantile(0.99))
        right.bar_chart(histogram_counts(amount_series, 25))
    else:
        histogram = deployment_summary["amount_histogram"]
        edges = histogram["edges"]
        labels = [f"{edges[index]:.0f}-{edges[index + 1]:.0f}" for index in range(len(edges) - 1)]
        right.bar_chart(pd.Series(histogram["counts"], index=labels))


def show_analytics(data: pd.DataFrame | None, deployment_summary: dict | None = None) -> None:
    page_header("Fraud Analytics", "Distribution and pattern analysis from the original dataset")
    summary = dataset_summary(data) if data is not None else {
        "total": deployment_summary["total"],
        "legitimate": deployment_summary["legitimate"],
        "fraudulent": deployment_summary["fraudulent"],
        "fraud_percentage": deployment_summary["fraud_percentage"],
    }

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Rate", f"{summary['fraud_percentage']:.4f}%")
    m2.metric("Fraud-to-Legitimate Ratio", f"{summary['legitimate'] / summary['fraudulent']:.1f}:1")
    median_amount = data["Amount"].median() if data is not None else deployment_summary["amount_median"]
    m3.metric("Median Amount", f"{median_amount:,.2f}")

    chart1, chart2 = st.columns(2)
    chart1.subheader("Transaction Counts")
    chart1.bar_chart(pd.Series({"Legitimate": summary["legitimate"], "Fraudulent": summary["fraudulent"]}))

    chart2.subheader("Fraudulent Transaction Amounts")
    if data is not None:
        fraud_amount = data.loc[data["Class"] == 1, "Amount"]
        chart2.bar_chart(histogram_counts(fraud_amount.clip(upper=fraud_amount.quantile(0.99)), 25))
    else:
        histogram = deployment_summary["fraud_amount_histogram"]
        edges = histogram["edges"]
        labels = [f"{edges[index]:.0f}-{edges[index + 1]:.0f}" for index in range(len(edges) - 1)]
        chart2.bar_chart(pd.Series(histogram["counts"], index=labels))

    chart3, chart4 = st.columns(2)
    chart3.subheader("Amount by Class (Median)")
    if data is not None:
        chart3.bar_chart(data.groupby("Class")["Amount"].median().rename(index={0: "Legitimate", 1: "Fraudulent"}))
    else:
        chart3.metric("Fraud Median Amount", f"{deployment_summary['fraud_amount_median']:,.2f}")

    chart4.subheader("Time Distribution — Fraud Rate by Hour")
    if data is not None:
        time_sample = data[["Time", "Class"]].sort_values("Time").copy()
        time_sample["Hour"] = (time_sample["Time"] / 3600).astype(int)
        chart4.line_chart(time_sample.groupby("Hour")["Class"].mean().rename("Fraud Rate"))
    else:
        hourly = deployment_summary["hourly_fraud_rates"]
        chart4.line_chart(pd.Series({int(hour): rate for hour, rate in hourly.items()}, name="Fraud Rate"))


def show_prediction(data: pd.DataFrame | None = None, deployment_summary: dict | None = None) -> None:
    page_header("Transaction Prediction", "Enter a transaction using model-compatible feature values")

    try:
        artifact = get_model_artifact()
    except (PredictionModelError, PredictionInputError) as error:
        st.error(f"Model could not be loaded: {error}")
        return

    feature_names = list(artifact["feature_names"])
    threshold = float(artifact["decision_threshold"])

    st.info(f"Final model: **Class-weighted Logistic Regression** · Decision threshold: **{threshold:.2f}**")

    test_option = st.toggle("Load an existing held-out test row")
    default_values = {}

    if test_option:
        try:
            max_test_row = 56745 if data is not None else 49
            row_index = st.number_input("Held-out test row index", min_value=0, max_value=max_test_row, value=0, step=1)
            if data is not None:
                processed_row = load_test_row(int(row_index))
            else:
                processed_row = load_deployment_test_row(int(row_index))
            scaler_path = PROJECT_ROOT / "data" / "processed" / "amount_time_scaler.joblib"
            scaler = joblib.load(scaler_path)
            default_values = inverse_scale_test_row(processed_row, scaler) if data is not None else processed_row
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

    values = {}
    st.subheader("Feature Values")
    top_left, top_right = st.columns(2)

    for index, name in enumerate(feature_names):
        container = top_left if index % 2 == 0 else top_right
        default = float(default_values.get(name, 0.0))
        values[name] = container.number_input(name, value=default, format="%.6f", key=f"feature_{name}")

    if st.button("Run Fraud Prediction", type="primary", width="stretch"):
        try:
            result = predict_transaction(values)
        except (PredictionInputError, PredictionModelError) as error:
            st.error(str(error))
            return

        result_card, gauge_card = st.columns([1.4, 1])
        risk = result["risk_percentage"]
        predicted_label = result["predicted_label"]

        if predicted_label == "Fraudulent":
            result_card.error(f"### FRAUD ALERT — {predicted_label}")
        else:
            result_card.success(f"### TRANSACTION CHECK — {predicted_label}")

        result_card.write(result["explanation"])
        result_card.metric("Risk Percentage", f"{risk:.2f}%")
        gauge_card.metric("Risk Category", result["risk_category"])
        gauge_card.metric("Fraud Probability", f"{result['fraud_probability']:.6f}")
        gauge_card.metric("Decision Threshold", f"{threshold:.2f}")
        gauge_card.progress(min(max(risk / 100.0, 0.0), 1.0))


def show_performance() -> None:
    page_header("Model Performance", "Actual Phase 6 held-out test results")

    try:
        metrics = parse_final_evaluation()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision", f"{metrics['precision']:.4f}")
    c2.metric("Recall", f"{metrics['recall']:.4f}")
    c3.metric("F1-Score", f"{metrics['f1']:.4f}")
    c4.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c6.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")

    left, right = st.columns([1.2, 1])
    left.subheader("Confusion Matrix")
    matrix = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    )
    left.dataframe(matrix, width="stretch")

    right.subheader("Error Counts")
    right.metric("False Positives", f"{metrics['fp']:,}")
    right.metric("False Negatives", f"{metrics['fn']:,}")


def show_explorer(data: pd.DataFrame) -> None:
    page_header("Data Explorer", "Filter and inspect the public transaction dataset")

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
    col3.metric("Visible Mean Amount", f"{filtered['Amount'].mean():,.2f}")

    search_text = st.text_input("Search row value", help="Filters matching text across every visible column")
    if search_text:
        searchable = filtered.fillna("").astype(str)
        mask = searchable.apply(lambda column: column.str.contains(search_text, case=False, regex=False)).any(axis=1)
        filtered = filtered.loc[mask]

    st.dataframe(filtered.head(500), width="stretch", height=480)


# Sidebar navigation
st.sidebar.markdown("## FRAUD // INTEL")
st.sidebar.caption("AI-powered transaction security")
st.sidebar.divider()
page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Fraud Analytics", "Transaction Prediction", "Model Performance", "Data Explorer"],
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

if page == "Overview":
    show_overview(dashboard_data, deployment_summary)
elif page == "Fraud Analytics":
    show_analytics(dashboard_data, deployment_summary)
elif page == "Transaction Prediction":
    show_prediction(dashboard_data, deployment_summary)
elif page == "Model Performance":
    show_performance()
else:
    if dashboard_data is None:
        st.warning("The full data explorer is unavailable in cloud mode because the raw public dataset is not committed. Aggregate dashboard views remain available.")
    else:
        show_explorer(dashboard_data)
