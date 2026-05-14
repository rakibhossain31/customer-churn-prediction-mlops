"""Streamlit dashboard for churn predictions, business value, and monitoring.

Run:
    streamlit run dashboards/streamlit_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Telco Churn Decision Dashboard", layout="wide")
st.title("Telco Churn Decision Dashboard")
st.caption("Business-aware churn scoring, retention value, model metrics, and monitoring outputs")

metrics_path = ARTIFACTS / "metrics.json"
predictions_path = ARTIFACTS / "batch_predictions.csv"
feature_importance_path = ARTIFACTS / "feature_importance.json"
drift_path = ARTIFACTS / "drift_report.json"
feedback_path = ARTIFACTS / "feedback_log.csv"

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    col2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    col3.metric("F2", f"{metrics.get('f2', 0):.3f}")
    col4.metric("Retention value", f"${metrics.get('retention_value', 0):,.0f}")
else:
    st.warning("No metrics found. Train the model first with scripts/run_pipeline.py.")

if predictions_path.exists():
    scored = pd.read_csv(predictions_path)
    st.subheader("Scored customer portfolio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(scored):,}")
    c2.metric("High risk", f"{(scored['risk_level'] == 'High').sum():,}" if "risk_level" in scored else "n/a")
    c3.metric("Average churn probability", f"{scored['churn_probability'].mean():.1%}" if "churn_probability" in scored else "n/a")
    c4.metric("Expected retention profit", f"${scored.get('expected_retention_profit', pd.Series(dtype=float)).sum():,.0f}")

    if "churn_probability" in scored:
        st.plotly_chart(px.histogram(scored, x="churn_probability", nbins=30, title="Churn probability distribution"), use_container_width=True)
    if {"risk_level", "expected_retention_profit"}.issubset(scored.columns):
        grouped = scored.groupby("risk_level", as_index=False)["expected_retention_profit"].sum()
        st.plotly_chart(px.bar(grouped, x="risk_level", y="expected_retention_profit", title="Expected retention profit by risk level"), use_container_width=True)

    cols = [c for c in ["customerID", "churn_probability", "risk_level", "recommended_action", "expected_retention_profit", "estimated_revenue_at_risk", "top_reasons"] if c in scored.columns]
    st.dataframe(scored.sort_values("churn_probability", ascending=False)[cols].head(100), use_container_width=True)
else:
    st.info("No batch predictions found. Run scripts/batch_predict.py to populate this dashboard.")

left, right = st.columns(2)
with left:
    st.subheader("Model drivers")
    if feature_importance_path.exists():
        importance = pd.DataFrame(json.loads(feature_importance_path.read_text(encoding="utf-8")))
        if not importance.empty:
            st.plotly_chart(px.bar(importance.head(20), x="importance", y="feature", orientation="h", title="Top feature importances"), use_container_width=True)
        else:
            st.info("Feature importance file exists but is empty.")
    else:
        st.info("No feature importance file found yet.")

with right:
    st.subheader("Drift monitoring")
    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        st.metric("Drift detected", "Yes" if drift.get("drift_detected") else "No")
        st.write("Drifted columns:", drift.get("drifted_columns") or "None")
        st.dataframe(pd.DataFrame(drift.get("columns", [])), use_container_width=True)
    else:
        st.info("No drift report found. Run scripts/monitor_drift.py after batch scoring.")

st.subheader("Feedback loop")
if feedback_path.exists():
    feedback = pd.read_csv(feedback_path)
    st.dataframe(feedback.tail(100), use_container_width=True)
else:
    st.info("No feedback log yet. POST to /feedback or fill data/feedback/feedback_template.csv after outcomes are known.")
