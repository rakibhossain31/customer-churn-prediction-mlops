"""FastAPI and Gradio application for Telco churn prediction."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import gradio as gr
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.config import get_nested, load_config
from src.database.sqlite_store import insert_prediction, read_predictions
from src.explainability.explainer import global_feature_importance, local_shap_explanation
from src.feedback.store import append_feedback
from src.monitoring.drift import drift_report
from src.serving.inference import load_model_bundle, model_status, predict

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]
CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_COUNTS = defaultdict(int)
REQUEST_LATENCY_SECONDS = defaultdict(float)
PREDICTION_COUNT = 0


class CustomerData(BaseModel):
    customerID: Optional[str] = Field(default=None, description="Optional customer identifier")
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(default=0, description="1 if senior citizen, else 0")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=250)
    TotalCharges: float = Field(ge=0, le=20000)


class FeedbackData(BaseModel):
    customer_id: str
    prediction_date: str
    churn_probability: float = Field(ge=0, le=1)
    recommended_action: str
    actual_churn_after_30_days: Literal[0, 1]
    campaign_cost: float = Field(default=0, ge=0)
    revenue_saved: float = Field(default=0, ge=0)
    notes: Optional[str] = None


app = FastAPI(
    title="Real-World Telco Customer Churn ML System",
    description="Business-aware churn prediction with API serving, explainability, feedback logging, and monitoring hooks.",
    version="3.0.0",
)


def _db_path() -> Path:
    return PROJECT_ROOT / get_nested(CONFIG, "paths.prediction_db", "artifacts/churn_predictions.sqlite")


def _feedback_path() -> Path:
    return PROJECT_ROOT / get_nested(CONFIG, "paths.feedback_log", "artifacts/feedback_log.csv")


async def verify_api_key(request: Request) -> None:
    """Optional API key protection. Set CHURN_API_KEY to enable it."""
    expected = os.getenv("CHURN_API_KEY")
    if not expected:
        return
    bearer = request.headers.get("authorization", "")
    token = bearer.replace("Bearer ", "", 1).strip() if bearer.startswith("Bearer ") else ""
    api_key = request.headers.get("x-api-key", "")
    if token != expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")




@app.middleware("http")
async def collect_runtime_metrics(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        REQUEST_COUNTS[f"{request.method} {request.url.path} {response.status_code}"] += 1
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY_SECONDS[f"{request.method} {request.url.path}"] += elapsed


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Lightweight Prometheus-style runtime metrics."""
    lines = [
        "# HELP telco_churn_requests_total Total HTTP requests by route and status",
        "# TYPE telco_churn_requests_total counter",
    ]
    for key, value in REQUEST_COUNTS.items():
        method, path, status = key.split(" ", 2)
        lines.append(f'telco_churn_requests_total{{method="{method}",path="{path}",status="{status}"}} {value}')
    lines.extend([
        "# HELP telco_churn_request_latency_seconds_total Total request latency seconds by route",
        "# TYPE telco_churn_request_latency_seconds_total counter",
    ])
    for key, value in REQUEST_LATENCY_SECONDS.items():
        method, path = key.split(" ", 1)
        lines.append(f'telco_churn_request_latency_seconds_total{{method="{method}",path="{path}"}} {value:.6f}')
    lines.append(f"telco_churn_predictions_total {PREDICTION_COUNT}")
    return "\n".join(lines) + "\n"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "real-world-telco-churn-ml",
        "docs": "/docs",
        "ui": "/ui",
        "model": model_status(),
    }


@app.get("/health")
def health():
    status = model_status()
    if not status.get("loaded"):
        return {"status": "degraded", "model": status}
    return {"status": "ok", "model": status}


@app.post("/predict")
def get_prediction(data: CustomerData, _: None = Depends(verify_api_key)):
    global PREDICTION_COUNT
    try:
        payload = data.model_dump(exclude_none=True)
        result = predict(payload)
        PREDICTION_COUNT += 1
        if os.getenv("STORE_PREDICTIONS", "false").lower() in {"1", "true", "yes"}:
            insert_prediction(_db_path(), payload, result)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain/predict")
def explain_prediction(data: CustomerData, _: None = Depends(verify_api_key)):
    """Return prediction plus optional local SHAP contributions."""
    try:
        payload = data.model_dump(exclude_none=True)
        result = predict(payload)
        result["model_explanation"] = local_shap_explanation(load_model_bundle(), payload)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/explain/global")
def explain_global(top_n: int = 20, _: None = Depends(verify_api_key)):
    try:
        return {"feature_importance": global_feature_importance(load_model_bundle(), top_n=top_n)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def log_feedback(data: FeedbackData, _: None = Depends(verify_api_key)):
    try:
        path = append_feedback(_feedback_path(), data.model_dump())
        return {"status": "saved", "feedback_log": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predictions/recent")
def recent_predictions(limit: int = 100, _: None = Depends(verify_api_key)):
    try:
        df = read_predictions(_db_path(), limit=limit)
        return {"rows": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/monitoring/drift/latest")
def latest_drift_report():
    """Compute drift between artifacts/reference_data.csv and latest batch_predictions.csv when both exist."""
    try:
        import pandas as pd

        reference_path = PROJECT_ROOT / "artifacts" / "reference_data.csv"
        current_path = PROJECT_ROOT / "artifacts" / "batch_predictions.csv"
        if not reference_path.exists() or not current_path.exists():
            raise FileNotFoundError("Run training and batch prediction first to create reference_data.csv and batch_predictions.csv")
        return drift_report(pd.read_csv(reference_path), pd.read_csv(current_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def gradio_interface(
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
):
    payload = {
        "customerID": customerID or "UI-CUSTOMER",
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    result = predict(payload)
    reasons = "\n".join(f"- {reason}" for reason in result.get("top_reasons", []))
    return (
        f"Prediction: {result['prediction']}\n"
        f"Probability: {result['churn_probability']:.1%}\n"
        f"Risk level: {result['risk_level']}\n"
        f"Threshold: {result['threshold']:.2f}\n"
        f"Expected retention profit: {result['expected_retention_profit']}\n"
        f"Revenue at risk: {result['estimated_revenue_at_risk']}\n"
        f"Recommended action: {result['recommended_action']}\n\n"
        f"Top reasons:\n{reasons}"
    )


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Customer ID", value="7590-VHVEG"),
        gr.Dropdown(["Male", "Female"], label="Gender", value="Female"),
        gr.Dropdown([0, 1], label="Senior Citizen", value=0),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=120),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method",
            value="Electronic check",
        ),
        gr.Number(label="Monthly Charges", value=85.0, minimum=0, maximum=250),
        gr.Number(label="Total Charges", value=85.0, minimum=0, maximum=20000),
    ],
    outputs=gr.Textbox(label="Business-ready churn result", lines=12),
    title="Telco Customer Churn Decision System",
    description="Train the model with scripts/run_pipeline.py first. The result includes probability, action recommendation, estimated value, and reasons.",
    examples=[
        ["7590-VHVEG", "Female", 0, "No", "No", 1, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 85.0, 85.0],
        ["5575-GNVDE", "Male", 0, "Yes", "Yes", 60, "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 45.0, 2700.0],
    ],
)

app = gr.mount_gradio_app(app, demo, path="/ui")
