"""Inference utilities for the saved Telco churn sklearn pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.business.decisioning import BusinessRules, build_decision_payload, risk_level
from src.config import get_nested, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATHS = [
    Path(os.getenv("MODEL_ARTIFACT_PATH", "")) if os.getenv("MODEL_ARTIFACT_PATH") else None,
    PROJECT_ROOT / "artifacts" / "model.joblib",
    Path("/app/artifacts/model.joblib"),
]

_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_MODEL_PATH: Optional[Path] = None
_CONFIG = load_config()


def _business_rules_from_bundle(bundle: Dict[str, Any] | None = None) -> BusinessRules:
    business = (bundle or {}).get("business", {}) if bundle else {}
    config_business = _CONFIG.get("business", {}) if isinstance(_CONFIG.get("business"), dict) else {}
    source = {**config_business, **business}
    actions = source.get("actions", {}) if isinstance(source.get("actions"), dict) else {}
    return BusinessRules(
        save_value=float(source.get("save_value", 200.0)),
        contact_cost=float(source.get("contact_cost", 10.0)),
        retention_effectiveness=float(source.get("retention_effectiveness", 0.35)),
        high_risk_threshold=float(source.get("high_risk_threshold", 0.70)),
        medium_risk_threshold=float(source.get("medium_risk_threshold", 0.40)),
        high_action=str(actions.get("high", BusinessRules().high_action)),
        medium_action=str(actions.get("medium", BusinessRules().medium_action)),
        low_action=str(actions.get("low", BusinessRules().low_action)),
    )


def _find_model_path() -> Path:
    for path in DEFAULT_ARTIFACT_PATHS:
        if path and path.exists():
            return path
    searched = [str(p) for p in DEFAULT_ARTIFACT_PATHS if p]
    raise FileNotFoundError(
        "No trained model bundle found. Run: "
        "python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv. "
        f"Searched: {searched}"
    )


def load_model_bundle(force_reload: bool = False) -> Dict[str, Any]:
    """Load the model bundle once and reuse it across predictions."""
    global _MODEL_BUNDLE, _MODEL_PATH
    if _MODEL_BUNDLE is not None and not force_reload:
        return _MODEL_BUNDLE

    _MODEL_PATH = _find_model_path()
    _MODEL_BUNDLE = joblib.load(_MODEL_PATH)
    required = {"pipeline", "threshold"}
    missing = required - set(_MODEL_BUNDLE.keys())
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")
    return _MODEL_BUNDLE


def model_status() -> Dict[str, Any]:
    """Return health details without forcing app startup failure."""
    try:
        bundle = load_model_bundle()
        return {
            "loaded": True,
            "model_path": str(_MODEL_PATH),
            "threshold": float(bundle["threshold"]),
            "model_type": bundle.get("model_type", "unknown"),
            "model_version": bundle.get("model_version", "unregistered"),
            "training_timestamp": bundle.get("training_timestamp"),
            "metrics": bundle.get("metrics", {}),
        }
    except Exception as exc:  # pragma: no cover - used by health endpoint
        return {"loaded": False, "error": str(exc)}


def predict_proba(input_dict: Dict[str, Any]) -> float:
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    df = pd.DataFrame([input_dict])
    probability = float(pipeline.predict_proba(df)[:, 1][0])
    return probability


def predict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict churn for one customer and return business-ready output."""
    bundle = load_model_bundle()
    threshold = float(bundle["threshold"])
    probability = predict_proba(input_dict)
    churn_flag = int(probability >= threshold)
    rules = _business_rules_from_bundle(bundle)
    decision = build_decision_payload(input_dict, probability, rules)
    return {
        "prediction": "Likely to churn" if churn_flag else "Not likely to churn",
        "prediction_flag": churn_flag,
        "churn_probability": round(probability, 4),
        "threshold": round(threshold, 4),
        **decision,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a dataframe of raw customer rows with business actions."""
    bundle = load_model_bundle()
    threshold = float(bundle["threshold"])
    rules = _business_rules_from_bundle(bundle)
    scored = df.copy()
    probabilities = bundle["pipeline"].predict_proba(scored)[:, 1]
    scored["churn_probability"] = probabilities
    scored["prediction"] = (probabilities >= threshold).astype(int)
    scored["prediction_label"] = scored["prediction"].map({1: "Likely to churn", 0: "Not likely to churn"})
    scored["risk_level"] = scored["churn_probability"].map(lambda value: risk_level(float(value), rules))

    decisions = [build_decision_payload(row, float(prob), rules) for row, prob in zip(scored.to_dict(orient="records"), probabilities)]
    scored["recommended_action"] = [d["recommended_action"] for d in decisions]
    scored["expected_retention_profit"] = [d["expected_retention_profit"] for d in decisions]
    scored["estimated_revenue_at_risk"] = [d["estimated_revenue_at_risk"] for d in decisions]
    scored["top_reasons"] = [" | ".join(d["top_reasons"]) for d in decisions]
    return scored
