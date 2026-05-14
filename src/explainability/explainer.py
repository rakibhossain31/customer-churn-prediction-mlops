"""Model explainability helpers.

The project uses two explanation layers:
1. Fast business reasons from raw customer fields, returned on every prediction.
2. Optional model-level explanations from feature importances and SHAP when the
   shap package is installed. SHAP is intentionally optional so API startup does
   not fail in lightweight environments.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def get_transformed_feature_names(pipeline: Any) -> List[str]:
    """Return post-preprocessing feature names when available."""
    preprocessor = pipeline.named_steps.get("preprocessing")
    if preprocessor is None:
        return []
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return []


def global_feature_importance(model_bundle: Dict[str, Any], top_n: int = 20) -> List[Dict[str, Any]]:
    """Return top global feature importances from the fitted XGBoost model."""
    pipeline = model_bundle["pipeline"]
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return []

    names = get_transformed_feature_names(pipeline)
    importances = np.asarray(model.feature_importances_, dtype=float)
    if not names or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": names[i], "importance": round(float(importances[i]), 6)}
        for i in order
        if float(importances[i]) > 0
    ]


def local_shap_explanation(model_bundle: Dict[str, Any], row: Dict[str, Any], top_n: int = 8) -> Dict[str, Any]:
    """Return local SHAP contributions for one customer when SHAP is installed.

    This function is designed for explicit explanation requests, not every API
    prediction, because SHAP can be expensive on large models.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": f"SHAP is not available: {exc}"}

    try:
        pipeline = model_bundle["pipeline"]
        engineered = pipeline.named_steps["feature_engineering"].transform(pd.DataFrame([row]))
        matrix = pipeline.named_steps["preprocessing"].transform(engineered)
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(matrix)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        values = np.asarray(values).reshape(-1)
        names = get_transformed_feature_names(pipeline)
        if not names or len(names) != len(values):
            names = [f"feature_{i}" for i in range(len(values))]
        order = np.argsort(np.abs(values))[::-1][:top_n]
        contributions = [
            {
                "feature": names[i],
                "contribution": round(float(values[i]), 6),
                "direction": "increases churn risk" if values[i] > 0 else "decreases churn risk",
            }
            for i in order
        ]
        return {"available": True, "top_contributions": contributions}
    except Exception as exc:
        return {"available": False, "reason": f"Unable to compute SHAP explanation: {exc}"}
