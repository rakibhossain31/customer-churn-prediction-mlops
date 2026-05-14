"""Model evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> Dict[str, object]:
    """Evaluate a fitted probability model using a configurable threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    report = classification_report(y_test, preds, digits=3, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return metrics
