"""Small threshold-tuning helper used by the training script."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score


def tune_threshold(y_true, probabilities, beta: float = 2.0) -> float:
    """Return the threshold that maximizes F-beta on validation data."""
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        preds = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))
    return best_threshold
