"""Reusable model training helpers."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import build_model_pipeline


def compute_scale_pos_weight(y: pd.Series) -> float:
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    return max(negative / max(positive, 1), 1.0)


def train_model(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train an XGBoost sklearn pipeline and return model plus holdout data."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipeline = build_model_pipeline(
        scale_pos_weight=compute_scale_pos_weight(y_train),
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test
