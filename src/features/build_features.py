"""Feature engineering and sklearn pipeline builders."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Business-aware feature engineering for raw Telco customer rows.

    The class is intentionally sklearn-compatible so the same fitted object is
    used during training, API inference, and batch scoring.
    """

    id_columns: Iterable[str] = ("customerID", "CustomerID", "customer_id")
    numeric_columns: Iterable[str] = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        df.columns = df.columns.str.strip()

        for col in self.id_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # The target must never enter the feature pipeline.
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        tenure = df.get("tenure", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
        monthly = df.get("MonthlyCharges", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
        total = df.get("TotalCharges", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)

        # Domain features that make the model more interpretable and useful for churn actioning.
        df["tenure_group"] = pd.cut(
            tenure,
            bins=[-0.001, 6, 12, 24, 48, 72, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49-72", "73+"],
        ).astype("object")
        df["estimated_total_charges"] = tenure * monthly
        df["charges_gap"] = total - df["estimated_total_charges"]
        df["charges_per_tenure"] = total / np.maximum(tenure, 1)

        if "InternetService" in df.columns:
            df["has_internet"] = (df["InternetService"].fillna("No") != "No").astype(int)

        add_on_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"] if c in df.columns]
        if add_on_cols:
            df["addon_service_count"] = df[add_on_cols].eq("Yes").sum(axis=1)

        if "Contract" in df.columns:
            df["is_month_to_month"] = df["Contract"].eq("Month-to-month").astype(int)

        if "PaymentMethod" in df.columns:
            df["uses_electronic_check"] = df["PaymentMethod"].eq("Electronic check").astype(int)

        return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fit on the training split only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
    )


def build_model_pipeline(scale_pos_weight: float = 1.0, random_state: int = 42, n_estimators: int = 250, max_depth: int = 3) -> Pipeline:
    """Build the complete raw-data-to-prediction pipeline."""
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.035,
        max_depth=max_depth,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=2.00,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("preprocessing", build_preprocessor()),
            ("model", classifier),
        ]
    )


# Backward-compatible helper used by the original training script/tests.
def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    transformed = TelcoFeatureEngineer().transform(df)
    if target_col in df.columns:
        transformed[target_col] = df[target_col].values
    return transformed
