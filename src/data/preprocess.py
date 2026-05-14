"""Data cleaning utilities for the Telco Customer Churn project."""

from __future__ import annotations

import pandas as pd


TARGET_MAP = {"No": 0, "Yes": 1, 0: 0, 1: 1}


def clean_telco_dataframe(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco churn dataframe.

    The function keeps feature columns in their business-readable form so the
    saved sklearn pipeline can learn preprocessing from the training split only.
    This avoids feature leakage and removes train/serve transformation drift.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    # Trim whitespace in text columns. The Kaggle dataset stores blank
    # TotalCharges values as strings, so stripping first is important.
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    for col in ["tenure", "MonthlyCharges"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    if target_col in cleaned.columns:
        cleaned[target_col] = cleaned[target_col].map(TARGET_MAP)
        if cleaned[target_col].isna().any():
            bad_values = sorted(df.loc[cleaned[target_col].isna(), target_col].dropna().astype(str).unique())
            raise ValueError(f"Target column {target_col!r} contains unexpected values: {bad_values}")
        cleaned[target_col] = cleaned[target_col].astype(int)

    return cleaned


# Backward-compatible name used by the original scripts.
def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    return clean_telco_dataframe(df, target_col=target_col)
