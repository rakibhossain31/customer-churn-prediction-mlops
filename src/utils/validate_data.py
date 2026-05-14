"""Lightweight data validation for the Kaggle Telco Customer Churn dataset."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}

ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    "Churn": {"Yes", "No", 0, 1},
}


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate schema, ranges, and categorical values.

    This replaces the older Great Expectations dependency with a simple, stable
    validator that works across environments and CI runners.
    """
    issues: List[str] = []
    columns = set(df.columns.str.strip())

    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        issues.append(f"Missing required columns: {missing}")

    working = df.copy()
    working.columns = working.columns.str.strip()

    for col, allowed in ALLOWED_VALUES.items():
        if col in working.columns:
            values = set(working[col].dropna().astype(str).str.strip().unique())
            allowed_as_str = {str(v) for v in allowed}
            unexpected = sorted(values - allowed_as_str)
            if unexpected:
                issues.append(f"Column {col!r} has unexpected values: {unexpected[:10]}")

    for col, lower, upper in [
        ("tenure", 0, 120),
        ("MonthlyCharges", 0, 250),
        ("TotalCharges", 0, 20000),
    ]:
        if col in working.columns:
            numeric = pd.to_numeric(working[col].replace("", pd.NA), errors="coerce")
            if numeric.dropna().lt(lower).any() or numeric.dropna().gt(upper).any():
                issues.append(f"Column {col!r} contains values outside expected range [{lower}, {upper}]")

    if "customerID" in working.columns and working["customerID"].duplicated().any():
        issues.append("Duplicate customerID values found")

    return len(issues) == 0, issues
