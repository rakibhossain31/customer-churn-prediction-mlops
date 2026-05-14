"""Simple production monitoring utilities for data and prediction drift."""

from __future__ import annotations

import html
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return cleaned


def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max() + 1e-9, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = pd.cut(reference, bins=edges, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(current, bins=edges, include_lowest=True).value_counts(sort=False)
    ref_pct = (ref_counts / max(ref_counts.sum(), 1)).replace(0, 1e-6)
    cur_pct = (cur_counts / max(cur_counts.sum(), 1)).replace(0, 1e-6)
    psi = ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum()
    return float(max(psi, 0.0))


def _categorical_drift(reference: pd.Series, current: pd.Series) -> Dict[str, float]:
    ref = reference.fillna("__MISSING__").astype(str)
    cur = current.fillna("__MISSING__").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_dist = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).replace(0, 1e-6)
    cur_dist = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).replace(0, 1e-6)
    psi = ((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum()
    max_share_change = (cur_dist - ref_dist).abs().max()
    return {"psi": float(max(psi, 0.0)), "max_share_change": float(max_share_change)}


def drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_alert_threshold: float = 0.20,
    max_category_share_change: float = 0.15,
) -> Dict[str, Any]:
    """Compare reference and current data distributions."""
    reference = _clean_frame(reference_df)
    current = _clean_frame(current_df)
    common_cols = [c for c in reference.columns if c in current.columns and c != "Churn"]
    columns: List[Dict[str, Any]] = []

    for col in common_cols:
        ref_numeric = pd.to_numeric(reference[col], errors="coerce")
        cur_numeric = pd.to_numeric(current[col], errors="coerce")
        numeric_ratio = ref_numeric.notna().mean()
        if numeric_ratio > 0.80:
            psi = _numeric_psi(ref_numeric, cur_numeric)
            drifted = psi >= psi_alert_threshold
            columns.append({
                "column": col,
                "type": "numeric",
                "psi": round(psi, 6),
                "drifted": bool(drifted),
                "reference_missing_rate": round(float(reference[col].isna().mean()), 6),
                "current_missing_rate": round(float(current[col].isna().mean()), 6),
            })
        else:
            values = _categorical_drift(reference[col], current[col])
            drifted = values["psi"] >= psi_alert_threshold or values["max_share_change"] >= max_category_share_change
            columns.append({
                "column": col,
                "type": "categorical",
                "psi": round(values["psi"], 6),
                "max_share_change": round(values["max_share_change"], 6),
                "drifted": bool(drifted),
                "reference_missing_rate": round(float(reference[col].isna().mean()), 6),
                "current_missing_rate": round(float(current[col].isna().mean()), 6),
            })

    drifted_cols = [row["column"] for row in columns if row["drifted"]]
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "columns_checked": len(columns),
        "drifted_columns": drifted_cols,
        "drift_detected": bool(drifted_cols),
        "thresholds": {
            "psi_alert_threshold": psi_alert_threshold,
            "max_category_share_change": max_category_share_change,
        },
        "columns": columns,
    }


def render_drift_report_html(report: Dict[str, Any]) -> str:
    """Create a lightweight HTML report for browser viewing."""
    rows = []
    for item in report.get("columns", []):
        status = "DRIFT" if item.get("drifted") else "ok"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('column')))}</td>"
            f"<td>{html.escape(str(item.get('type')))}</td>"
            f"<td>{item.get('psi', '')}</td>"
            f"<td>{item.get('max_share_change', '')}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    badge = "Drift detected" if report.get("drift_detected") else "No major drift detected"
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Telco Churn Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px; text-align: left; }}
    th {{ background: #f7f7f7; }}
  </style>
</head>
<body>
  <h1>Telco Churn Data Drift Report</h1>
  <div class="card">
    <h2>{badge}</h2>
    <p>Reference rows: {report.get('reference_rows')} | Current rows: {report.get('current_rows')} | Columns checked: {report.get('columns_checked')}</p>
    <p>Drifted columns: {html.escape(', '.join(report.get('drifted_columns', [])) or 'None')}</p>
  </div>
  <table>
    <thead><tr><th>Column</th><th>Type</th><th>PSI</th><th>Max category share change</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
