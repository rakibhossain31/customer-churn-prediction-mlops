"""Feedback logging for closed-loop churn learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

FEEDBACK_COLUMNS = [
    "customer_id",
    "prediction_date",
    "churn_probability",
    "recommended_action",
    "actual_churn_after_30_days",
    "campaign_cost",
    "revenue_saved",
    "notes",
]


def append_feedback(path: str | Path, record: Dict[str, Any]) -> Path:
    feedback_path = Path(path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {col: record.get(col) for col in FEEDBACK_COLUMNS}
    new_row = pd.DataFrame([normalized])
    if feedback_path.exists():
        existing = pd.read_csv(feedback_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(feedback_path, index=False)
    return feedback_path
