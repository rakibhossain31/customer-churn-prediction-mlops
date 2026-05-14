"""SQLite storage helpers for predictions and feedback."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

PREDICTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS churn_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
    churn_probability REAL,
    prediction_label TEXT,
    risk_level TEXT,
    recommended_action TEXT,
    expected_retention_profit REAL,
    estimated_revenue_at_risk REAL,
    payload_json TEXT
);
"""

FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS churn_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    prediction_date TEXT,
    actual_churn_after_30_days INTEGER,
    campaign_cost REAL,
    revenue_saved REAL,
    notes TEXT,
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(PREDICTION_SCHEMA)
    con.execute(FEEDBACK_SCHEMA)
    con.commit()
    return con


def insert_prediction(db_path: str | Path, row: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    customer_id = str(row.get("customerID") or row.get("customer_id") or row.get("CustomerID") or "")
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO churn_predictions (
                customer_id, churn_probability, prediction_label, risk_level,
                recommended_action, expected_retention_profit,
                estimated_revenue_at_risk, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                float(prediction.get("churn_probability", 0.0)),
                prediction.get("prediction"),
                prediction.get("risk_level"),
                prediction.get("recommended_action"),
                float(prediction.get("expected_retention_profit", 0.0)),
                float(prediction.get("estimated_revenue_at_risk", 0.0)),
                json.dumps(row, default=str),
            ),
        )
        con.commit()


def insert_predictions(db_path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with connect(db_path) as con:
        for row in rows:
            con.execute(
                """
                INSERT INTO churn_predictions (
                    customer_id, churn_probability, prediction_label, risk_level,
                    recommended_action, expected_retention_profit,
                    estimated_revenue_at_risk, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("customerID") or row.get("customer_id") or row.get("CustomerID") or ""),
                    float(row.get("churn_probability", 0.0)),
                    row.get("prediction_label") or row.get("prediction"),
                    row.get("risk_level"),
                    row.get("recommended_action"),
                    float(row.get("expected_retention_profit", 0.0)),
                    float(row.get("estimated_revenue_at_risk", 0.0)),
                    json.dumps(row, default=str),
                ),
            )
            count += 1
        con.commit()
    return count


def read_predictions(db_path: str | Path, limit: int = 1000) -> pd.DataFrame:
    with connect(db_path) as con:
        return pd.read_sql_query(
            "SELECT * FROM churn_predictions ORDER BY scored_at DESC, id DESC LIMIT ?",
            con,
            params=(limit,),
        )
