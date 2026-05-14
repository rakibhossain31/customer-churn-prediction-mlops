#!/usr/bin/env python3
"""Batch score customers and store predictions in SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.sqlite_store import insert_predictions
from src.serving.inference import predict_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers and write predictions to SQLite")
    parser.add_argument("--input", required=True)
    parser.add_argument("--db", default="artifacts/churn_predictions.sqlite")
    parser.add_argument("--csv-output", default="artifacts/batch_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    scored = predict_batch(df)
    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(csv_path, index=False)
    inserted = insert_predictions(args.db, scored.to_dict(orient="records"))
    print(f"Saved {len(scored)} scored rows to {csv_path}")
    print(f"Inserted {inserted} prediction rows into {args.db}")


if __name__ == "__main__":
    main()
