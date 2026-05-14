#!/usr/bin/env python3
"""Clean the raw Kaggle CSV and save data/processed/telco_churn_cleaned.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess import clean_telco_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"))
    parser.add_argument("--target", default="Churn")
    args = parser.parse_args()

    df = load_data(args.input)
    cleaned = clean_telco_dataframe(df, target_col=args.target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    print(f"Saved cleaned data to {output} with shape {cleaned.shape}")
