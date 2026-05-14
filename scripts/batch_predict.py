#!/usr/bin/env python3
"""Run batch predictions using the exported model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.inference import predict_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch score Telco customers")
    parser.add_argument("--input", required=True, help="CSV with raw customer rows")
    parser.add_argument("--output", default="artifacts/batch_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    scored = predict_batch(df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Saved scored file to {output_path}")
