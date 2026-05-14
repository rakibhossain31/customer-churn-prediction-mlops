#!/usr/bin/env python3
"""Validate a raw Telco churn CSV before training or scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.validate_data import validate_telco_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Telco Customer Churn data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/data_validation.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    passed, issues = validate_telco_data(df)
    output = {"passed": passed, "issues": issues, "rows": len(df), "columns": list(df.columns)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
