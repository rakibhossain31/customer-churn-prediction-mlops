#!/usr/bin/env python3
"""Compare current scoring data against training reference data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift import drift_report, render_drift_report_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Telco churn drift report")
    parser.add_argument("--reference", default="artifacts/reference_data.csv", help="Reference training data CSV")
    parser.add_argument("--current", required=True, help="Current production/scoring data CSV")
    parser.add_argument("--json-output", default="artifacts/drift_report.json")
    parser.add_argument("--html-output", default="artifacts/drift_report.html")
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    parser.add_argument("--category-share-threshold", type=float, default=0.15)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    report = drift_report(reference, current, args.psi_threshold, args.category_share_threshold)

    json_path = Path(args.json_output)
    html_path = Path(args.html_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_drift_report_html(report), encoding="utf-8")

    print(f"Drift detected: {report['drift_detected']}")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved HTML report to {html_path}")


if __name__ == "__main__":
    main()
