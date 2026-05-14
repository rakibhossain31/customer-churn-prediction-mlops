#!/usr/bin/env python3
"""Generate a small Telco-like sample dataset for local smoke testing.

This is not a replacement for the Kaggle dataset. It only helps users verify
that the pipeline, API, dashboard, and monitoring commands work end to end.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def make_row(i: int) -> dict:
    random.seed(i)
    contract = random.choices(["Month-to-month", "One year", "Two year"], weights=[0.55, 0.25, 0.20])[0]
    tenure = random.randint(0, 72)
    internet = random.choices(["DSL", "Fiber optic", "No"], weights=[0.35, 0.45, 0.20])[0]
    monthly = round(random.uniform(20, 115), 2) if internet != "No" else round(random.uniform(18, 35), 2)
    addons = "No internet service" if internet == "No" else random.choice(["Yes", "No"])
    payment = random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    churn_score = 0
    churn_score += 2 if contract == "Month-to-month" else -1
    churn_score += 1 if tenure < 12 else -1
    churn_score += 1 if internet == "Fiber optic" else 0
    churn_score += 1 if payment == "Electronic check" else 0
    churn_score += 1 if monthly > 80 else 0
    churn = "Yes" if churn_score + random.uniform(-2, 2) > 1.5 else "No"
    return {
        "customerID": f"SYN-{i:05d}",
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": random.choice(["Yes", "No"]),
        "InternetService": internet,
        "OnlineSecurity": addons,
        "OnlineBackup": addons if internet == "No" else random.choice(["Yes", "No"]),
        "DeviceProtection": addons if internet == "No" else random.choice(["Yes", "No"]),
        "TechSupport": addons if internet == "No" else random.choice(["Yes", "No"]),
        "StreamingTV": addons if internet == "No" else random.choice(["Yes", "No"]),
        "StreamingMovies": addons if internet == "No" else random.choice(["Yes", "No"]),
        "Contract": contract,
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": round(max(0, tenure * monthly + random.uniform(-50, 50)), 2),
        "Churn": churn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Telco-like sample data")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--output", default="data/sample/synthetic_telco_sample.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([make_row(i) for i in range(args.rows)]).to_csv(output, index=False)
    print(f"Saved sample data to {output}")


if __name__ == "__main__":
    main()
