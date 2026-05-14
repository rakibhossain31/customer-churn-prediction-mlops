"""Business decision layer for churn predictions.

The ML model estimates churn risk. This module translates that probability into
retention actions, expected value, and human-readable reasons so the output can
be used by non-technical business teams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class BusinessRules:
    save_value: float = 200.0
    contact_cost: float = 10.0
    retention_effectiveness: float = 0.35
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40
    high_action: str = "Retention specialist call, personalized discount, and service quality review"
    medium_action: str = "Targeted retention email, plan review, and add-on support offer"
    low_action: str = "Standard customer success nurture campaign"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def risk_level(probability: float, rules: BusinessRules | None = None) -> str:
    rules = rules or BusinessRules()
    if probability >= rules.high_risk_threshold:
        return "High"
    if probability >= rules.medium_risk_threshold:
        return "Medium"
    return "Low"


def revenue_at_risk(row: Dict[str, Any], probability: float, rules: BusinessRules | None = None) -> float:
    """Estimate revenue at risk from churn.

    The Kaggle dataset does not include contractual margin or customer lifetime
    value. This lightweight estimate uses the larger of configured save value
    and a tenure-adjusted monthly charge window.
    """
    rules = rules or BusinessRules()
    monthly = _as_float(row.get("MonthlyCharges"), 0.0)
    tenure = _as_float(row.get("tenure"), 0.0)
    remaining_months_proxy = 12 if tenure < 24 else 6
    estimated_value = max(rules.save_value, monthly * remaining_months_proxy)
    return round(probability * estimated_value, 2)


def expected_retention_profit(row: Dict[str, Any], probability: float, rules: BusinessRules | None = None) -> float:
    rules = rules or BusinessRules()
    recoverable_value = revenue_at_risk(row, probability, rules) * rules.retention_effectiveness
    return round(recoverable_value - rules.contact_cost, 2)


def recommend_action(row: Dict[str, Any], probability: float, rules: BusinessRules | None = None) -> str:
    rules = rules or BusinessRules()
    level = risk_level(probability, rules)
    profit = expected_retention_profit(row, probability, rules)
    if level == "High" and profit > 0:
        return rules.high_action
    if level == "Medium" and profit > 0:
        return rules.medium_action
    if profit <= 0:
        return "Do not send costly retention offer. Keep in low-cost nurture campaign."
    return rules.low_action


def churn_reason_rules(row: Dict[str, Any], max_reasons: int = 6) -> List[str]:
    """Return human-readable churn-risk reasons from raw customer attributes."""
    reasons: List[str] = []
    tenure = _as_float(row.get("tenure"), 0.0)
    monthly = _as_float(row.get("MonthlyCharges"), 0.0)
    total = _as_float(row.get("TotalCharges"), 0.0)

    if str(row.get("Contract", "")).strip() == "Month-to-month":
        reasons.append("Month-to-month contract reduces switching friction")
    if tenure <= 6:
        reasons.append("Very low tenure customer with weak relationship history")
    elif tenure <= 12:
        reasons.append("Early-tenure customer still at risk of switching")
    if str(row.get("InternetService", "")).strip() == "Fiber optic":
        reasons.append("Fiber optic customers in this dataset often show higher churn risk")
    if str(row.get("PaymentMethod", "")).strip() == "Electronic check":
        reasons.append("Electronic check payment method is a common churn-risk signal")
    if str(row.get("OnlineSecurity", "")).strip() == "No":
        reasons.append("No online security add-on")
    if str(row.get("TechSupport", "")).strip() == "No":
        reasons.append("No technical support add-on")
    if monthly >= 80:
        reasons.append("High monthly charges may increase price sensitivity")
    if str(row.get("PaperlessBilling", "")).strip() == "Yes":
        reasons.append("Paperless billing aligns with a higher-risk segment in the dataset")
    if str(row.get("SeniorCitizen", "")).strip() in {"1", "1.0", "True", "true"}:
        reasons.append("Senior citizen segment requires careful retention handling")
    if tenure > 0 and total / max(tenure, 1) > monthly * 1.20:
        reasons.append("Historical charges are higher than current monthly charge pattern")

    if not reasons:
        reasons.append("No strong rule-based churn drivers detected; review model probability and customer history")
    return reasons[:max_reasons]


def build_decision_payload(row: Dict[str, Any], probability: float, rules: BusinessRules | None = None) -> Dict[str, Any]:
    rules = rules or BusinessRules()
    return {
        "risk_level": risk_level(probability, rules),
        "recommended_action": recommend_action(row, probability, rules),
        "expected_retention_profit": expected_retention_profit(row, probability, rules),
        "estimated_revenue_at_risk": revenue_at_risk(row, probability, rules),
        "top_reasons": churn_reason_rules(row),
        "business_assumptions": {
            "save_value": rules.save_value,
            "contact_cost": rules.contact_cost,
            "retention_effectiveness": rules.retention_effectiveness,
        },
    }
