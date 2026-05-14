# Model Card: Telco Customer Churn Decision System

## Intended use

This model estimates the probability that a telecom customer will churn and supports retention prioritization. It is intended for portfolio demonstration, churn analytics, and decision-support workflows where a retention team needs ranked customer lists, suggested actions, and monitoring outputs.

## Dataset

The project expects the Kaggle BlastChar Telco Customer Churn CSV with the original schema. The raw dataset is not redistributed in this repository. Place it at:

```text
data/raw/Telco-Customer-Churn.csv
```

## Target

The target variable is `Churn`, mapped as:

```text
No -> 0
Yes -> 1
```

## Model

The training pipeline exports one joblib bundle containing:

- sklearn-compatible business feature engineering
- numeric imputation and scaling
- categorical imputation and one-hot encoding
- XGBoost binary classifier
- selected decision threshold
- model metadata and business assumptions

## Business decision layer

The API response includes:

- churn probability
- class prediction
- risk level
- recommended retention action
- expected retention profit
- estimated revenue at risk
- top human-readable churn reasons

The default decision threshold is selected using validation data and can optimize F2 or business retention value.

## Evaluation

After training, metrics are saved to:

```text
artifacts/metrics.json
artifacts/classification_report.txt
```

Expected metrics include ROC-AUC, average precision, precision, recall, F1, F2, confusion matrix, contact rate, and estimated retention value.

## Explainability

The project provides two explanation layers:

1. Rule-based business reasons that are returned on every prediction.
2. Model-level feature importance and optional local SHAP explanations through `/explain/global` and `/explain/predict`.

## Monitoring

The training pipeline saves `artifacts/reference_data.csv`. Production or batch data can be compared against this file using:

```bash
python scripts/monitor_drift.py --reference artifacts/reference_data.csv --current artifacts/batch_predictions.csv
```

The drift report is saved as JSON and HTML.

## Feedback loop

Retention outcomes can be logged through the `/feedback` endpoint or appended to `artifacts/feedback_log.csv`. This supports future retraining using real customer outcomes.

## Limitations

- The Kaggle dataset is static and does not contain real-time customer behavior, support tickets, call quality, complaints, or competitor offers.
- Expected profit uses configurable assumptions because true customer lifetime value and campaign acceptance rates are not included in the dataset.
- Rule-based explanations are intended for business interpretability and should not be treated as causal proof.
- SHAP explanations are optional and may be slower than normal prediction.

## Ethical and operational considerations

- Do not use the model as the only basis for customer treatment decisions.
- Review retention offers for fairness across age, gender, and other sensitive or proxy attributes.
- Monitor drift and retrain when input distributions or realized churn outcomes change.
- Protect customer data in transit and at rest. Set `CHURN_API_KEY` for API key protection before public deployment.
