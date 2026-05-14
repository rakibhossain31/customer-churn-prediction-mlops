# Real-World Telco Customer Churn ML System

This is an end-to-end machine learning project for customer churn prediction using the Kaggle BlastChar Telco Customer Churn dataset. It goes beyond a basic notebook by adding a production-style pipeline, model serving, business decisioning, explainability, monitoring, feedback logging, dashboarding, Docker support, and CI tests.

## What this project does

The system answers a business question:

> Which customers are most likely to churn, what should the retention team do, and is the action expected to be financially useful?

The workflow is:

```text
Raw Kaggle CSV
  -> schema and data validation
  -> cleaning
  -> feature engineering
  -> train/validation/test split
  -> XGBoost model training
  -> business-aware threshold selection
  -> holdout evaluation
  -> model registry metadata
  -> API, UI, batch scoring, SQLite storage, dashboard, monitoring, feedback loop
```

## Major improvements added

### 1. Business-aware churn decisioning

Predictions now include more than a churn label:

```json
{
  "prediction": "Likely to churn",
  "prediction_flag": 1,
  "churn_probability": 0.82,
  "risk_level": "High",
  "recommended_action": "Retention specialist call, personalized discount, and service quality review",
  "expected_retention_profit": 145.5,
  "estimated_revenue_at_risk": 444.0,
  "top_reasons": [
    "Month-to-month contract reduces switching friction",
    "High monthly charges may increase price sensitivity"
  ]
}
```

Business assumptions are configurable in `configs/config.yaml`.

### 2. One fitted training-to-serving pipeline

The saved model bundle contains feature engineering, preprocessing, encoding, scaling, the XGBoost model, threshold, metrics, version, and metadata. This reduces training-serving mismatch.

### 3. Cost-sensitive thresholding

The training script can optimize threshold by F2, recall, precision, F1, or estimated retention value.

```bash
python scripts/run_pipeline.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --threshold_metric retention_value \
  --save_value 200 \
  --contact_cost 10
```

### 4. Explainability

The project provides:

- `artifacts/feature_importance.json`
- `/explain/global` for global model drivers
- `/explain/predict` for prediction plus optional SHAP explanation
- business-readable churn reasons in every prediction

### 5. Data validation

Before training, the pipeline checks required columns, allowed categorical values, numeric ranges, duplicate customer IDs, and target values.

```bash
python scripts/validate_data.py --input data/raw/Telco-Customer-Churn.csv
```

### 6. Runtime metrics plus monitoring and drift reports

The API exposes lightweight Prometheus-style runtime metrics at `/metrics`, including request counts, accumulated latency, and prediction volume.

The training pipeline saves a reference dataset. Later scoring data can be compared against the reference distribution.

```bash
python scripts/monitor_drift.py \
  --reference artifacts/reference_data.csv \
  --current artifacts/batch_predictions.csv
```

Outputs:

```text
artifacts/drift_report.json
artifacts/drift_report.html
```

### 7. Feedback loop

The API has a `/feedback` endpoint, and feedback can also be stored in:

```text
artifacts/feedback_log.csv
```

A template is available at:

```text
data/feedback/feedback_template.csv
```

### 8. SQLite storage

Batch predictions can be written into SQLite for downstream reporting:

```bash
python scripts/score_to_sqlite.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --db artifacts/churn_predictions.sqlite
```

### 9. Business dashboard

A Streamlit dashboard summarizes model metrics, scored customers, expected retention value, top model drivers, drift status, and feedback.

```bash
streamlit run dashboards/streamlit_dashboard.py
```

### 10. API security hook

Set `CHURN_API_KEY` to require either:

```text
Authorization: Bearer <key>
```

or:

```text
x-api-key: <key>
```

for prediction, explanation, feedback, and stored prediction endpoints.

## Project structure

```text
.
├── configs/
│   └── config.yaml
├── dashboards/
│   └── streamlit_dashboard.py
├── data/
│   ├── feedback/
│   ├── processed/
│   ├── raw/
│   └── sample/
├── scripts/
│   ├── batch_predict.py
│   ├── generate_sample_data.py
│   ├── monitor_drift.py
│   ├── run_pipeline.py
│   ├── score_to_sqlite.py
│   └── validate_data.py
├── src/
│   ├── app/
│   ├── business/
│   ├── config.py
│   ├── database/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   ├── feedback/
│   ├── monitoring/
│   ├── models/
│   ├── serving/
│   └── utils/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── MODEL_CARD.md
└── README.md
```

## How to run locally

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Kaggle dataset

Download the Kaggle BlastChar Telco Customer Churn dataset and place the CSV here:

```text
data/raw/Telco-Customer-Churn.csv
```

The raw Kaggle dataset is not included in this zip.

### 4. Train the model

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv
```

Created files include:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_metadata.json
artifacts/model_registry.json
artifacts/feature_importance.json
artifacts/reference_data.csv
artifacts/classification_report.txt
artifacts/sample_payload.json
data/processed/telco_churn_cleaned.csv
```

### 5. Start the API and Gradio UI

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/ui
http://127.0.0.1:8000/health
```

### 6. Test one prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @artifacts/sample_payload.json
```

### 7. Run batch scoring

```bash
python scripts/batch_predict.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --output artifacts/batch_predictions.csv
```

### 8. Run drift monitoring

```bash
python scripts/monitor_drift.py \
  --reference artifacts/reference_data.csv \
  --current artifacts/batch_predictions.csv
```

### 9. Open the dashboard

```bash
streamlit run dashboards/streamlit_dashboard.py
```

## Docker

Train the model first so `artifacts/model.joblib` exists, then run:

```bash
docker compose up --build
```

API:

```text
http://127.0.0.1:8000/docs
```

Dashboard:

```text
http://127.0.0.1:8501
```

## Quick smoke test without Kaggle data

This creates synthetic Telco-like data only to test the pipeline wiring.

```bash
python scripts/generate_sample_data.py --rows 500 --output data/sample/synthetic_telco_sample.csv
python scripts/run_pipeline.py --input data/sample/synthetic_telco_sample.csv --n_estimators 20
python scripts/batch_predict.py --input data/sample/synthetic_telco_sample.csv --output artifacts/batch_predictions.csv
python scripts/monitor_drift.py --reference artifacts/reference_data.csv --current artifacts/batch_predictions.csv
pytest -q
```

## Useful Make commands

```bash
make setup
make train DATA=data/raw/Telco-Customer-Churn.csv
make api
make batch DATA=data/raw/Telco-Customer-Churn.csv
make drift
make dashboard
make test
```

## Why this is closer to a real-world ML project

A real-world ML system needs more than a trained classifier. This version includes business objectives, data validation, threshold optimization, explainable prediction outputs, API serving, web UI, batch scoring, model metadata and registry records, monitoring, feedback collection, SQLite persistence, dashboarding, Docker deployment, and CI tests.

The most important upgrade is that the model output is now actionable. It tells the business who is risky, why they may churn, what action to take, and the expected value of acting.

