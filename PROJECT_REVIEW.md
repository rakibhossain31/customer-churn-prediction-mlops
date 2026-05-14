# Project Review: Real-World Telco Customer Churn ML System

## Summary

The project has been upgraded from a standard churn classifier into a fuller end-to-end ML system. It now supports model training, validation, threshold tuning, model registry metadata, API serving, Gradio UI, Streamlit dashboard, batch scoring, SQLite persistence, drift monitoring, feedback logging, optional SHAP explanations, Docker deployment, and CI smoke tests.

## What changed

### Machine learning pipeline

- Added configuration-driven defaults in `configs/config.yaml`.
- Kept one saved sklearn-compatible pipeline for feature engineering, preprocessing, and XGBoost inference.
- Added model metadata, model version, run ID, data hash, model registry JSON, and feature importance export.
- Saved reference training data for monitoring.

### Business layer

- Added churn risk level.
- Added recommended retention action.
- Added expected retention profit.
- Added estimated revenue at risk.
- Added human-readable top churn reasons.

### Serving layer

- Expanded FastAPI response from simple prediction to business-ready output.
- Added `/explain/global` and `/explain/predict`.
- Added `/feedback` endpoint.
- Added `/metrics` endpoint for lightweight runtime monitoring.
- Added optional API key security through `CHURN_API_KEY`.
- Added optional prediction storage when `STORE_PREDICTIONS=true`.

### Monitoring and feedback

- Added drift monitoring with PSI and categorical share change.
- Added HTML and JSON drift reports.
- Added feedback log template and feedback append utility.
- Added SQLite storage for predictions.

### User-facing tools

- Added Streamlit dashboard.
- Added Makefile shortcuts.
- Added Docker Compose for API and dashboard.
- Added synthetic data generator for smoke testing without the Kaggle CSV.

## Validation performed

The Python modules compile successfully, and the pytest smoke suite passes. The pipeline was also smoke-tested on generated Telco-like sample data using training, batch prediction, and drift monitoring commands.

## Important note

The Kaggle raw dataset is not included in the zip. Users should download it and place it at `data/raw/Telco-Customer-Churn.csv`.
