# Developer Notes

Use `README.md` as the source of truth for setup and execution.

Current primary commands:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
pytest -q
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
python scripts/batch_predict.py --input data/raw/Telco-Customer-Churn.csv --output artifacts/batch_predictions.csv
```

The serving app expects `artifacts/model.joblib`. Create it by running the training pipeline first.
