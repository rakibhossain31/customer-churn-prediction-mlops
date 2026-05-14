PYTHON ?= python
DATA ?= data/raw/Telco-Customer-Churn.csv
SAMPLE ?= data/sample/synthetic_telco_sample.csv

.PHONY: setup sample validate train test api dashboard batch drift sqlite docker-build docker-run clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

sample:
	$(PYTHON) scripts/generate_sample_data.py --rows 500 --output $(SAMPLE)

validate:
	$(PYTHON) scripts/validate_data.py --input $(DATA)

train:
	$(PYTHON) scripts/run_pipeline.py --input $(DATA)

test:
	pytest -q

api:
	uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboards/streamlit_dashboard.py

batch:
	$(PYTHON) scripts/batch_predict.py --input $(DATA) --output artifacts/batch_predictions.csv

drift:
	$(PYTHON) scripts/monitor_drift.py --reference artifacts/reference_data.csv --current artifacts/batch_predictions.csv

sqlite:
	$(PYTHON) scripts/score_to_sqlite.py --input $(DATA) --db artifacts/churn_predictions.sqlite

docker-build:
	docker build -t telco-churn-api .

docker-run:
	docker run --rm -p 8000:8000 -e STORE_PREDICTIONS=true telco-churn-api

clean:
	rm -rf artifacts/*.joblib artifacts/*.json artifacts/*.txt artifacts/*.csv artifacts/*.sqlite artifacts/*.html data/processed/*.csv
