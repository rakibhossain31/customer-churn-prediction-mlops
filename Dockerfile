FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_ARTIFACT_PATH=/app/artifacts/model.joblib

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
