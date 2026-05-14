#!/usr/bin/env python3
"""Train, evaluate, register, and export an end-to-end Telco churn model pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_nested, load_config
from src.data.load_data import load_data
from src.data.preprocess import clean_telco_dataframe
from src.explainability.explainer import global_feature_importance
from src.features.build_features import build_model_pipeline
from src.utils.validate_data import validate_telco_data


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_weight(y: pd.Series) -> float:
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    return max(negative / max(positive, 1), 1.0)


def _retention_value(y_true: np.ndarray, proba: np.ndarray, threshold: float, save_value: float, contact_cost: float) -> float:
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    contacted = tp + fp
    return (tp * save_value) - (contacted * contact_cost)


def choose_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    metric: str = "retention_value",
    save_value: float = 200.0,
    contact_cost: float = 10.0,
) -> Tuple[float, Dict[str, float]]:
    """Choose a threshold using validation data only."""
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    y_np = y_true.to_numpy()

    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        row = {
            "threshold": float(threshold),
            "precision": precision_score(y_np, pred, zero_division=0),
            "recall": recall_score(y_np, pred, zero_division=0),
            "f1": f1_score(y_np, pred, zero_division=0),
            "f2": fbeta_score(y_np, pred, beta=2, zero_division=0),
            "retention_value": _retention_value(y_np, proba, threshold, save_value, contact_cost),
        }
        rows.append(row)

    if metric not in rows[0]:
        raise ValueError(f"Unknown threshold metric {metric!r}. Choose one of: {sorted(rows[0])}")

    best = max(rows, key=lambda r: (r[metric], r["f2"], r["recall"]))
    return float(best["threshold"]), best


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float, save_value: float, contact_cost: float) -> Dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    contacted = int(tp + fp)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "retention_value": float(_retention_value(y_true.to_numpy(), proba, threshold, save_value, contact_cost)),
        "customers_contacted": contacted,
        "contact_rate": float(contacted / len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def write_sample_payload(path: Path) -> None:
    payload = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_model_registry(registry_path: Path, record: Dict[str, Any]) -> None:
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"models": []}
    registry.setdefault("models", []).append(record)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir or get_nested(config, "paths.artifacts_dir", "artifacts")).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or get_nested(config, "paths.raw_data", "data/raw/Telco-Customer-Churn.csv")
    target = args.target or get_nested(config, "project.target", "Churn")
    random_state = args.random_state if args.random_state is not None else int(get_nested(config, "project.random_state", 42))
    test_size = args.test_size if args.test_size is not None else float(get_nested(config, "training.test_size", 0.20))
    validation_size = args.validation_size if args.validation_size is not None else float(get_nested(config, "training.validation_size", 0.20))
    threshold_metric = args.threshold_metric or get_nested(config, "training.threshold_metric", "retention_value")
    n_estimators = args.n_estimators if args.n_estimators is not None else int(get_nested(config, "training.n_estimators", 250))
    max_depth = args.max_depth if args.max_depth is not None else int(get_nested(config, "training.max_depth", 3))
    save_value = args.save_value if args.save_value is not None else float(get_nested(config, "business.save_value", 200.0))
    contact_cost = args.contact_cost if args.contact_cost is not None else float(get_nested(config, "business.contact_cost", 10.0))

    print("Loading raw data...")
    raw_df = load_data(input_path)
    print(f"Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]:,} columns")

    is_valid, issues = validate_telco_data(raw_df)
    (artifacts_dir / "data_validation.json").write_text(
        json.dumps({"passed": is_valid, "issues": issues, "rows": len(raw_df), "columns": list(raw_df.columns)}, indent=2),
        encoding="utf-8",
    )
    if not is_valid:
        raise ValueError("Data validation failed. See artifacts/data_validation.json for details.")

    df = clean_telco_dataframe(raw_df, target_col=target)
    processed_path = PROJECT_ROOT / get_nested(config, "paths.processed_data", "data/processed/telco_churn_cleaned.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found")

    X = df.drop(columns=[target])
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    reference_path = artifacts_dir / "reference_data.csv"
    X_train_full.assign(**{target: y_train_full.values}).to_csv(reference_path, index=False)

    scale_pos_weight = _class_weight(y_train)
    print(f"Training validation model with scale_pos_weight={scale_pos_weight:.3f}...")
    validation_pipeline = build_model_pipeline(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    start = time.time()
    validation_pipeline.fit(X_train, y_train)
    train_seconds = time.time() - start

    valid_proba = validation_pipeline.predict_proba(X_valid)[:, 1]
    threshold, threshold_details = choose_threshold(
        y_valid,
        valid_proba,
        metric=threshold_metric,
        save_value=save_value,
        contact_cost=contact_cost,
    )
    print(f"Selected threshold={threshold:.2f} using validation {threshold_metric}: {threshold_details[threshold_metric]:.4f}")

    print("Refitting final pipeline on full training split...")
    final_pipeline = build_model_pipeline(
        scale_pos_weight=_class_weight(y_train_full),
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    final_pipeline.fit(X_train_full, y_train_full)

    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_proba, threshold, save_value, contact_cost)
    metrics.update(
        {
            "rows": int(df.shape[0]),
            "features_before_encoding": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "test_size": float(test_size),
            "validation_size_within_train": float(validation_size),
            "threshold_selection": threshold_details,
            "train_seconds_validation_model": float(train_seconds),
            "save_value": float(save_value),
            "contact_cost": float(contact_cost),
        }
    )

    run_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    model_version = f"churn-xgb-{timestamp[:10]}-{run_id[:8]}"
    business_config = get_nested(config, "business", {}) or {
        "save_value": save_value,
        "contact_cost": contact_cost,
        "retention_effectiveness": 0.35,
        "high_risk_threshold": 0.70,
        "medium_risk_threshold": 0.40,
    }
    business_config["save_value"] = save_value
    business_config["contact_cost"] = contact_cost

    model_bundle = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "metrics": metrics,
        "target": target,
        "model_type": "XGBoost + sklearn Pipeline",
        "model_version": model_version,
        "run_id": run_id,
        "training_timestamp": timestamp,
        "training_data_sha256": _file_sha256(input_path),
        "business": business_config,
        "created_by": "scripts/run_pipeline.py",
    }

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps({k: v for k, v in model_bundle.items() if k != "pipeline"}, indent=2),
        encoding="utf-8",
    )
    report = classification_report(y_test, (test_proba >= threshold).astype(int), digits=3)
    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    write_sample_payload(artifacts_dir / "sample_payload.json")

    feature_importance = global_feature_importance(model_bundle, top_n=30)
    (artifacts_dir / "feature_importance.json").write_text(json.dumps(feature_importance, indent=2), encoding="utf-8")

    registry_record = {
        "model_version": model_version,
        "run_id": run_id,
        "training_timestamp": timestamp,
        "model_path": str(model_path),
        "training_data_sha256": model_bundle["training_data_sha256"],
        "threshold": threshold,
        "threshold_metric": threshold_metric,
        "metrics": {key: value for key, value in metrics.items() if isinstance(value, (int, float))},
    }
    _append_model_registry(artifacts_dir / "model_registry.json", registry_record)

    if args.use_mlflow:
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name=model_version):
                mlflow.log_params(
                    {
                        "model": "xgboost_pipeline",
                        "threshold": threshold,
                        "threshold_metric": threshold_metric,
                        "test_size": test_size,
                        "validation_size": validation_size,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "save_value": save_value,
                        "contact_cost": contact_cost,
                    }
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                mlflow.log_artifacts(str(artifacts_dir))
                mlflow.sklearn.log_model(final_pipeline, artifact_path="sklearn_pipeline")
        except ImportError:
            print("MLflow is not installed. Skipping MLflow logging.")

    print("\nFinal holdout metrics")
    for key in ["precision", "recall", "f1", "f2", "roc_auc", "average_precision", "retention_value", "contact_rate"]:
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Saved model bundle to {model_path}")
    print(f"Saved registry to {artifacts_dir / 'model_registry.json'}")
    print(f"Saved reference data for drift monitoring to {reference_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export an end-to-end Telco churn model")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--input", default=None, help="Raw Kaggle Telco CSV path")
    parser.add_argument("--target", default=None)
    parser.add_argument("--artifacts_dir", default=None)
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--validation_size", type=float, default=None, help="Validation fraction inside the training split")
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--threshold_metric", default=None, choices=["precision", "recall", "f1", "f2", "retention_value"])
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--save_value", type=float, default=None, help="Estimated value of retaining a true churn customer")
    parser.add_argument("--contact_cost", type=float, default=None, help="Cost of contacting one customer")
    parser.add_argument("--use_mlflow", action="store_true", help="Also log metrics/artifacts to MLflow if installed")
    parser.add_argument("--experiment", default="Telco Churn Real World ML")
    main(parser.parse_args())
