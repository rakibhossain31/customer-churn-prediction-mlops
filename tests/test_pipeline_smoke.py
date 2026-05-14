import pandas as pd

from src.data.preprocess import clean_telco_dataframe
from src.features.build_features import TelcoFeatureEngineer, build_model_pipeline


def sample_telco_df():
    return pd.DataFrame(
        [
            {
                "customerID": "0001-A",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": "85.0",
                "Churn": "Yes",
            },
            {
                "customerID": "0002-B",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 45.0,
                "TotalCharges": "2700.0",
                "Churn": "No",
            },
            {
                "customerID": "0003-C",
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 8,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
                "TotalCharges": "760.0",
                "Churn": "Yes",
            },
            {
                "customerID": "0004-D",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 20.0,
                "TotalCharges": "480.0",
                "Churn": "No",
            },
        ]
    )


def test_cleaning_maps_target_and_total_charges():
    cleaned = clean_telco_dataframe(sample_telco_df())
    assert cleaned["Churn"].tolist() == [1, 0, 1, 0]
    assert cleaned["TotalCharges"].dtype.kind in "fi"


def test_feature_engineer_adds_business_features():
    cleaned = clean_telco_dataframe(sample_telco_df())
    transformed = TelcoFeatureEngineer().transform(cleaned.drop(columns=["Churn"]))
    assert "tenure_group" in transformed.columns
    assert "addon_service_count" in transformed.columns
    assert "customerID" not in transformed.columns


def test_pipeline_fit_predict_smoke():
    cleaned = clean_telco_dataframe(sample_telco_df())
    X = cleaned.drop(columns=["Churn"])
    y = cleaned["Churn"]
    pipeline = build_model_pipeline(scale_pos_weight=1, random_state=42)
    pipeline.set_params(model__n_estimators=5, model__max_depth=2)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_business_decision_payload_contains_action():
    from src.business.decisioning import build_decision_payload

    row = sample_telco_df().iloc[0].drop(labels=["Churn"]).to_dict()
    decision = build_decision_payload(row, probability=0.82)
    assert decision["risk_level"] == "High"
    assert decision["recommended_action"]
    assert decision["top_reasons"]


def test_drift_report_detects_basic_schema():
    from src.monitoring.drift import drift_report

    reference = sample_telco_df()
    current = sample_telco_df().copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.5
    report = drift_report(reference, current)
    assert report["columns_checked"] > 0
    assert "columns" in report
