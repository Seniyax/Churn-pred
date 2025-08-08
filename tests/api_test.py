import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api import app,model_path
import joblib
import os

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client



@pytest.fixture
def sample_data():
    return {
        "customerID": "1234-XYZ",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "600.0"

    }

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction(client,sample_data):

    if not os.path.exists(model_path):
        pytest.skip("model not available")

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    result = response.json()
    assert "customerID" in result,"Missing customer ID in response"
    assert result["customerID"] == sample_data["customerID"],"Customer ID mismatch"
    assert "churn_probability" in result,"Churn probability in response"
    assert isinstance(result["churn_probability"], float)
    assert 0 <= result["churn_probability"] <= 1
    assert "churn_prediction" in result
    assert result["churn_prediction"] in [0, 1]

def test_prediction_error(client):
    invalid_data = {"customerID": "1234-XYZ", "gender": "Invalid"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
    assert "detail" in response.json()


