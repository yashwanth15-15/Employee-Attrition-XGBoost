import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def admin_token(client):
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "Admin@123"}
    )
    return response.json()["access_token"]


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(client, admin_token):
    payload = {
        "Age": 30,
        "Gender": "Male",
        "Department": "Sales",
        "Monthly Income": 5000,
        "Marital Status": "Single",
        "OverTime": "No",
        "Years At Company": 5,
        "Total Working Years": 8,
        "Job Satisfaction": 3,
        "Environment Satisfaction": 3,
        "Work Life Balance": 3,
        "Years Since Last Promotion": 0,
        "Training Times Last Year": 2,
        "Business Travel": "Travel_Rarely",
        "Distance From Home": 5,
    }

    response = client.post("/api/v1/predict", json=payload, headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk_category" in data
    assert "top_risk_drivers" in data
    assert "recommendations" in data
    assert "immediate_actions" in data["recommendations"]


def test_simulate_endpoint(client, admin_token):
    payload = {
        "base_features": {
            "Age": 30,
            "Gender": "Male",
            "Department": "Sales",
            "Monthly Income": 5000,
            "Marital Status": "Single",
            "OverTime": "Yes",
            "Years At Company": 5,
            "Total Working Years": 8,
            "Job Satisfaction": 2,
            "Environment Satisfaction": 3,
            "Work Life Balance": 2,
            "Years Since Last Promotion": 0,
            "Training Times Last Year": 2,
            "Business Travel": "Travel_Rarely",
            "Distance From Home": 5,
        },
        "modified_features": {"OverTime": "No", "Job Satisfaction": 4},
    }

    response = client.post("/api/v1/simulate", json=payload, headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    data = response.json()
    assert "probability_change" in data
    assert "impact_analysis" in data
