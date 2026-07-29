import os
import requests

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://employee-attrition-xgboost.onrender.com"
)


def check_backend_health():
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_employee(features_dict):
    """Call the predict endpoint."""
    url = f"{BACKEND_URL}/api/v1/predict"
    response = requests.post(url, json=features_dict, timeout=10)
    response.raise_for_status()
    return response.json()


def simulate_prediction(base_features, modified_features):
    """Call the simulate endpoint."""
    url = f"{BACKEND_URL}/api/v1/simulate"
    payload = {"base_features": base_features, "modified_features": modified_features}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def get_analytics_summary():
    """Call the analytics summary endpoint."""
    url = f"{BACKEND_URL}/api/v1/analytics/summary"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
