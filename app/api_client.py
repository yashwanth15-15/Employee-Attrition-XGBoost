import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def check_backend_health():
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_employee(features_dict):
    """Call the predict endpoint."""
    url = f"{API_BASE_URL}/api/v1/predict"
    response = requests.post(url, json=features_dict, timeout=10)
    response.raise_for_status()
    return response.json()


def simulate_prediction(base_features, modified_features):
    """Call the simulate endpoint."""
    url = f"{API_BASE_URL}/api/v1/simulate"
    payload = {"base_features": base_features, "modified_features": modified_features}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def get_analytics_summary():
    """Call the analytics summary endpoint."""
    url = f"{API_BASE_URL}/api/v1/analytics/summary"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
