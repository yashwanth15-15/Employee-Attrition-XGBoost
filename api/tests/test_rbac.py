import pytest
from fastapi.testclient import TestClient

from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def setup_users(client):
    from api.auth.auth_service import auth_service
    from api.services.db_service import db_service
    import uuid
    from api.auth.password import hash_password
    from datetime import datetime
    
    from api.database.session import SessionLocal
    
    db = SessionLocal()
    try:
        if not auth_service.get_user_by_username(db, "test_viewer"):
            db_service.create_user(db, {
                "id": str(uuid.uuid4()),
                "username": "test_viewer",
                "email": "viewer@example.com",
                "hashed_password": hash_password("Viewer@123"),
                "role": "Viewer",
                "is_active": True,
                "created_at": datetime.now().isoformat()
            })
        if not auth_service.get_user_by_username(db, "test_hr"):
            db_service.create_user(db, {
                "id": str(uuid.uuid4()),
                "username": "test_hr",
                "email": "hr@example.com",
                "hashed_password": hash_password("HR@123"),
                "role": "HR_Manager",
                "is_active": True,
                "created_at": datetime.now().isoformat()
            })
    finally:
        db.close()
    
@pytest.fixture
def viewer_token(client, setup_users):
    response = client.post("/api/v1/auth/login", data={"username": "test_viewer", "password": "Viewer@123"})
    return response.json()["access_token"]

@pytest.fixture
def hr_token(client, setup_users):
    response = client.post("/api/v1/auth/login", data={"username": "test_hr", "password": "HR@123"})
    return response.json()["access_token"]

def test_viewer_access_denied_to_predict(client, viewer_token):
    response = client.post("/api/v1/predict", json={}, headers={"Authorization": f"Bearer {viewer_token}"})
    assert response.status_code == 403

def test_viewer_access_allowed_to_analytics(client, viewer_token):
    response = client.get("/api/v1/analytics/summary", headers={"Authorization": f"Bearer {viewer_token}"})
    assert response.status_code == 200

def test_hr_access_allowed_to_predict(client, hr_token):
    payload = {
        "Age": 30, "Gender": "Male", "Department": "Sales", "Monthly Income": 5000,
        "Marital Status": "Single", "OverTime": "No", "Years At Company": 5,
        "Total Working Years": 8, "Job Satisfaction": 3, "Environment Satisfaction": 3,
        "Work Life Balance": 3, "Years Since Last Promotion": 0, "Training Times Last Year": 2,
        "Business Travel": "Travel_Rarely", "Distance From Home": 5
    }
    response = client.post("/api/v1/predict", json=payload, headers={"Authorization": f"Bearer {hr_token}"})
    assert response.status_code == 200
