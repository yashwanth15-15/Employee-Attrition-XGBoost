import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_login_success(client):
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "Admin@123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_invalid_password(client):
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "wrongpassword"}
    )
    assert response.status_code == 401


def test_login_invalid_username(client):
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "invaliduser", "password": "Admin@123"}
    )
    assert response.status_code == 401


def test_get_me_success(client):
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "Admin@123"}
    )
    token = login_response.json()["access_token"]
    
    # Get profile
    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "admin"
    assert me_response.json()["role"] == "Admin"


def test_get_me_no_token(client):
    me_response = client.get("/api/v1/auth/me")
    assert me_response.status_code == 401


def test_get_me_invalid_token(client):
    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_token_here"}
    )
    assert me_response.status_code == 401
