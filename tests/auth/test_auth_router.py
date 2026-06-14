"""Tests for POST /auth/login."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.auth.models import User


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def _mock_user():
    return User(username="alice", hashed_password="$hashed$", role="admin")


def test_login_success_returns_token(client):
    with patch("backend.auth.router.get_user", return_value=_mock_user()), \
         patch("backend.auth.router._verify_password", return_value=True):
        resp = client.post("/auth/login", data={"username": "alice", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client):
    with patch("backend.auth.router.get_user", return_value=_mock_user()), \
         patch("backend.auth.router._verify_password", return_value=False):
        resp = client.post("/auth/login", data={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(client):
    with patch("backend.auth.router.get_user", return_value=None):
        resp = client.post("/auth/login", data={"username": "nobody", "password": "x"})
    assert resp.status_code == 401
