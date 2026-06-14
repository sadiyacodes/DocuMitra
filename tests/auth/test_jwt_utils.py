"""Tests for JWT token creation and decoding."""
import pytest
import jwt as pyjwt
import backend.auth.jwt_utils as jwt_utils


def test_create_and_decode_roundtrip():
    token = jwt_utils.create_token("alice", "admin")
    payload = jwt_utils.decode_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_expired_token_raises():
    original = jwt_utils.TOKEN_EXPIRE_MINUTES
    jwt_utils.TOKEN_EXPIRE_MINUTES = -1
    token = jwt_utils.create_token("bob", "hr")
    jwt_utils.TOKEN_EXPIRE_MINUTES = original
    with pytest.raises(pyjwt.ExpiredSignatureError):
        jwt_utils.decode_token(token)


def test_tampered_token_raises():
    token = jwt_utils.create_token("alice", "admin")
    bad = token[:-4] + "xxxx"
    with pytest.raises(pyjwt.InvalidTokenError):
        jwt_utils.decode_token(bad)


def test_token_is_string():
    token = jwt_utils.create_token("carol", "finance")
    assert isinstance(token, str)
    assert len(token) > 20
