"""JWT token creation and decoding."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM: str = "HS256"
TOKEN_EXPIRE_MINUTES: int = 480


def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
