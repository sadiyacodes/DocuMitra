"""User model and in-file user store (loaded from data/users.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROLES = {"admin", "hr", "finance", "engineering"}

_USERS_FILE = Path(__file__).parent.parent.parent / "data" / "users.json"


@dataclass
class User:
    username: str
    hashed_password: str
    role: str


def load_users() -> dict[str, User]:
    """Load users from data/users.json. Returns empty dict if file absent (pre-seed)."""
    if not _USERS_FILE.exists():
        return {}
    data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    return {u["username"]: User(**u) for u in data}


def get_user(username: str) -> User | None:
    return load_users().get(username)
