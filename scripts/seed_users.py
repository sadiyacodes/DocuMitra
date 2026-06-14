#!/usr/bin/env python3
"""Generate data/users.json with bcrypt-hashed passwords."""
import json
import bcrypt
from pathlib import Path

USERS = [
    {"username": "alice", "password": "admin123",   "role": "admin"},
    {"username": "bob",   "password": "hr123",      "role": "hr"},
    {"username": "carol", "password": "finance123", "role": "finance"},
    {"username": "dave",  "password": "eng123",     "role": "engineering"},
]

out = [
    {
        "username": u["username"],
        "hashed_password": bcrypt.hashpw(u["password"].encode(), bcrypt.gensalt()).decode(),
        "role": u["role"],
    }
    for u in USERS
]

out_path = Path(__file__).parent.parent / "data" / "users.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Written {len(out)} users to {out_path}")
for u in USERS:
    print(f"  {u['username']} / {u['password']}  →  role: {u['role']}")
