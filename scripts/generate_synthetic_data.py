#!/usr/bin/env python3
"""Generate synthetic enterprise CSV and JSON datasets."""
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(exist_ok=True)

# --- employees.csv (hr + admin access) ---
departments = ["Engineering", "HR", "Finance", "Marketing", "Legal"]
employees = [
    {
        "id": i + 1,
        "name": f"Employee {i+1:03d}",
        "department": departments[i % len(departments)],
        "salary": random.randint(60000, 130000),
        "manager": f"Manager {(i // 5) + 1:02d}",
        "hire_date": (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2000))).strftime("%Y-%m-%d"),
        "status": "active" if i % 10 != 0 else "inactive",
    }
    for i in range(50)
]

csv_path = out_dir / "employees.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=employees[0].keys())
    writer.writeheader()
    writer.writerows(employees)
print(f"Written {len(employees)} rows to {csv_path}")

# --- audit_logs.json (admin only) ---
actions = ["login", "logout", "view_report", "export_data", "modify_record", "delete_record", "access_denied"]
logs = [
    {
        "id": i + 1,
        "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i * 3)).isoformat() + "Z",
        "user": f"user{(i % 10) + 1:02d}",
        "action": actions[i % len(actions)],
        "resource": f"document_{random.randint(1, 20):03d}",
        "ip": f"192.168.1.{random.randint(1, 254)}",
        "status": "success" if i % 7 != 0 else "denied",
        "duration_ms": random.randint(10, 500),
    }
    for i in range(80)
]

json_path = out_dir / "audit_logs.json"
json_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
print(f"Written {len(logs)} records to {json_path}")
