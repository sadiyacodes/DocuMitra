"""Tests for JSON log → Chunk ingestion."""
import json
from pathlib import Path

import pytest

from backend.ingestion.ingest_json import ingest_json


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    path = tmp_path / "audit.json"
    records = [
        {"timestamp": f"2024-01-15T10:{i:02d}:00Z", "user": f"user{i}", "action": "login", "status": "success"}
        for i in range(20)
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_ingest_json_returns_at_least_one_chunk(json_file):
    chunks = ingest_json(json_file, source_id="json-001", filename="audit.json", access_roles=["admin"])
    assert len(chunks) >= 1


def test_ingest_json_source_fields(json_file):
    chunks = ingest_json(json_file, source_id="json-001", filename="audit.json", access_roles=["admin"])
    for c in chunks:
        assert c.source_id == "json-001"
        assert c.source_type == "json"
        assert c.filename == "audit.json"
        assert "admin" in c.access_roles


def test_ingest_json_chunk_text_contains_field_names(json_file):
    chunks = ingest_json(json_file, source_id="s1", filename="audit.json", access_roles=[])
    combined = " ".join(c.text for c in chunks).lower()
    assert "timestamp" in combined
    assert "action" in combined


def test_ingest_json_unique_chunk_ids(json_file):
    chunks = ingest_json(json_file, source_id="s2", filename="audit.json", access_roles=[])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_json_single_dict(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": "1.0", "env": "prod"}), encoding="utf-8")
    chunks = ingest_json(path, source_id="s3", filename="config.json", access_roles=[])
    assert len(chunks) >= 1


def test_ingest_json_empty_list(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    chunks = ingest_json(path, source_id="s4", filename="empty.json", access_roles=[])
    assert chunks == []
