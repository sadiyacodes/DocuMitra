"""Tests for CSV → Chunk ingestion."""
import csv
from pathlib import Path

import pytest

from backend.ingestion.ingest_csv import ingest_csv


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "employees.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "department", "salary"])
        writer.writeheader()
        for i in range(10):
            writer.writerow({"id": i, "name": f"Person {i}", "department": "Engineering", "salary": 80000 + i * 1000})
    return path


def test_ingest_csv_returns_at_least_one_chunk(csv_file):
    chunks = ingest_csv(csv_file, source_id="csv-001", filename="employees.csv", access_roles=["hr"])
    assert len(chunks) >= 1


def test_ingest_csv_source_fields(csv_file):
    chunks = ingest_csv(csv_file, source_id="csv-001", filename="employees.csv", access_roles=["hr", "admin"])
    for c in chunks:
        assert c.source_id == "csv-001"
        assert c.source_type == "csv"
        assert c.filename == "employees.csv"
        assert "hr" in c.access_roles
        assert "admin" in c.access_roles


def test_ingest_csv_chunk_text_contains_column_names(csv_file):
    chunks = ingest_csv(csv_file, source_id="s1", filename="test.csv", access_roles=[])
    combined = " ".join(c.text for c in chunks).lower()
    assert "name" in combined
    assert "department" in combined
    assert "salary" in combined


def test_ingest_csv_unique_chunk_ids(csv_file):
    chunks = ingest_csv(csv_file, source_id="s2", filename="test.csv", access_roles=[])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_csv_empty_file_returns_no_chunks(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,name\n", encoding="utf-8")
    chunks = ingest_csv(path, source_id="s3", filename="empty.csv", access_roles=[])
    assert chunks == []


def test_ingest_csv_positive_token_counts(csv_file):
    chunks = ingest_csv(csv_file, source_id="s4", filename="test.csv", access_roles=[])
    assert all(c.token_count > 0 for c in chunks)
