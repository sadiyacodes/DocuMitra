"""Ingest JSON log files into the chunk pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.ingestion.chunker import (
    CHUNK_MAX_TOKENS,
    Chunk,
    _count_tokens,
    _detect_language,
    _get_tokenizer,
)


def _record_to_text(record: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in record.items())


def ingest_json(
    path: Path,
    source_id: str,
    filename: str,
    access_roles: list[str],
) -> list[Chunk]:
    """Read a JSON file (list of records or single dict) and return token-bounded Chunks."""
    tokenizer = _get_tokenizer()

    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = raw if isinstance(raw, list) else [raw]
    if not records:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0
    record_start = 0

    for i, record in enumerate(records):
        rec_text = _record_to_text(record)
        rec_tokens = _count_tokens(rec_text, tokenizer)
        if buffer_tokens + rec_tokens > CHUNK_MAX_TOKENS and buffer:
            text = "\n---\n".join(buffer)
            chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=source_id,
                source_type="json",
                filename=filename,
                page_number=record_start + 1,
                text=text,
                token_count=buffer_tokens,
                language=_detect_language(text),
                bbox=None,
                access_roles=list(access_roles),
            ))
            buffer = []
            buffer_tokens = 0
            chunk_index += 1
            record_start = i

        buffer.append(rec_text)
        buffer_tokens += rec_tokens

    if buffer:
        text = "\n---\n".join(buffer)
        chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            source_type="json",
            filename=filename,
            page_number=record_start + 1,
            text=text,
            token_count=buffer_tokens,
            language=_detect_language(text),
            bbox=None,
            access_roles=list(access_roles),
        ))

    return chunks
