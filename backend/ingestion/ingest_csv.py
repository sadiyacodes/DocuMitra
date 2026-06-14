"""Ingest CSV files into the chunk pipeline."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from backend.ingestion.chunker import (
    CHUNK_MAX_TOKENS,
    Chunk,
    _count_tokens,
    _detect_language,
    _get_tokenizer,
)


def _row_to_text(row: dict[str, str]) -> str:
    return " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())


def ingest_csv(
    path: Path,
    source_id: str,
    filename: str,
    access_roles: list[str],
) -> list[Chunk]:
    """Read a CSV file and return a list of token-bounded Chunks."""
    tokenizer = _get_tokenizer()

    with open(path, newline="", encoding="utf-8") as f:
        row_texts = [_row_to_text(r) for r in csv.DictReader(f) if any(r.values())]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0
    record_start = 0

    for i, row_text in enumerate(row_texts):
        row_tokens = _count_tokens(row_text, tokenizer)
        if buffer_tokens + row_tokens > CHUNK_MAX_TOKENS and buffer:
            text = "\n".join(buffer)
            chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=source_id,
                source_type="csv",
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

        buffer.append(row_text)
        buffer_tokens += row_tokens

    if buffer:
        text = "\n".join(buffer)
        chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            source_type="csv",
            filename=filename,
            page_number=record_start + 1,
            text=text,
            token_count=buffer_tokens,
            language=_detect_language(text),
            bbox=None,
            access_roles=list(access_roles),
        ))

    return chunks
