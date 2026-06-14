"""Splits extracted page text into overlapping chunks with metadata."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import langdetect
from langdetect import DetectorFactory, LangDetectException
from transformers import AutoTokenizer

from backend.ingestion.extract import ExtractedDocument, PageContent

DetectorFactory.seed = 0

CHUNK_MAX_TOKENS = 1000
CHUNK_MIN_TOKENS = 500  # informational target; sub-minimum chunks (e.g. short last page) are kept as-is
OVERLAP_TOKENS = 200
TOKENIZER_MODEL = "BAAI/bge-small-en-v1.5"

log = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


def _count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except LangDetectException:
        return "unknown"


@dataclass
class Chunk:
    chunk_id:     str
    source_id:    str   # stable identifier for the source document (was pdf_id)
    source_type:  str   # "pdf" | "csv" | "json"
    filename:     str
    page_number:  int
    text:         str
    token_count:  int
    language:     str
    bbox:         tuple[float, float, float, float] | None
    access_roles: list[str]


def _chunk_page(
    page: PageContent,
    tokenizer: AutoTokenizer,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    access_roles: list[str] | None = None,
) -> list[Chunk]:
    """Split a single page into overlapping token-bounded chunks.

    Greedily accumulates sentences up to max_tokens, then flushes. Carries
    the tail sentences (up to overlap_tokens) into the next chunk. Sentences
    exceeding max_tokens are truncated by the tokenizer. Returns [] for empty pages.
    """
    if access_roles is None:
        access_roles = []
    if not page.text.strip():
        return []

    sentences = _split_sentences(page.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens: int = 0
    chunk_index: int = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence, tokenizer)

        if sentence_tokens > max_tokens:
            ids = tokenizer.encode(sentence, add_special_tokens=False)[:max_tokens]
            sentence = tokenizer.decode(ids)
            sentence_tokens = _count_tokens(sentence, tokenizer)

        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            chunk_text = " ".join(buffer)
            chunk_id = hashlib.sha256(
                f"{page.pdf_id}:{page.filename}:{page.page_number}:{chunk_index}".encode()
            ).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=page.pdf_id,
                source_type="pdf",
                filename=page.filename,
                page_number=page.page_number,
                text=chunk_text,
                token_count=buffer_tokens,
                language=_detect_language(chunk_text),
                bbox=page.bbox,
                access_roles=access_roles,
            ))
            chunk_index += 1

            overlap: list[str] = []
            overlap_token_count: int = 0
            for s in reversed(buffer):
                s_tokens = _count_tokens(s, tokenizer)
                if overlap_token_count + s_tokens <= overlap_tokens:
                    overlap.insert(0, s)
                    overlap_token_count += s_tokens
                else:
                    break

            buffer = overlap + [sentence]
            buffer_tokens = overlap_token_count + sentence_tokens
        else:
            buffer.append(sentence)
            buffer_tokens += sentence_tokens

    if buffer:
        chunk_text = " ".join(buffer)
        chunk_id = hashlib.sha256(
            f"{page.pdf_id}:{page.page_number}:{chunk_index}".encode()
        ).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_id=page.pdf_id,
            source_type="pdf",
            filename=page.filename,
            page_number=page.page_number,
            text=chunk_text,
            token_count=buffer_tokens,
            language=_detect_language(chunk_text),
            bbox=page.bbox,
            access_roles=access_roles,
        ))

    return chunks


def _get_tokenizer() -> AutoTokenizer:
    """Lazy singleton: load the bge-small tokenizer once and keep it warm."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    return _tokenizer


def chunk_document(
    doc: ExtractedDocument,
    access_roles: list[str] | None = None,
) -> list[Chunk]:
    """Split all pages of an extracted document into overlapping chunks.

    Returns a list of Chunk objects aggregated from all pages.
    Empty pages (whitespace-only text) are skipped.
    """
    tokenizer = _get_tokenizer()
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(_chunk_page(page, tokenizer, access_roles=access_roles or []))
    return chunks
