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
CHUNK_MIN_TOKENS = 500
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
    chunk_id: str
    pdf_id: str
    filename: str
    page_number: int
    text: str
    token_count: int
    language: str
    bbox: tuple[float, float, float, float] | None
