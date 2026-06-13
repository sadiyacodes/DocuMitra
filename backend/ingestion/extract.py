"""PDF → per-page text extraction (PyMuPDF + Tesseract)."""
from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

MIN_NATIVE_CHARS = 50
HEADER_LINES = 3
FOOTER_LINES = 3
HEURISTIC_THRESHOLD = 0.60
OCR_DPI = 300

log = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("­", "")
    for ligature, replacement in (
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
        ("ﬀ", "ff"),
        ("ﬃ", "ffi"),
        ("ﬄ", "ffl"),
    ):
        text = text.replace(ligature, replacement)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_scanned(text: str) -> bool:
    return len(text.strip()) < MIN_NATIVE_CHARS


@dataclass
class PageContent:
    pdf_id: str
    filename: str
    page_number: int
    text: str
    bbox: tuple[float, float, float, float] | None
    is_ocr: bool


@dataclass
class ExtractedDocument:
    pdf_id: str
    filename: str
    pages: list[PageContent]


class ExtractionError(Exception):
    def __init__(self, filename: str, cause: Exception | str) -> None:
        self.filename = filename
        self.cause = cause
        super().__init__(f"Failed to extract '{filename}': {cause}")
