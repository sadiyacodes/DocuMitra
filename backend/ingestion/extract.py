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


def _strip_headers_footers(pages: list[str]) -> list[str]:
    if len(pages) <= 1:
        return pages

    total = len(pages)
    line_counts: dict[str, int] = {}

    for text in pages:
        lines = text.splitlines()
        candidates = (
            [l.strip() for l in lines[:HEADER_LINES]]
            + [l.strip() for l in lines[-FOOTER_LINES:]]
        )
        for line in set(candidates):
            if line:
                line_counts[line] = line_counts.get(line, 0) + 1

    to_strip = {
        line for line, count in line_counts.items()
        if count / total >= HEURISTIC_THRESHOLD
    }

    if not to_strip:
        return pages

    return [
        "\n".join(l for l in text.splitlines() if l.strip() not in to_strip)
        for text in pages
    ]


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


def _ocr_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang="eng")


def _ocr_images(page: fitz.Page, doc: fitz.Document) -> str:
    texts: list[str] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pixmap = fitz.Pixmap(doc, xref)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image)
            if text.strip():
                texts.append(text.strip())
        except Exception as e:
            log.warning("Failed to OCR image xref=%d: %s", xref, e)
    return "\n".join(texts)
