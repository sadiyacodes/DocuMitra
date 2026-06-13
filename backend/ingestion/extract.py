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
            [line.strip() for line in lines[:HEADER_LINES]]
            + [line.strip() for line in lines[-FOOTER_LINES:]]
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
        "\n".join(line for line in text.splitlines() if line.strip() not in to_strip)
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
    """Render page as a 300-DPI PNG and return Tesseract OCR text."""
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang="eng")


def _ocr_images(page: fitz.Page, doc: fitz.Document) -> str:
    """OCR each embedded image on a native-text page; join non-blank results with newline."""
    texts: list[str] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pixmap = fitz.Pixmap(doc, xref)
            if pixmap.n > 4:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image)
            if text.strip():
                texts.append(text.strip())
        except Exception as e:
            log.warning("Failed to OCR image xref=%d: %s", xref, e)
    return "\n".join(texts)


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text from a PDF file, using native text or OCR as appropriate.

    Reads each page, detects scanned pages, strips repeated headers/footers,
    OCRs embedded images, normalizes text, and returns an ExtractedDocument
    with per-page PageContent objects.

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        ExtractedDocument with pdf_id, filename, and list of PageContent.

    Raises:
        ExtractionError: If the PDF cannot be opened or is password-protected.
    """
    file_bytes = path.read_bytes()
    pdf_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    filename = path.name

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ExtractionError(filename, e)

    try:
        if doc.is_encrypted:
            raise ExtractionError(filename, "PDF is password-protected")

        raw_texts: list[str] = []
        bboxes: list[tuple[float, float, float, float]] = []
        is_ocr_flags: list[bool] = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            native_text = "\n".join(
                span["text"]
                for block in blocks
                if block["type"] == 0
                for line in block["lines"]
                for span in line["spans"]
            )

            rect = page.rect
            bboxes.append((rect.x0, rect.y0, rect.x1, rect.y1))

            is_ocr = _detect_scanned(native_text)
            is_ocr_flags.append(is_ocr)

            if is_ocr:
                try:
                    page_text = _ocr_page(page)
                except pytesseract.TesseractNotFoundError as e:
                    raise ExtractionError(
                        filename,
                        f"Tesseract not installed: {e}. Install with: brew install tesseract",
                    )
            else:
                page_text = native_text
                image_text = _ocr_images(page, doc)
                if image_text:
                    page_text = page_text + "\n" + image_text

            raw_texts.append(page_text)

        stripped_texts = _strip_headers_footers(raw_texts)

        return ExtractedDocument(
            pdf_id=pdf_id,
            filename=filename,
            pages=[
                PageContent(
                    pdf_id=pdf_id,
                    filename=filename,
                    page_number=i + 1,
                    text=_normalize_text(text),
                    bbox=bbox,
                    is_ocr=is_ocr,
                )
                for i, (text, bbox, is_ocr) in enumerate(zip(stripped_texts, bboxes, is_ocr_flags))
            ],
        )
    finally:
        doc.close()
