"""Runs full ingestion pipeline for all PDFs in a directory tree: extract → chunk → embed → store."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from supabase import create_client  # noqa: E402
import os  # noqa: E402

from backend.ingestion.chunker import chunk_document  # noqa: E402
from backend.ingestion.embed import embed_chunks  # noqa: E402
from backend.ingestion.extract import ExtractionError, extract_pdf  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def ingest_directory(pdf_dir: Path) -> None:
    pdfs = sorted(pdf_dir.rglob("*.pdf")) + sorted(pdf_dir.rglob("*.PDF"))
    if not pdfs:
        log.error("No PDFs found under %s", pdf_dir)
        sys.exit(1)

    log.info("Found %d PDFs under %s", len(pdfs), pdf_dir)

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    results: list[dict] = []

    for i, pdf_path in enumerate(pdfs, 1):
        rel = pdf_path.relative_to(pdf_dir)
        log.info("─" * 60)
        log.info("[%d/%d] %s", i, len(pdfs), rel)
        t0 = time.time()

        # ── extract ──────────────────────────────────────────────
        try:
            doc = extract_pdf(pdf_path, filename=pdf_path.name)
        except ExtractionError as exc:
            log.error("  EXTRACT FAILED: %s", exc)
            results.append({"file": str(rel), "status": "extract_error", "detail": str(exc)})
            continue
        except Exception as exc:
            log.error("  EXTRACT CRASHED: %s", exc, exc_info=True)
            results.append({"file": str(rel), "status": "extract_crash", "detail": str(exc)})
            continue

        pages = len(doc.pages)
        non_empty = sum(1 for p in doc.pages if p.text.strip())
        log.info("  Extracted %d pages (%d non-empty)", pages, non_empty)

        if non_empty == 0:
            log.warning("  SKIP — all pages empty after extraction (likely scanned + no Tesseract data)")
            results.append({"file": str(rel), "status": "empty", "detail": "0 non-empty pages"})
            continue

        # ── chunk ─────────────────────────────────────────────────
        try:
            chunks = chunk_document(doc)
        except Exception as exc:
            log.error("  CHUNK FAILED: %s", exc, exc_info=True)
            results.append({"file": str(rel), "status": "chunk_error", "detail": str(exc)})
            continue

        log.info("  Produced %d chunks", len(chunks))

        # ── embed + store ─────────────────────────────────────────
        try:
            added = embed_chunks(chunks, client)
        except Exception as exc:
            log.error("  EMBED FAILED: %s", exc, exc_info=True)
            results.append({"file": str(rel), "status": "embed_error", "detail": str(exc)})
            continue

        elapsed = time.time() - t0
        log.info("  ✓ Stored %d new chunks in %.1fs", added, elapsed)
        results.append({"file": str(rel), "status": "ok", "chunks": added, "elapsed": elapsed})

    # ── summary ───────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("SUMMARY")
    log.info("═" * 60)
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "empty"]
    errors  = [r for r in results if r["status"] not in ("ok", "empty")]

    for r in ok:
        log.info("  ✓ %-50s  %d chunks  (%.1fs)", r["file"], r["chunks"], r["elapsed"])
    for r in skipped:
        log.warning("  ⚠ %-50s  SKIPPED — %s", r["file"], r["detail"])
    for r in errors:
        log.error("  ✗ %-50s  %s — %s", r["file"], r["status"], r["detail"])

    log.info("─" * 60)
    log.info("OK: %d  |  Skipped: %d  |  Errors: %d  |  Total PDFs: %d",
             len(ok), len(skipped), len(errors), len(pdfs))
    total_chunks = sum(r.get("chunks", 0) for r in ok)
    log.info("Total chunks stored: %d", total_chunks)


if __name__ == "__main__":
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/sadiya/projects/PDF")
    if not pdf_dir.exists():
        log.error("Directory not found: %s", pdf_dir)
        sys.exit(1)
    ingest_directory(pdf_dir)
