from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, Literal

import pdfplumber
from docx import Document

logger = logging.getLogger("internmatch.parser")


def _clean_text(text: str) -> str:
    """
    Normalize extracted CV text.

    - Convert to lowercase.
    - Collapse multiple whitespace characters into single spaces.
    - Strip leading/trailing whitespace.
    """
    cleaned = re.sub(r"\s+", " ", text or "")
    return cleaned.strip().lower()


def extract_text_from_pdf(path: str | Path) -> str:
    """
    Extract and clean text from a PDF file using pdfplumber.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: for generic parsing issues.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Extracting text from PDF: %s", pdf_path)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_text = [page.extract_text(x_tolerance=2, y_tolerance=2) or "" for page in pdf.pages]
        raw = "\n".join(pages_text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to parse PDF %s", pdf_path)
        raise RuntimeError(f"Failed to parse PDF: {pdf_path}") from exc

    return _clean_text(raw)


def extract_text_from_docx(path: str | Path) -> str:
    """
    Extract and clean text from a DOCX file using python-docx.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: for generic parsing issues.
    """
    docx_path = Path(path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    logger.info("Extracting text from DOCX: %s", docx_path)
    try:
        document = Document(str(docx_path))
        raw = "\n".join(p.text for p in document.paragraphs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to parse DOCX %s", docx_path)
        raise RuntimeError(f"Failed to parse DOCX: {docx_path}") from exc

    return _clean_text(raw)


def extract_text_from_file(path: str | Path) -> str:
    """
    Convenience helper that dispatches based on file extension.

    Supports:
    - .pdf
    - .docx

    For local development and tests, `.txt` is also accepted and simply read
    as UTF‑8 text.
    """
    file_path = Path(path)
    ext: Literal[".pdf", ".docx", ".txt"] | str = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".txt":
        logger.info("Reading plain text CV file: %s", file_path)
        text = file_path.read_text(encoding="utf-8")
        return _clean_text(text)

    raise ValueError(f"Unsupported file extension for CV parsing: {ext}")


def parse_cv_with_ai(path: str | Path) -> Dict[str, Any]:
    """
    Parse a CV locally, then let Gemini structure the profile.

    This keeps file parsing in Python while making the parser module capable of
    returning the same AI-enriched profile used elsewhere in the app.
    """
    text = extract_text_from_file(path)

    try:
        from .ollama_helper import OllamaCareerEngine
    except ImportError:  # pragma: no cover - direct script execution
        from ollama_helper import OllamaCareerEngine

    engine = OllamaCareerEngine()
    profile = engine.extract_cv_profile(text)
    profile["cv_text"] = text
    profile["source"] = profile.get("source", "unavailable")
    return profile
