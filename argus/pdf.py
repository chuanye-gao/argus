"""PDF parsing helpers.

The benchmark core is dependency-light, but PDF extraction needs a parser.
PyMuPDF is used when available because it gives stable page text quickly.
"""

from __future__ import annotations

from pathlib import Path

from .models import PageText, ParsedDocument


class PdfParsingError(ValueError):
    """Raised when a PDF cannot be parsed into text."""


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Extract page text with document-level character offsets."""

    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PdfParsingError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PdfParsingError(f"Expected a .pdf file, got: {pdf_path}")

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PdfParsingError(
            "PDF parsing requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    pages: list[PageText] = []
    offset = 0
    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, 1):
                text = page.get_text("text") or ""
                start = offset
                end = start + len(text)
                pages.append(
                    PageText(
                        source_doc=pdf_path.name,
                        page=index,
                        text=text,
                        start_offset=start,
                        end_offset=end,
                    )
                )
                offset = end + 1
    except Exception as exc:  # pragma: no cover - parser-specific failures
        raise PdfParsingError(f"Could not parse PDF {pdf_path}: {exc}") from exc

    if not pages or not any(page.text.strip() for page in pages):
        raise PdfParsingError(f"No extractable text found in PDF: {pdf_path}")

    return ParsedDocument(source_doc=pdf_path.name, pages=tuple(pages))


def parse_text_file(path: str | Path) -> ParsedDocument:
    """Parse a plain text file as a single-page document for tests and demos."""

    text_path = Path(path)
    text = text_path.read_text(encoding="utf-8")
    page = PageText(
        source_doc=text_path.name,
        page=1,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )
    return ParsedDocument(source_doc=text_path.name, pages=(page,))
