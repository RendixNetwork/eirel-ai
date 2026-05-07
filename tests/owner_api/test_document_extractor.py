"""Tests for the document extractor."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestration.orchestrator.document_extractor import (
    MAX_RAW_BYTES,
    ExtractedDocument,
    extract_text,
    guess_format,
)


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "attachments"


def _read(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


# -- guess_format ----------------------------------------------------------


def test_guess_format_by_content_type():
    assert guess_format("anything", "application/pdf") == "pdf"
    assert guess_format("x", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "docx"
    assert guess_format("x", "text/csv") == "csv"
    assert guess_format("x", "application/json") == "json"
    assert guess_format("x", "text/plain") == "text"
    assert guess_format("x", "text/markdown") == "markdown"


def test_guess_format_by_extension():
    assert guess_format("foo.pdf", "") == "pdf"
    assert guess_format("foo.docx", "") == "docx"
    assert guess_format("foo.csv", "") == "csv"
    assert guess_format("foo.tsv", "") == "tsv"
    assert guess_format("foo.json", "") == "json"
    assert guess_format("foo.md", "") == "markdown"
    assert guess_format("foo.markdown", "") == "markdown"
    assert guess_format("foo.txt", "") == "text"
    assert guess_format("foo.log", "") == "text"


def test_guess_format_unsupported():
    assert guess_format("a.bin", "application/octet-stream") == "unsupported"
    assert guess_format("", "") == "unsupported"


# -- PDF (depends on pypdf) ------------------------------------------------


def test_extract_pdf_returns_metadata_for_blank_doc():
    pdf_bytes = _read("blank.pdf")
    doc = extract_text(pdf_bytes, filename="blank.pdf", content_type="application/pdf")
    assert doc.status in ("ok", "truncated")
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["n_pages"] == 1


def test_extract_pdf_unsupported_when_pypdf_missing(monkeypatch):
    """Simulate pypdf missing — extractor should report unsupported, not crash."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("pypdf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    doc = extract_text(b"%PDF-1.4 stub", filename="x.pdf", content_type="application/pdf")
    assert doc.status == "unsupported"
    assert doc.metadata.get("reason") == "pypdf_not_installed"


# -- DOCX ------------------------------------------------------------------


def test_extract_docx_text_and_table():
    raw = _read("sample.docx")
    doc = extract_text(
        raw,
        filename="sample.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert doc.status == "ok"
    assert "Hello world from DOCX." in doc.text
    assert "Second paragraph." in doc.text
    # Table content surfaced with pipe separators.
    assert "Alice" in doc.text
    assert "99" in doc.text
    assert doc.metadata["format"] == "docx"
    assert doc.metadata["n_paragraphs"] >= 2
    assert doc.metadata["n_tables"] >= 1


# -- CSV / TSV -------------------------------------------------------------


def test_extract_csv_keeps_header_and_rows():
    doc = extract_text(_read("sample.csv"), filename="sample.csv", content_type="text/csv")
    assert doc.status == "ok"
    lines = doc.text.split("\n")
    assert lines[0] == "name,score"
    assert "Alice,99" in lines
    assert "Bob,42" in lines
    assert doc.metadata["n_rows"] == 3
    assert doc.metadata["n_columns"] == 2


def test_extract_tsv_uses_tab_delimiter():
    raw = b"a\tb\tc\n1\t2\t3\n"
    doc = extract_text(raw, filename="x.tsv", content_type="")
    assert doc.status == "ok"
    assert doc.metadata["format"] == "tsv"
    assert "1\t2\t3" in doc.text


def test_extract_csv_caps_preview_at_100_rows():
    rows = ["col"] + [str(i) for i in range(500)]
    raw = "\n".join(rows).encode("utf-8")
    doc = extract_text(raw, filename="big.csv", content_type="text/csv")
    # n_rows reflects the full count; only 100 are kept in the preview.
    assert doc.metadata["n_rows"] == 501
    assert doc.metadata["n_preview_rows"] == 100
    assert doc.status == "truncated"


# -- JSON ------------------------------------------------------------------


def test_extract_json_pretty_prints():
    doc = extract_text(_read("sample.json"), filename="sample.json", content_type="application/json")
    assert doc.status == "ok"
    assert "hello" in doc.text
    assert "  " in doc.text  # indented
    assert doc.metadata["top_level_type"] == "dict"


def test_extract_json_invalid_returns_failed():
    doc = extract_text(b"{not json", filename="x.json", content_type="application/json")
    assert doc.status == "failed"
    assert "json" in doc.metadata["reason"].lower()


# -- Markdown / text -------------------------------------------------------


def test_extract_markdown_passthrough():
    doc = extract_text(_read("sample.md"), filename="sample.md", content_type="text/markdown")
    assert doc.status == "ok"
    assert "# Heading" in doc.text
    assert "**bold**" in doc.text
    assert doc.metadata["format"] == "markdown"


def test_extract_text_with_charset_detection():
    raw = "café".encode("latin-1")
    doc = extract_text(raw, filename="x.txt", content_type="text/plain")
    assert doc.status == "ok"
    # Either utf-8 replace path or latin-1 decode — caller sees something
    # that contains the letters.
    assert "caf" in doc.text


# -- Caps + unsupported ----------------------------------------------------


def test_extract_rejects_oversized_raw_input():
    big = b"x" * (MAX_RAW_BYTES + 1)
    doc = extract_text(big, filename="huge.txt", content_type="text/plain")
    assert doc.status == "failed"
    assert doc.metadata["reason"] == "raw_size_exceeds_limit"


def test_extract_truncates_extracted_text_to_max_chars():
    raw = ("line " * 100_000).encode("utf-8")
    doc = extract_text(raw, filename="big.txt", content_type="text/plain", max_chars=1_000)
    assert doc.status == "truncated"
    assert len(doc.text) <= 1_010  # 1000 + ellipsis


def test_extract_unsupported_format():
    doc = extract_text(b"\x00\x01\x02", filename="bin.bin", content_type="application/octet-stream")
    assert doc.status == "unsupported"
    assert doc.metadata["filename"] == "bin.bin"
