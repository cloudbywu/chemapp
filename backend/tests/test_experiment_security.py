from __future__ import annotations

import io
import zipfile

import pytest

from app.ai import experiment_report_agent as report_agent


def _docx_with_document(xml: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types />")
        zf.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_docx_preflight_rejects_large_expansion(monkeypatch):
    monkeypatch.setattr(report_agent, "_MAX_DOCX_UNCOMPRESSED_BYTES", 128)
    content = _docx_with_document(b"<document>" + b"x" * 500 + b"</document>")
    with pytest.raises(RuntimeError, match="safe size"):
        report_agent.extract_handout_text("handout.docx", content)


def test_docx_fallback_rejects_entity_declarations():
    content = _docx_with_document(
        b'<!DOCTYPE x [<!ENTITY boom "payload">]><w:document '
        b'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:p><w:r><w:t>&boom;</w:t></w:r></w:p></w:document>"
    )
    with pytest.raises(RuntimeError, match="DTD and entity"):
        report_agent._extract_docx_ooxml_text(content)


def test_report_spectrum_ids_are_bounded_and_validated():
    with pytest.raises(ValueError, match="invalid characters"):
        report_agent.parse_ids_field('["valid-id", "../../secret"]')
    with pytest.raises(ValueError, match="At most 100"):
        report_agent.parse_ids_field(",".join(f"id-{index}" for index in range(101)))
