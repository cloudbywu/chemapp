from __future__ import annotations

import logging
import os
from functools import partial

import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.ai.experiment_report_agent import (
    generate_experiment_report,
    markdown_to_docx_bytes,
    parse_ids_field,
)

router = APIRouter(prefix="/api/experiment-agent", tags=["experiment-agent"])
logger = logging.getLogger(__name__)
_MAX_HANDOUT_BYTES = int(
    os.environ.get("CHEMAPP_MAX_HANDOUT_BYTES", str(25 * 1024 * 1024))
)
_HANDOUT_CHUNK_SIZE = 1024 * 1024


async def _read_handout(file: UploadFile) -> bytes:
    content = bytearray()
    while True:
        chunk = await file.read(_HANDOUT_CHUNK_SIZE)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > _MAX_HANDOUT_BYTES:
            raise HTTPException(
                413,
                f"Handout exceeds the {_MAX_HANDOUT_BYTES // (1024 * 1024)} MB limit",
            )
    if not content:
        raise HTTPException(400, "Uploaded handout is empty")
    suffix = (file.filename or "").lower()
    if not suffix.endswith((".docx", ".md", ".txt")):
        raise HTTPException(415, "Handout must be a .docx, .md, or .txt file")
    return bytes(content)


def _validated_ids(value: str) -> list[str]:
    try:
        return parse_ids_field(value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _report_error(exc: ValueError) -> HTTPException:
    """Map report-generation input errors to the right status.

    A missing spectrum is a 404; any other invalid input (malformed ids,
    unreadable handout content, etc.) is a client error and maps to 422.
    """
    message = str(exc)
    if message.startswith("Spectrum ") and message.endswith(" not found"):
        return HTTPException(404, message)
    return HTTPException(422, message)


def _student_payload(
    name: str = "",
    student_id: str = "",
    college: str = "",
    major: str = "",
    teacher: str = "",
    location: str = "",
    date: str = "",
) -> dict[str, str]:
    return {
        "name": name,
        "student_id": student_id,
        "college": college,
        "major": major,
        "teacher": teacher,
        "location": location,
        "date": date,
    }


@router.post("/report")
async def generate_report_preview(
    handout: UploadFile = File(...),
    ids: str = Form("[]"),
    title: str = Form(""),
    name: str = Form(""),
    student_id: str = Form(""),
    college: str = Form(""),
    major: str = Form(""),
    teacher: str = Form(""),
    location: str = Form(""),
    date: str = Form(""),
    use_llm: bool = Form(False),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
):
    selected_ids = _validated_ids(ids)
    if not selected_ids:
        raise HTTPException(400, "Provide at least one spectrum/data id")
    content = await _read_handout(handout)
    try:
        report = await anyio.to_thread.run_sync(
            partial(
                generate_experiment_report,
                handout_filename=handout.filename or "handout.docx",
                handout_content=content,
                ids=selected_ids,
                student=_student_payload(name, student_id, college, major, teacher, location, date),
                title=title or None,
                use_llm=use_llm,
                model=model or None,
                api_key=api_key or None,
                base_url=base_url or None,
            )
        )
    except ValueError as exc:
        raise _report_error(exc) from exc
    except Exception as exc:
        logger.exception("Experiment report generation failed")
        raise HTTPException(422, "The experiment report could not be generated") from exc
    return report.to_dict()


@router.post("/report/markdown")
async def download_report_markdown(
    handout: UploadFile = File(...),
    ids: str = Form("[]"),
    title: str = Form(""),
    name: str = Form(""),
    student_id: str = Form(""),
    college: str = Form(""),
    major: str = Form(""),
    teacher: str = Form(""),
    location: str = Form(""),
    date: str = Form(""),
    use_llm: bool = Form(False),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
):
    selected_ids = _validated_ids(ids)
    if not selected_ids:
        raise HTTPException(400, "Provide at least one spectrum/data id")
    content = await _read_handout(handout)
    try:
        report = await anyio.to_thread.run_sync(
            partial(
                generate_experiment_report,
                handout_filename=handout.filename or "handout.docx",
                handout_content=content,
                ids=selected_ids,
                student=_student_payload(name, student_id, college, major, teacher, location, date),
                title=title or None,
                use_llm=use_llm,
                model=model or None,
                api_key=api_key or None,
                base_url=base_url or None,
            )
        )
    except ValueError as exc:
        raise _report_error(exc) from exc
    return PlainTextResponse(
        report.markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=experiment-report.md"},
    )


@router.post("/report/docx")
async def download_report_docx(
    handout: UploadFile = File(...),
    ids: str = Form("[]"),
    title: str = Form(""),
    name: str = Form(""),
    student_id: str = Form(""),
    college: str = Form(""),
    major: str = Form(""),
    teacher: str = Form(""),
    location: str = Form(""),
    date: str = Form(""),
    use_llm: bool = Form(False),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
):
    selected_ids = _validated_ids(ids)
    if not selected_ids:
        raise HTTPException(400, "Provide at least one spectrum/data id")
    content = await _read_handout(handout)
    try:
        report = await anyio.to_thread.run_sync(
            partial(
                generate_experiment_report,
                handout_filename=handout.filename or "handout.docx",
                handout_content=content,
                ids=selected_ids,
                student=_student_payload(name, student_id, college, major, teacher, location, date),
                title=title or None,
                use_llm=use_llm,
                model=model or None,
                api_key=api_key or None,
                base_url=base_url or None,
            )
        )
    except ValueError as exc:
        raise _report_error(exc) from exc
    try:
        docx = await anyio.to_thread.run_sync(
            partial(markdown_to_docx_bytes, report.markdown, report.title)
        )
    except Exception as exc:
        logger.exception("DOCX report rendering failed")
        raise HTTPException(500, "DOCX report rendering failed") from exc
    return StreamingResponse(
        iter([docx]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=experiment-report.docx"},
    )
