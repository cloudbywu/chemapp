from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
import csv
import io
import json
import tempfile
import zipfile

from app.api.deps import get_store, require_admin
from app.api.routes.upload import _find_nmr_dir, _parse_and_store, _safe_extract_zip
from app.api.store import SpectrumDeleteConflict

router = APIRouter(prefix="/api/spectra", tags=["spectra"])

DATA_EXAMPLE_DIR = Path(__file__).resolve().parents[3].parent / "dataexample"
SpectrumId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class ExampleLoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Annotated[str, StringConstraints(min_length=1, max_length=512)]


class SpectrumIdsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SpectrumId] = Field(min_length=1, max_length=100)


def _csv_safe(value: Any) -> Any:
    """Prevent spreadsheet formula execution while preserving numeric cells."""
    if not isinstance(value, str):
        return value
    if value.startswith(("\t", "\r", "\n", "=", "+", "-", "@")):
        return "'" + value
    return value


@router.get("")
def list_spectra(
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    store = get_store()
    return [
        {
            **row,
            "summary": f"Spectrum(technique={row['technique']}, points={row['points']})",
        }
        for row in store.list_metadata(limit=limit, offset=offset)
    ]


@router.get("/examples")
def list_examples() -> list[dict[str, Any]]:
    examples = [
        ("UV-Vis", "紫外example/LA.txt", "UV-Vis LA sample"),
        ("UV-Vis", "紫外example/SE.txt", "UV-Vis SE sample"),
        ("UV-Vis", "紫外example/标准.txt", "UV-Vis calibration"),
        ("NMR", "HNMRexample.zip", "Bruker 1H NMR ZIP"),
        ("NMR", "26-7-11-cyj-sja_proton-1-1.jdf", "JEOL Delta 1H NMR JDF"),
        ("XRD", "XRDexample/CdS-1_Theta_2-Theta.asc", "XRD CdS pattern"),
        ("Fluorescence", "荧光example/em-lao(FDS).DX", "Fluorescence emission"),
        ("HPLC", "液相色谱example/-S-001.sirslt/-S-001.dx", "HPLC DAD export"),
    ]
    results: list[dict[str, Any]] = []
    for technique, rel, label in examples:
        path = DATA_EXAMPLE_DIR / rel
        if path.exists():
            results.append({
                "technique": technique,
                "label": label,
                "path": rel.replace("\\", "/"),
                "filename": path.name,
                "size": path.stat().st_size,
            })
    return results


@router.post("/examples/load", dependencies=[Depends(require_admin)])
def load_example(payload: ExampleLoadRequest) -> list[dict[str, Any]] | dict[str, Any]:
    rel_path = payload.path

    target = (DATA_EXAMPLE_DIR / rel_path).resolve()
    base = DATA_EXAMPLE_DIR.resolve()
    if target != base and base not in target.parents:
        raise HTTPException(400, "Invalid example path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Example file not found")

    try:
        if target.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(target) as zf:
                    _safe_extract_zip(zf, Path(extract_dir))
                nmr_dir = _find_nmr_dir(Path(extract_dir))
                if nmr_dir is None:
                    raise HTTPException(400, "No Bruker NMR data found in example ZIP")
                results = _parse_and_store(nmr_dir)
        else:
            results = _parse_and_store(target)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Example load failed: {e}")

    for item in results:
        item["name"] = target.name
    return results[0] if len(results) == 1 else results


@router.post("/export/csv.zip")
def export_batch_csv_zip(payload: SpectrumIdsRequest):
    ids = payload.ids

    store = get_store()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for sid in ids:
            stored = store.get(sid)
            if stored is None:
                manifest.append({"id": sid, "error": "not found"})
                continue
            spectrum = stored.spectrum
            stem = f"{spectrum.technique.value}_{sid[:8]}"
            csv_output = io.StringIO()
            writer = csv.writer(csv_output)
            channels = spectrum.parameters.get("channels") or []
            if channels:
                writer.writerow([
                    _csv_safe(spectrum.x_label),
                    *[_csv_safe(f"{ch.get('name')} ({ch.get('wavelength_nm')} nm)") for ch in channels],
                ])
                for i, x in enumerate(spectrum.x_data):
                    writer.writerow([float(x), *[float(ch["y_data"][i]) for ch in channels if i < len(ch["y_data"])]])
            else:
                writer.writerow([_csv_safe(spectrum.x_label), _csv_safe(spectrum.y_label)])
                for x, y in zip(spectrum.x_data, spectrum.y_data):
                    writer.writerow([float(x), float(y)])
            zf.writestr(f"{stem}.csv", csv_output.getvalue())
            if stored.result is not None:
                zf.writestr(
                    f"{stem}.result.json",
                    json.dumps(stored.result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2),
                )
            manifest.append({
                "id": sid,
                "technique": spectrum.technique.value,
                "source_file": spectrum.source_file,
                "points": spectrum.num_points,
                "has_result": stored.result is not None,
            })
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2),
        )

    zip_buffer.seek(0)
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=chemapp-export.zip"},
    )


@router.get("/{sid}")
def get_spectrum(sid: SpectrumId):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    d = stored.spectrum.to_dict(include_internal=False)
    d["id"] = sid
    d["spectrum_revision"] = stored.spectrum_revision
    d["result_revision"] = stored.result_revision
    return d


@router.get("/{sid}/csv")
def export_csv(sid: SpectrumId):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")

    spectrum = stored.spectrum
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([_csv_safe(spectrum.x_label), _csv_safe(spectrum.y_label)])
    for x, y in zip(spectrum.x_data, spectrum.y_data):
        writer.writerow([float(x), float(y)])

    output.seek(0)
    filename = f"{spectrum.technique.value}_{sid[:8]}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/{sid}", dependencies=[Depends(require_admin)])
def delete_spectrum(
    sid: SpectrumId,
    expected_spectrum_revision: Annotated[int, Query(ge=1)],
    expected_result_revision: Annotated[int, Query(ge=0)],
):
    store = get_store()
    try:
        removed = store.remove(
            sid,
            expected_spectrum_revision=expected_spectrum_revision,
            expected_result_revision=expected_result_revision,
        )
    except SpectrumDeleteConflict as exc:
        raise HTTPException(
            409,
            detail={
                "code": "revision_conflict",
                "message": (
                    "The spectrum or its analysis result changed after it was "
                    "loaded; reload before deleting."
                ),
                "current_spectrum_revision": exc.current_spectrum_revision,
                "current_result_revision": exc.current_result_revision,
            },
        ) from exc
    if not removed:
        raise HTTPException(404, f"Spectrum {sid} not found")
    return {"ok": True}
