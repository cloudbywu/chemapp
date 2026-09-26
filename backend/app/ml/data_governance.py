"""Machine-checkable governance rules for third-party NMR assets.

This module is deliberately conservative.  It verifies recorded facts and
release gates, but it does not decide which software license the project owner
should adopt or offer legal advice.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROOT_LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.md",
    "COPYING.txt",
)
NMRSHIFTDB2_LICENSE_URI = (
    "https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/"
    "nmrshiftdb2datalicense.txt"
)
NMRSHIFTDB2_NOTICE = (
    "Contains information from nmrshiftdb2 (www.nmrshiftdb.org), "
    "which is made available here under the nmrshiftdb2 Database License."
)
ALLOWED_ASSET_KINDS = {
    "benchmark_manifest",
    "database",
    "dataset",
    "evaluation_report",
    "model_weights",
    "raw_snapshot",
    "retrieval_index",
}
ALLOWED_LINEAGE_STATES = {"complete", "legacy_incomplete"}
ALLOWED_PRESENCE_STATES = {"local_optional", "repository_required"}
ALLOWED_RELEASE_STATES = {
    "blocked_pending_legal_review",
    "blocked_pending_project_license",
    "internal_evaluation_only",
}
REQUIRED_RELEASE_GATE_IDS = frozenset(
    {
        "project_license",
        "content_rights_review",
        "derivative_database_offer",
        "legacy_lineage",
        "model_weight_classification",
        "external_smoke_scope",
    }
)
HUMAN_REVIEW_RELEASE_GATE_IDS = REQUIRED_RELEASE_GATE_IDS - {
    "external_smoke_scope"
}


@dataclass(frozen=True)
class Finding:
    """One deterministic governance finding."""

    severity: str
    code: str
    message: str
    asset_id: str | None = None


@dataclass(frozen=True)
class GovernanceReport:
    """Complete audit outcome suitable for JSON output and CI gating."""

    manifest_path: str
    integrity_ok: bool
    release_ready: bool
    release_ready_restricted_posture: bool
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = {
            severity: sum(
                finding.severity == severity for finding in self.findings
            )
            for severity in ("error", "blocker", "warning", "info")
        }
        return payload


def _finding(
    findings: list[Finding],
    severity: str,
    code: str,
    message: str,
    *,
    asset_id: str | None = None,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            code=code,
            message=message,
            asset_id=asset_id,
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and "\\" not in value
        and ".." not in path.parts
        and path.as_posix() == value
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load governance manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("governance manifest root must be an object")
    return value


def _validate_source_records(
    manifest: dict[str, Any],
    findings: list[Finding],
) -> dict[str, dict[str, Any]]:
    raw_sources = manifest.get("upstream_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        _finding(
            findings,
            "error",
            "SOURCES_MISSING",
            "upstream_sources must be a non-empty array",
        )
        return {}

    sources: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_sources):
        if not isinstance(value, dict):
            _finding(
                findings,
                "error",
                "SOURCE_INVALID",
                f"upstream_sources[{index}] must be an object",
            )
            continue
        source_id = value.get("id")
        if not isinstance(source_id, str) or not source_id:
            _finding(
                findings,
                "error",
                "SOURCE_ID_INVALID",
                f"upstream_sources[{index}] has no valid id",
            )
            continue
        if source_id in sources:
            _finding(
                findings,
                "error",
                "SOURCE_ID_DUPLICATE",
                f"duplicate upstream source id: {source_id}",
            )
            continue
        sources[source_id] = value

        for field in ("source_uri", "license_uri"):
            uri = value.get(field)
            parsed = urlparse(str(uri or ""))
            if parsed.scheme != "https" or not parsed.netloc:
                _finding(
                    findings,
                    "error",
                    "SOURCE_URI_INVALID",
                    f"{source_id}.{field} must be an absolute HTTPS URI",
                )
        for field in ("snapshot_sha256", "license_snapshot_sha256"):
            digest = value.get(field)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                _finding(
                    findings,
                    "error",
                    "SOURCE_HASH_INVALID",
                    f"{source_id}.{field} must be a lowercase SHA-256 digest",
                )
        if source_id == "nmrshiftdb2":
            if value.get("license_uri") != NMRSHIFTDB2_LICENSE_URI:
                _finding(
                    findings,
                    "error",
                    "NMRSHIFTDB2_LICENSE_URI_MISMATCH",
                    "nmrshiftdb2 must use the recorded official database-license URI",
                )
            if value.get("content_rights_status") != "record_level_rights_not_cleared":
                _finding(
                    findings,
                    "error",
                    "CONTENT_RIGHTS_STATUS_INVALID",
                    "nmrshiftdb2 individual-content rights must remain explicitly unresolved",
                )
    return sources


def _validate_asset_records(
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    project_license_pending: bool,
    findings: list[Finding],
) -> list[dict[str, Any]]:
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        _finding(
            findings,
            "error",
            "ASSETS_MISSING",
            "assets must be a non-empty array",
        )
        return []

    assets: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    for index, value in enumerate(raw_assets):
        if not isinstance(value, dict):
            _finding(
                findings,
                "error",
                "ASSET_INVALID",
                f"assets[{index}] must be an object",
            )
            continue
        asset_id = value.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            _finding(
                findings,
                "error",
                "ASSET_ID_INVALID",
                f"assets[{index}] has no valid id",
            )
            continue
        if asset_id in ids:
            _finding(
                findings,
                "error",
                "ASSET_ID_DUPLICATE",
                f"duplicate asset id: {asset_id}",
                asset_id=asset_id,
            )
            continue
        ids.add(asset_id)
        assets.append(value)

        path = value.get("path")
        if not _is_safe_relative_path(path):
            _finding(
                findings,
                "error",
                "ASSET_PATH_INVALID",
                "asset paths must be normalized, project-relative POSIX paths",
                asset_id=asset_id,
            )
        elif path in paths:
            _finding(
                findings,
                "error",
                "ASSET_PATH_DUPLICATE",
                f"duplicate asset path: {path}",
                asset_id=asset_id,
            )
        else:
            paths.add(str(path))

        if value.get("kind") not in ALLOWED_ASSET_KINDS:
            _finding(
                findings,
                "error",
                "ASSET_KIND_INVALID",
                f"unsupported asset kind: {value.get('kind')!r}",
                asset_id=asset_id,
            )
        if value.get("presence") not in ALLOWED_PRESENCE_STATES:
            _finding(
                findings,
                "error",
                "ASSET_PRESENCE_INVALID",
                f"unsupported presence policy: {value.get('presence')!r}",
                asset_id=asset_id,
            )
        if value.get("lineage_status") not in ALLOWED_LINEAGE_STATES:
            _finding(
                findings,
                "error",
                "ASSET_LINEAGE_INVALID",
                f"unsupported lineage status: {value.get('lineage_status')!r}",
                asset_id=asset_id,
            )
        digest = value.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            _finding(
                findings,
                "error",
                "ASSET_HASH_INVALID",
                "sha256 must be a lowercase 64-character digest",
                asset_id=asset_id,
            )
        source_ids = value.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            _finding(
                findings,
                "error",
                "ASSET_SOURCE_MISSING",
                "source_ids must be a non-empty array",
                asset_id=asset_id,
            )
        else:
            for source_id in source_ids:
                if source_id not in sources:
                    _finding(
                        findings,
                        "error",
                        "ASSET_SOURCE_UNKNOWN",
                        f"unknown source id: {source_id!r}",
                        asset_id=asset_id,
                    )

        release_state = value.get("public_distribution")
        if release_state not in ALLOWED_RELEASE_STATES:
            _finding(
                findings,
                "error",
                "ASSET_RELEASE_STATE_INVALID",
                f"unsupported public_distribution state: {release_state!r}",
                asset_id=asset_id,
            )
        elif (
            project_license_pending
            and "nmrshiftdb2" in (source_ids or [])
            and not str(release_state).startswith("blocked_")
        ):
            _finding(
                findings,
                "error",
                "ASSET_RELEASE_POLICY_TOO_PERMISSIVE",
                "assets cannot be releasable while the project license is pending",
                asset_id=asset_id,
            )
    return assets


def _audit_project_license(
    project_root: Path,
    manifest: dict[str, Any],
    findings: list[Finding],
) -> bool:
    project_license = manifest.get("project_license")
    if not isinstance(project_license, dict):
        _finding(
            findings,
            "error",
            "PROJECT_LICENSE_RECORD_MISSING",
            "project_license must be an object",
        )
        return True

    status = project_license.get("status")
    existing_license_files = [
        name for name in ROOT_LICENSE_NAMES if (project_root / name).is_file()
    ]
    if status == "pending_owner_selection":
        _finding(
            findings,
            "blocker",
            "PROJECT_LICENSE_UNSELECTED",
            (
                "The project owner has not selected and documented an "
                "OSI-approved software license required by nmrshiftdb2 section 4.5."
            ),
        )
        if existing_license_files:
            _finding(
                findings,
                "warning",
                "UNREGISTERED_LICENSE_FILE",
                (
                    "License-like files exist but the decision record still says "
                    f"pending: {', '.join(existing_license_files)}"
                ),
            )
        return True

    if status != "selected":
        _finding(
            findings,
            "error",
            "PROJECT_LICENSE_STATUS_INVALID",
            f"unsupported project license status: {status!r}",
        )
        return True

    license_file = project_license.get("license_file")
    if not _is_safe_relative_path(license_file):
        _finding(
            findings,
            "error",
            "PROJECT_LICENSE_PATH_INVALID",
            "selected project licenses require a normalized relative license_file",
        )
    elif not (project_root / str(license_file)).is_file():
        _finding(
            findings,
            "blocker",
            "PROJECT_LICENSE_FILE_MISSING",
            f"selected project license file is missing: {license_file}",
        )
    if not project_license.get("spdx_id"):
        _finding(
            findings,
            "blocker",
            "PROJECT_LICENSE_SPDX_MISSING",
            "selected project license must record an SPDX identifier",
        )
    if project_license.get("osi_approval_verified") is not True:
        _finding(
            findings,
            "blocker",
            "PROJECT_LICENSE_OSI_REVIEW_MISSING",
            "the owner/legal review must record OSI approval verification",
        )
    if not project_license.get("decision_record"):
        _finding(
            findings,
            "blocker",
            "PROJECT_LICENSE_DECISION_RECORD_MISSING",
            "selected project license must link to an owner-approved decision record",
        )
    return False


def _audit_required_notices(
    project_root: Path,
    findings: list[Finding],
) -> None:
    notice_path = project_root / "NOTICE"
    third_party_path = project_root / "THIRD_PARTY_DATA.md"
    if not notice_path.is_file():
        _finding(
            findings,
            "error",
            "NOTICE_MISSING",
            "root NOTICE is required for repository-level data attribution",
        )
    else:
        notice = notice_path.read_text(encoding="utf-8")
        if NMRSHIFTDB2_NOTICE not in notice:
            _finding(
                findings,
                "error",
                "NMRSHIFTDB2_NOTICE_MISSING",
                "NOTICE does not contain the upstream section 4.3 example notice",
            )
        if NMRSHIFTDB2_LICENSE_URI not in notice:
            _finding(
                findings,
                "error",
                "NMRSHIFTDB2_LICENSE_LINK_MISSING",
                "NOTICE does not contain the official database-license URI",
            )
    if not third_party_path.is_file():
        _finding(
            findings,
            "error",
            "THIRD_PARTY_DATA_MISSING",
            "THIRD_PARTY_DATA.md is required",
        )
    elif NMRSHIFTDB2_LICENSE_URI not in third_party_path.read_text(encoding="utf-8"):
        _finding(
            findings,
            "error",
            "THIRD_PARTY_LICENSE_LINK_MISSING",
            "THIRD_PARTY_DATA.md does not contain the official license URI",
        )


def _audit_release_gates(
    project_root: Path,
    manifest: dict[str, Any],
    findings: list[Finding],
) -> None:
    gates = manifest.get("release_gates")
    if not isinstance(gates, list) or not gates:
        _finding(
            findings,
            "error",
            "RELEASE_GATES_MISSING",
            "release_gates must be a non-empty array",
        )
        return
    ids: set[str] = set()
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict) or not isinstance(gate.get("id"), str):
            _finding(
                findings,
                "error",
                "RELEASE_GATE_INVALID",
                f"release_gates[{index}] must have an id",
            )
            continue
        gate_id = gate["id"]
        if gate_id in ids:
            _finding(
                findings,
                "error",
                "RELEASE_GATE_DUPLICATE",
                f"duplicate release gate id: {gate_id}",
            )
            continue
        ids.add(gate_id)
        status = gate.get("status")
        if status not in {"blocked", "satisfied"}:
            _finding(
                findings,
                "error",
                "RELEASE_GATE_STATUS_INVALID",
                f"release gate {gate_id} has invalid status {status!r}",
            )
        elif status == "blocked":
            _finding(
                findings,
                "blocker",
                f"RELEASE_GATE_{gate_id.upper()}",
                str(gate.get("reason") or f"release gate {gate_id} is blocked"),
            )
        elif gate_id in HUMAN_REVIEW_RELEASE_GATE_IDS:
            evidence = gate.get("evidence")
            decision_record = (
                evidence.get("decision_record")
                if isinstance(evidence, dict)
                else None
            )
            reviewer_role = (
                evidence.get("reviewer_role")
                if isinstance(evidence, dict)
                else None
            )
            basis = evidence.get("basis") if isinstance(evidence, dict) else None
            evidence_valid = (
                isinstance(evidence, dict)
                and _is_safe_relative_path(decision_record)
                and (project_root / str(decision_record)).is_file()
                and isinstance(reviewer_role, str)
                and bool(reviewer_role.strip())
                and isinstance(basis, str)
                and bool(basis.strip())
            )
            if not evidence_valid:
                _finding(
                    findings,
                    "error",
                    "SATISFIED_RELEASE_GATE_EVIDENCE_MISSING",
                    (
                        f"satisfied release gate {gate_id} requires an existing "
                        "project-relative decision_record, reviewer_role and basis"
                    ),
                )
    missing_ids = sorted(REQUIRED_RELEASE_GATE_IDS - ids)
    if missing_ids:
        _finding(
            findings,
            "error",
            "REQUIRED_RELEASE_GATES_MISSING",
            "required release gate ids are missing: " + ", ".join(missing_ids),
        )


def _audit_v2_sqlite_provenance(
    path: Path,
    source: dict[str, Any],
    findings: list[Finding],
    *,
    asset_id: str,
) -> None:
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve().as_posix()}?mode=ro",
            uri=True,
        )
        try:
            schema_row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            snapshot = connection.execute(
                """
                SELECT source_name, source_uri, sha256, byte_size, license_uri
                FROM source_snapshots
                """
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        _finding(
            findings,
            "error",
            "V2_INDEX_PROVENANCE_UNREADABLE",
            f"cannot read v2 provenance tables: {exc}",
            asset_id=asset_id,
        )
        return

    if schema_row != ("2",):
        _finding(
            findings,
            "error",
            "V2_INDEX_SCHEMA_MISMATCH",
            f"expected schema version 2, found {schema_row!r}",
            asset_id=asset_id,
        )
    if len(snapshot) != 1:
        _finding(
            findings,
            "error",
            "V2_INDEX_SOURCE_COUNT_INVALID",
            f"expected one source snapshot, found {len(snapshot)}",
            asset_id=asset_id,
        )
        return
    source_name, source_uri, digest, byte_size, license_uri = snapshot[0]
    expected = (
        source["id"],
        source["source_uri"],
        source["snapshot_sha256"],
        source["snapshot_bytes"],
        source["license_uri"],
    )
    actual = (source_name, source_uri, digest, byte_size, license_uri)
    if actual != expected:
        _finding(
            findings,
            "error",
            "V2_INDEX_SOURCE_MISMATCH",
            "v2 source_snapshots does not match the governance manifest",
            asset_id=asset_id,
        )


def _audit_local_assets(
    project_root: Path,
    assets: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    findings: list[Finding],
) -> None:
    for asset in assets:
        asset_id = str(asset["id"])
        path = project_root / str(asset.get("path", ""))
        if not path.is_file():
            severity = (
                "error"
                if asset.get("presence") == "repository_required"
                else "info"
            )
            _finding(
                findings,
                severity,
                "ASSET_MISSING" if severity == "error" else "OPTIONAL_ASSET_ABSENT",
                f"asset is not present: {asset.get('path')}",
                asset_id=asset_id,
            )
            continue
        actual_digest = _sha256_file(path)
        if actual_digest != asset.get("sha256"):
            _finding(
                findings,
                "error",
                "ASSET_HASH_MISMATCH",
                (
                    f"expected {asset.get('sha256')}, got {actual_digest} "
                    f"for {asset.get('path')}"
                ),
                asset_id=asset_id,
            )
        if asset_id == "nmr-spectral-index-v2":
            source = sources.get("nmrshiftdb2")
            if source is not None:
                _audit_v2_sqlite_provenance(
                    path,
                    source,
                    findings,
                    asset_id=asset_id,
                )


def audit_governance(
    project_root: str | Path,
    manifest_path: str | Path,
    *,
    verify_assets: bool = False,
) -> GovernanceReport:
    """Audit the manifest and optionally hash every present local asset."""

    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    manifest_file = manifest_file.resolve()
    manifest = _load_json_object(manifest_file)
    findings: list[Finding] = []

    if manifest.get("schema_version") != SCHEMA_VERSION:
        _finding(
            findings,
            "error",
            "SCHEMA_VERSION_INVALID",
            f"expected schema_version {SCHEMA_VERSION}",
        )
    project_license_pending = _audit_project_license(root, manifest, findings)
    sources = _validate_source_records(manifest, findings)
    assets = _validate_asset_records(
        manifest,
        sources,
        project_license_pending,
        findings,
    )
    _audit_required_notices(root, findings)
    _audit_release_gates(root, manifest, findings)
    if verify_assets:
        _audit_local_assets(root, assets, sources, findings)

    integrity_ok = not any(
        finding.severity == "error" for finding in findings
    )
    restricted_posture_ready = integrity_ok and not any(
        finding.severity == "blocker" for finding in findings
    )
    blocked_asset_ids = sorted(
        {
            str(
                asset.get("asset_id")
                or asset.get("path")
                or f"asset-{index}"
            )
            for index, asset in enumerate(assets)
            if str(asset.get("public_distribution") or "").startswith(
                "blocked_"
            )
            or asset.get("lineage_status") == "legacy_incomplete"
        }
    )
    for asset_id in blocked_asset_ids:
        _finding(
            findings,
            "blocker",
            "ASSET_NOT_DISTRIBUTION_READY",
            (
                f"asset {asset_id} is blocked from public distribution or "
                "has incomplete lineage; full release is not ready"
            ),
            asset_id=asset_id,
        )
    release_ready = restricted_posture_ready and not blocked_asset_ids
    return GovernanceReport(
        manifest_path=str(manifest_file),
        integrity_ok=integrity_ok,
        release_ready=release_ready,
        release_ready_restricted_posture=restricted_posture_ready,
        findings=tuple(findings),
    )


def fetch_sha256(
    uri: str,
    *,
    expected_host: str,
    timeout_seconds: float = 15.0,
    max_bytes: int = 128 * 1024,
) -> tuple[str, int]:
    """Fetch a small HTTPS policy document and return its byte hash.

    The explicit host and size limits make this suitable for the optional
    upstream-license drift check without turning the manifest into an arbitrary
    URL fetcher.
    """

    parsed = urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname != expected_host:
        raise ValueError("upstream URI does not match the pinned HTTPS host")
    request = Request(uri, headers={"User-Agent": "ChemApp-governance-audit/1"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"upstream document exceeds {max_bytes} bytes")
    return hashlib.sha256(body).hexdigest(), len(body)


def audit_upstream_license(
    manifest_path: str | Path,
    *,
    source_id: str = "nmrshiftdb2",
) -> Finding:
    """Check whether the current official license bytes match the frozen record."""

    manifest = _load_json_object(Path(manifest_path).resolve())
    source = next(
        (
            item
            for item in manifest.get("upstream_sources", [])
            if isinstance(item, dict) and item.get("id") == source_id
        ),
        None,
    )
    if source is None:
        return Finding(
            severity="error",
            code="UPSTREAM_SOURCE_NOT_FOUND",
            message=f"no upstream source record named {source_id!r}",
        )
    parsed = urlparse(str(source.get("license_uri", "")))
    try:
        digest, byte_size = fetch_sha256(
            str(source.get("license_uri", "")),
            expected_host=str(parsed.hostname or ""),
        )
    except (OSError, TimeoutError, ValueError) as exc:
        return Finding(
            severity="warning",
            code="UPSTREAM_LICENSE_UNAVAILABLE",
            message=f"could not verify current upstream license: {exc}",
        )
    expected = source.get("license_snapshot_sha256")
    expected_bytes = source.get("license_snapshot_bytes")
    if digest != expected or byte_size != expected_bytes:
        return Finding(
            severity="blocker",
            code="UPSTREAM_LICENSE_CHANGED",
            message=(
                "official license bytes changed from the frozen governance "
                f"record (expected {expected}/{expected_bytes}, "
                f"got {digest}/{byte_size}); repeat owner/legal review"
            ),
        )
    return Finding(
        severity="info",
        code="UPSTREAM_LICENSE_MATCH",
        message=f"official license matches the frozen SHA-256 {digest}",
    )
