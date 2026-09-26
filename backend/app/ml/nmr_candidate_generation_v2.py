"""Hybrid open-world candidate generation: local index + PubChem formula source.

The v1 generator enumerates only the read-only NMR index (exact canonical
formula).  This v2 module adds PubChem's ``fastformula`` endpoint as a second,
online formula-constrained source, so the candidate pool can contain
structures never seen by the local index.  The open-world contract is
unchanged: generation queries are formula-only (``case_id``, ``formula``,
``split``); truth binding happens only after generation.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from app.ml.nmr_candidate_generation_v1 import (
    NMRCandidateGenerationError,
    _validate_queries,
    _candidate_id,
    _coverage_counts,
    _validate_holder_gold,
    canonical_sha256,
)
from app.ml.nmr_evidence import canonical_formula


HYBRID_BUNDLE_SCHEMA_VERSION = "chemapp.nmr.roleless-candidate-bundle.v2"
HYBRID_COVERAGE_SCHEMA_VERSION = "chemapp.nmr.candidate-coverage-report.v2"
HYBRID_PROTOCOL_VERSION = "hybrid-index-formula-plus-pubchem-fastformula-v1"
HYBRID_MODE = "deployment_open_world_hybrid_formula_enumeration"
PUBCHM_URI = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula"
USER_AGENT = "chemapp-candidate-generation/1.0 (research)"
MAX_PROPERTIES_PER_FORMULA = 20000
MIN_INTERVAL_SECONDS = 0.9


class HybridGenerationError(NMRCandidateGenerationError):
    """Raised when hybrid candidate generation fails closed."""


class PubChemFormulaSource:
    """Formula-only PubChem candidate source with a persistent cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_properties: int = MAX_PROPERTIES_PER_FORMULA,
        timeout: int = 60,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_properties = int(max_properties)
        self.timeout = int(timeout)
        self.offline = bool(offline)
        self._lock = threading.Lock()
        self._last = 0.0

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = self._last + MIN_INTERVAL_SECONDS - now
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()

    @staticmethod
    def _http_raw(url: str, timeout: int) -> tuple[int, str]:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")

    def fetch_formula(self, formula: str) -> dict[str, Any]:
        """Return cached or freshly fetched PubChem properties for a formula."""
        cache_file = self.cache_dir / f"{formula}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("formula") == formula and "properties" in cached:
                    return cached
            except (OSError, ValueError):
                pass
        if self.offline:
            raise HybridGenerationError(
                f"PubChem cache miss for formula {formula} in offline mode"
            )
        encoded = quote(formula, safe="")
        url = (
            f"{PUBCHM_URI}/{encoded}/property/"
            "ConnectivitySMILES,InChIKey/JSON"
        )
        last_error: Exception | None = None
        for attempt in range(4):
            self._throttle()
            try:
                status, body = self._http_raw(url, self.timeout)
                if status == 202:
                    waiting = json.loads(body).get("Waiting", {})
                    list_key = str(waiting.get("ListKey") or "")
                    poll_url = f"{url}?listkey={list_key}"
                    status, body = self._http_raw(poll_url, timeout=30)
                    for _ in range(20):
                        if status != 202:
                            break
                        time.sleep(3)
                        status, body = self._http_raw(poll_url, timeout=30)
                if status != 200:
                    raise RuntimeError(
                        f"PubChem HTTP status {status} for formula {formula}"
                    )
                payload = json.loads(body)
                props = payload.get("PropertyTable", {}).get("Properties", [])
                payload = {
                    "formula": formula,
                    "properties": props[: self.max_properties],
                    "truncated": len(props) > self.max_properties,
                    "fetched_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(cache_file)
                return payload
            except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        raise HybridGenerationError(
            f"PubChem fetch failed for formula {formula}: {last_error}"
        )

    def candidate_smiles(
        self,
        formula: str,
        *,
        limit: int = 10000,
    ) -> list[str]:
        """Canonical, deduplicated, formula-validated SMILES for a formula."""
        payload = self.fetch_formula(formula)
        seen: set[str] = set()
        for prop in payload.get("properties", ()):
            raw = str(prop.get("ConnectivitySMILES") or "")
            if not raw:
                continue
            try:
                molecule = Chem.MolFromSmiles(raw)
                if molecule is None:
                    continue
                canonical = Chem.MolToSmiles(molecule)
                if canonical_formula(rdMolDescriptors.CalcMolFormula(molecule)) != formula:
                    continue
            except Exception:
                continue
            seen.add(canonical)
            if len(seen) >= limit:
                break
        return sorted(seen)

    def binding(self) -> dict[str, Any]:
        return {
            "source_uri": PUBCHM_URI,
            "cache_dir": str(self.cache_dir),
            "max_properties_per_formula": self.max_properties,
            "cache_file_count": len(list(self.cache_dir.glob("*.json"))),
        }


class HybridOpenWorldProvider:
    """``CandidateProvider`` implementation for the hybrid predictor."""

    provider_id = "hybrid-index-pubchem-formula-v1"

    def __init__(
        self,
        index_path: str | Path,
        pubchem: PubChemFormulaSource,
        *,
        pool_limit: int = 2000,
    ) -> None:
        self.index_path = Path(index_path)
        self.pubchem = pubchem
        self.pool_limit = int(pool_limit)
        self._index_by_formula: dict[str, list[str]] = {}

    def _index_candidates(self, formula: str) -> list[str]:
        if formula in self._index_by_formula:
            return self._index_by_formula[formula]
        smiles: list[str] = []
        if self.index_path.exists():
            try:
                from app.ml.nmr_candidate_generation_v1 import ReadonlyNMRIndex

                with ReadonlyNMRIndex(self.index_path) as index:
                    by_formula, _ = index.enumerate_formulas([formula])
                smiles = sorted(
                    {str(item["smiles"]) for item in by_formula.get(formula, ())}
                )
            except Exception:
                smiles = []
        self._index_by_formula[formula] = smiles
        return smiles

    def generate(
        self,
        *,
        formula: str | None,
        constraints: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        if not formula:
            return []
        try:
            canonical = canonical_formula(formula)
        except Exception:
            return []
        index_smiles = self._index_candidates(canonical)
        try:
            pubchem_smiles = self.pubchem.candidate_smiles(
                canonical, limit=max(1, self.pool_limit)
            )
        except HybridGenerationError:
            pubchem_smiles = []
        combined: list[str] = []
        seen: set[str] = set()
        for smiles in [*index_smiles, *pubchem_smiles]:
            if smiles in seen:
                continue
            seen.add(smiles)
            combined.append(smiles)
            if len(combined) >= min(int(limit), self.pool_limit):
                break
        return [
            {
                "smiles": smiles,
                "source": "hybrid-index-pubchem-formula-v1",
                "source_id": _candidate_id(smiles),
            }
            for smiles in combined
        ]


def build_hybrid_open_world_bundle(
    queries: Sequence[Mapping[str, Any]],
    *,
    index_bundle: Mapping[str, Any],
    pubchem: PubChemFormulaSource,
    pool_limit: int = 400,
) -> dict[str, Any]:
    """Build a v2 roleless bundle from index bundle rows + PubChem candidates."""
    normalised_queries = _validate_queries(queries)
    index_rows = {
        str(row["case_id"]): row for row in index_bundle.get("rows", ())
    }
    rows: list[dict[str, Any]] = []
    for query in normalised_queries:
        case_id = str(query["case_id"])
        formula = str(query["formula"])
        split = query.get("split")
        candidates: dict[str, str] = {}
        index_row = index_rows.get(case_id)
        if index_row is not None:
            for candidate in index_row.get("candidates", ()):
                candidates[str(candidate["smiles"])] = str(candidate["candidate_id"])
        try:
            for smiles in pubchem.candidate_smiles(formula, limit=pool_limit):
                candidates.setdefault(smiles, _candidate_id(smiles))
        except HybridGenerationError:
            pass
        ordered = sorted(
            candidates.items(),
            key=lambda item: item[1],
        )
        ordered = ordered[: int(pool_limit)]
        rows.append(
            {
                "case_id": case_id,
                "formula": formula,
                "split": split,
                "candidate_count": len(ordered),
                "candidates": [
                    {"candidate_id": candidate_id, "smiles": smiles}
                    for smiles, candidate_id in ordered
                ],
            }
        )
    core = {
        "schema_version": HYBRID_BUNDLE_SCHEMA_VERSION,
        "generation_mode": HYBRID_MODE,
        "generation_protocol": HYBRID_PROTOCOL_VERSION,
        "label_free": True,
        "index_binding": dict(index_bundle.get("index_binding") or {}),
        "pubchem_binding": pubchem.binding(),
        "query_count": len(rows),
        "rows": rows,
        "pool_limit": int(pool_limit),
    }
    return {**core, "generation_id": canonical_sha256(core)[:24]}


def evaluate_hybrid_coverage(
    bundle: Mapping[str, Any],
    holder_gold: Mapping[str, Any],
    *,
    recall_k: Sequence[int] = (1, 5, 10, 25, 50, 100),
) -> dict[str, Any]:
    """Join held identities after hybrid generation and report coverage."""
    if bundle.get("schema_version") != HYBRID_BUNDLE_SCHEMA_VERSION:
        raise HybridGenerationError("hybrid bundle schema changed")
    if bundle.get("label_free") is not True:
        raise HybridGenerationError("hybrid bundle must be label-free")
    gold_by_case = _validate_holder_gold(holder_gold)
    rows = bundle.get("rows")
    if not isinstance(rows, list):
        raise HybridGenerationError("hybrid bundle rows are invalid")
    row_cases = {str(row["case_id"]) for row in rows}
    if set(gold_by_case) != row_cases:
        raise HybridGenerationError("hybrid coverage join is not one-to-one")
    overall = _coverage_counts(rows, gold_by_case, recall_k)
    by_split: dict[str, Any] = {}
    split_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[str(row.get("split") or "unassigned")].append(row)
    for split, selected in sorted(split_rows.items()):
        by_split[split] = _coverage_counts(selected, gold_by_case, recall_k)
    return {
        "schema_version": HYBRID_COVERAGE_SCHEMA_VERSION,
        "evaluation_scope": "candidate_generator_coverage_only",
        "generation_mode": HYBRID_MODE,
        "roleless_bundle_sha256": canonical_sha256(bundle),
        "holder_mapping_sha256": holder_gold["mapping_sha256"],
        "join_performed_after_generation": True,
        "model_scores_consumed": False,
        "ranking_metrics_included": False,
        "overall": overall,
        "by_split": by_split,
    }


__all__ = [
    "HybridOpenWorldProvider",
    "PubChemFormulaSource",
    "build_hybrid_open_world_bundle",
    "evaluate_hybrid_coverage",
]
