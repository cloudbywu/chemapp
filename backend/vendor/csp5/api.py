"""Programmatic CSP5 prediction API."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from ._runtime import cascade_core as core
from .model_registry import (
    ModelSpec,
    get_model_spec,
    get_model_spec_by_id,
    normalize_nucleus,
    resolve_model_weights,
)


_RUNTIME_DIR = Path(__file__).resolve().parent / "_runtime" / "Predict_SMILES_FF"
_PREPROCESSOR = _RUNTIME_DIR / "preprocessor_orig.p"
GAS_CONSTANT_KCAL_PER_MOL_K = 0.00198720425864083
DEFAULT_BOLTZMANN_TEMPERATURE_K = 298.15
MIN_CONFORMER_DISTANCE_ANGSTROM = 0.90

if not _PREPROCESSOR.exists():
    raise FileNotFoundError(f"Missing bundled preprocessor: {_PREPROCESSOR}")


@dataclass
class MoleculeRecord:
    molecule_id: int
    smiles: str
    mol: Mol
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConformerMolRecord:
    molecule_id: int
    mol: Mol
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    failures: List[str]
    model_id: str
    model_name: str
    nucleus: str
    device: str
    molecule_records: List[MoleculeRecord] = field(default_factory=list)
    conformer_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    conformer_mol_records: List[ConformerMolRecord] = field(default_factory=list)

    def to_compact_mapped_payload(self) -> Dict[str, Any]:
        return _compact_mapped_payload(self)

    def to_conformer_mapped_payload(self) -> Dict[str, Any]:
        return _conformer_mapped_payload(self)

    def write_conformers_sdf(self, path: str | Path) -> None:
        _write_conformers_sdf(self, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _mapped_smiles_explicit_h(mol: Mol) -> str:
    mapped_mol = Chem.Mol(mol)
    for atom in mapped_mol.GetAtoms():
        atom.SetAtomMapNum(int(atom.GetIdx()) + 1)
    mapped_smiles = Chem.MolToSmiles(mapped_mol)
    if not mapped_smiles:
        raise RuntimeError("Failed to generate mapped explicit-H SMILES")
    return mapped_smiles


def _prediction_row_payload(row: Mapping[str, Any], mol: Mol) -> Dict[str, Any]:
    atom_index = int(row["atom_index"])
    if atom_index < 0 or atom_index >= int(mol.GetNumAtoms()):
        raise ValueError(
            f"Prediction atom_index {atom_index} is out of range for molecule with {mol.GetNumAtoms()} atoms"
        )
    atom = mol.GetAtomWithIdx(atom_index)
    payload: Dict[str, Any] = {
        "atom_map": atom_index + 1,
        "element": atom.GetSymbol(),
        "shift_ppm": _jsonable(row["shift_ppm"]),
    }
    for key in ("shift_std_ppm", "shift_q10_ppm", "shift_q90_ppm"):
        if key in row:
            payload[key] = _jsonable(row[key])
    for key in sorted(row):
        if key.startswith("shift_q") and key.endswith("_ppm") and key not in payload:
            payload[key] = _jsonable(row[key])
    return payload


def _conformer_payload(
    rows: pd.DataFrame,
    mol: Mol,
) -> Dict[str, Any]:
    first = rows.iloc[0]
    payload: Dict[str, Any] = {
        "predictions": [
            _prediction_row_payload(row, mol)
            for row in rows.to_dict(orient="records")
        ],
    }
    for key in ("conformer_rank", "conformer_id", "conformer_energy", "conformer_energy_method"):
        if key in rows.columns:
            payload[key] = _jsonable(first[key])
    return payload


def _set_sdf_prop(mol: Mol, key: str, value: Any) -> None:
    clean_value = _jsonable(value)
    if clean_value is None:
        return
    mol.SetProp(str(key), str(clean_value))


def _sdf_mol_with_props(
    mol: Mol,
    *,
    molecule_id: int,
    smiles: str,
    result: PredictionResult,
    metadata: Mapping[str, Any],
) -> Mol:
    if mol.GetNumConformers() <= 0:
        raise ValueError(f"Cannot write molecule_id {molecule_id} to SDF because it has no conformer")

    sdf_mol = Chem.Mol(mol)
    _set_sdf_prop(sdf_mol, "input_smiles", smiles)
    _set_sdf_prop(sdf_mol, "molecule_id", molecule_id)
    _set_sdf_prop(sdf_mol, "nucleus", result.nucleus)
    _set_sdf_prop(sdf_mol, "model_id", result.model_id)
    _set_sdf_prop(sdf_mol, "model_name", result.model_name)

    for key in ("conformer_rank", "conformer_id", "conformer_energy", "conformer_energy_method"):
        if key in metadata:
            _set_sdf_prop(sdf_mol, key, metadata[key])

    if "conformer_id" not in metadata and sdf_mol.HasProp("ConfId"):
        _set_sdf_prop(sdf_mol, "conformer_id", sdf_mol.GetProp("ConfId"))

    return sdf_mol


def _conformer_sdf_mols(result: PredictionResult) -> List[Mol]:
    records_by_id = {int(record.molecule_id): record for record in result.molecule_records}
    if len(records_by_id) != len(result.molecule_records):
        raise ValueError("Duplicate molecule_id values in PredictionResult.molecule_records")

    if result.conformer_mol_records:
        sdf_mols: List[Mol] = []
        for conformer_record in result.conformer_mol_records:
            molecule_id = int(conformer_record.molecule_id)
            molecule_record = records_by_id.get(molecule_id)
            if molecule_record is None:
                raise ValueError(f"Missing molecule record for conformer molecule_id {molecule_id}")
            sdf_mols.append(
                _sdf_mol_with_props(
                    conformer_record.mol,
                    molecule_id=molecule_id,
                    smiles=molecule_record.smiles,
                    result=result,
                    metadata=conformer_record.metadata,
                )
            )
        return sdf_mols

    if not result.conformer_predictions.empty:
        raise ValueError("Cannot write conformer SDF without conformer_mol_records")

    return [
        _sdf_mol_with_props(
            record.mol,
            molecule_id=int(record.molecule_id),
            smiles=record.smiles,
            result=result,
            metadata=record.metadata,
        )
        for record in result.molecule_records
    ]


def _write_conformers_sdf(result: PredictionResult, path: str | Path) -> None:
    sdf_mols = _conformer_sdf_mols(result)
    if not sdf_mols:
        raise ValueError("No conformers are available to write")

    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(resolved_path))
    if writer is None:
        raise RuntimeError(f"Failed to create SDF writer: {resolved_path}")

    try:
        for mol in sdf_mols:
            writer.write(mol)
    except Exception as exc:
        raise RuntimeError(f"Failed to write conformer SDF: {resolved_path}") from exc
    finally:
        writer.close()


def _compact_mapped_payload(result: PredictionResult) -> Dict[str, Any]:
    if "molecule_id" not in result.predictions.columns and not result.predictions.empty:
        raise ValueError("Cannot build compact mapped output without molecule_id prediction column")

    records_by_id = {int(record.molecule_id): record for record in result.molecule_records}
    if len(records_by_id) != len(result.molecule_records):
        raise ValueError("Duplicate molecule_id values in PredictionResult.molecule_records")

    if result.predictions.empty:
        molecule_ids: List[int] = []
    else:
        molecule_ids = [
            int(value)
            for value in result.predictions["molecule_id"].drop_duplicates().tolist()
        ]
    missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in records_by_id]
    if missing:
        raise ValueError(f"Missing molecule records for molecule_id values: {missing}")

    molecules: List[Dict[str, Any]] = []
    for molecule_id in molecule_ids:
        record = records_by_id[molecule_id]
        rows = result.predictions[result.predictions["molecule_id"] == molecule_id]
        molecule_payload: Dict[str, Any] = {
            "molecule_id": molecule_id,
            "smiles": record.smiles,
            "mapped_smiles_explicit_h": _mapped_smiles_explicit_h(record.mol),
            "predictions": [
                _prediction_row_payload(row, record.mol)
                for row in rows.to_dict(orient="records")
            ],
        }
        for key, value in record.metadata.items():
            molecule_payload[str(key)] = _jsonable(value)
        molecules.append(molecule_payload)

    return {
        "model": {
            "id": result.model_id,
            "name": result.model_name,
            "nucleus": result.nucleus,
            "device": result.device,
        },
        "molecules": molecules,
        "failures": list(result.failures),
    }


def _conformer_mapped_payload(result: PredictionResult) -> Dict[str, Any]:
    if result.conformer_predictions.empty and result.predictions.empty:
        molecules: List[Dict[str, Any]] = []
    elif result.conformer_predictions.empty:
        if "molecule_id" not in result.predictions.columns:
            raise ValueError("Cannot build conformer output without molecule_id prediction column")
        records_by_id = {int(record.molecule_id): record for record in result.molecule_records}
        if len(records_by_id) != len(result.molecule_records):
            raise ValueError("Duplicate molecule_id values in PredictionResult.molecule_records")

        molecule_ids = [int(value) for value in result.predictions["molecule_id"].drop_duplicates().tolist()]
        missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in records_by_id]
        if missing:
            raise ValueError(f"Missing molecule records for molecule_id values: {missing}")

        molecules = []
        for molecule_id in molecule_ids:
            record = records_by_id[molecule_id]
            rows = result.predictions[result.predictions["molecule_id"] == molecule_id]
            molecule_payload: Dict[str, Any] = {
                "molecule_id": molecule_id,
                "smiles": record.smiles,
                "mapped_smiles_explicit_h": _mapped_smiles_explicit_h(record.mol),
                "conformers": [_conformer_payload(rows, record.mol)],
            }
            for key, value in record.metadata.items():
                molecule_payload[str(key)] = _jsonable(value)
            molecules.append(molecule_payload)
    else:
        if "molecule_id" not in result.conformer_predictions.columns:
            raise ValueError("Cannot build conformer output without molecule_id conformer prediction column")
        records_by_id = {int(record.molecule_id): record for record in result.molecule_records}
        molecule_ids = [
            int(value)
            for value in result.conformer_predictions["molecule_id"].drop_duplicates().tolist()
        ]
        missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in records_by_id]
        if missing:
            raise ValueError(f"Missing molecule records for conformer molecule_id values: {missing}")

        molecules = []
        for molecule_id in molecule_ids:
            record = records_by_id[molecule_id]
            conformer_rows = result.conformer_predictions[
                result.conformer_predictions["molecule_id"] == molecule_id
            ]
            group_columns = [
                key
                for key in ("conformer_rank", "conformer_id", "conformer_energy")
                if key in conformer_rows.columns
            ]
            if not group_columns:
                raise ValueError("Conformer predictions require conformer metadata columns")
            molecule_payload: Dict[str, Any] = {
                "molecule_id": molecule_id,
                "smiles": record.smiles,
                "mapped_smiles_explicit_h": _mapped_smiles_explicit_h(record.mol),
                "conformers": [
                    _conformer_payload(group, record.mol)
                    for _, group in conformer_rows.groupby(group_columns, sort=True, dropna=False)
                ],
            }
            for key, value in record.metadata.items():
                molecule_payload[str(key)] = _jsonable(value)
            molecules.append(molecule_payload)

    return {
        "model": {
            "id": result.model_id,
            "name": result.model_name,
            "nucleus": result.nucleus,
            "device": result.device,
        },
        "molecules": molecules,
        "failures": list(result.failures),
    }


def _atom_indices_for_nucleus(mol: Mol, nucleus: str) -> np.ndarray:
    atomic_num = 6 if nucleus == "13C" else 1
    return np.asarray(
        [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) == atomic_num],
        dtype=np.int64,
    )


def _geometry_failure_reason(mol: Mol) -> str | None:
    if int(mol.GetNumConformers()) < 1:
        return "geometry_no_conformer"
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if coordinates.shape != (int(mol.GetNumAtoms()), 3):
        return "geometry_shape_invalid"
    if not np.isfinite(coordinates).all():
        return "geometry_nonfinite"
    minimum_squared = math.inf
    for atom_index in range(int(coordinates.shape[0]) - 1):
        deltas = coordinates[atom_index + 1 :] - coordinates[atom_index]
        squared = np.einsum("ij,ij->i", deltas, deltas)
        if squared.size:
            minimum_squared = min(minimum_squared, float(np.min(squared)))
    if minimum_squared < float(MIN_CONFORMER_DISTANCE_ANGSTROM**2):
        minimum_distance = math.sqrt(max(minimum_squared, 0.0))
        return (
            f"geometry_close_contact_{minimum_distance:.4f}A"
            f"_lt_{MIN_CONFORMER_DISTANCE_ANGSTROM:.2f}A"
        )
    return None


class _ReleaseGraphPreprocessor:
    """Build the exact neighbour graphs used to train the packaged models."""

    def __init__(self, legacy_preprocessor: object, *, n_neighbors: int, cutoff: float) -> None:
        atom_tokenizer = getattr(legacy_preprocessor, "atom_tokenizer", None)
        atom_features = getattr(legacy_preprocessor, "atom_features", None)
        if not callable(atom_tokenizer) or not callable(atom_features):
            raise RuntimeError("Bundled preprocessor has no usable atom tokenizer")
        if int(n_neighbors) <= 0:
            raise ValueError(f"graph n_neighbors must be positive, got {n_neighbors}")
        if not math.isfinite(float(cutoff)) or float(cutoff) <= 0.0:
            raise ValueError(f"graph cutoff must be finite and positive, got {cutoff}")
        self.atom_tokenizer = atom_tokenizer
        self.atom_features = atom_features
        self.n_neighbors = int(n_neighbors)
        self.cutoff = float(cutoff)

    def _edge_indices(self, coordinates: np.ndarray) -> np.ndarray:
        n_atoms = int(coordinates.shape[0])
        if n_atoms == 1:
            return np.asarray([[0, 0]], dtype=np.int64)
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        edges: List[Tuple[int, int]] = []
        for source in range(n_atoms):
            neighbors: List[int] = []
            for destination_raw in np.argsort(distances[source])[1:]:
                destination = int(destination_raw)
                if float(distances[source, destination]) >= self.cutoff:
                    break
                neighbors.append(destination)
                if len(neighbors) >= self.n_neighbors:
                    break
            if not neighbors:
                neighbors = [source]
            edges.extend((source, destination) for destination in neighbors)
        return np.asarray(edges, dtype=np.int64)

    def predict(self, entries: Iterable[Tuple[Mol, np.ndarray]]) -> Iterable[Dict[str, Any]]:
        for mol, target_atom_indices_raw in entries:
            if mol is None or int(mol.GetNumConformers()) != 1:
                raise ValueError("Release graph preprocessing requires one molecular conformer")
            n_atoms = int(mol.GetNumAtoms())
            coordinates = np.asarray(
                mol.GetConformer().GetPositions(),
                dtype=np.float32,
            )
            if coordinates.shape != (n_atoms, 3) or not np.isfinite(coordinates).all():
                raise ValueError("Release graph preprocessing received invalid coordinates")
            target_atom_indices = np.asarray(target_atom_indices_raw, dtype=np.int64)
            if target_atom_indices.ndim != 1 or target_atom_indices.size == 0:
                raise ValueError("Release graph preprocessing requires target atom indices")
            if (
                np.any(target_atom_indices < 0)
                or np.any(target_atom_indices >= n_atoms)
                or np.unique(target_atom_indices).size != target_atom_indices.size
            ):
                raise ValueError("Release graph preprocessing received invalid target atom indices")
            atom_index = np.full(n_atoms, -1, dtype=np.int64)
            atom_index[target_atom_indices] = np.arange(
                target_atom_indices.size,
                dtype=np.int64,
            )
            node_attributes = np.asarray(
                [
                    int(self.atom_tokenizer(self.atom_features(atom)))
                    for atom in mol.GetAtoms()
                ],
                dtype=np.int64,
            )
            yield {
                "n_atom": n_atoms,
                "n_pro": int(target_atom_indices.size),
                "node_attributes": node_attributes,
                "node_coordinates": coordinates,
                "edge_indices": self._edge_indices(coordinates),
                "atom_index": atom_index,
            }


@lru_cache(maxsize=16)
def _load_artifacts_cached(model_id: str, device: str):
    spec = get_model_spec_by_id(model_id)
    weights_path = resolve_model_weights(spec)
    resolved_device = core._resolve_device(str(device), _RUNTIME_DIR)
    artifacts = core.load_artifacts(
        model_dir=_RUNTIME_DIR,
        weights_path=weights_path,
        preprocessor_path=_PREPROCESSOR,
        device=resolved_device,
        output_scale=float(spec.output_scale),
        output_bias=float(spec.output_bias),
        output_affine_source="csp5_default",
        output_quantile_scale=float(spec.output_quantile_scale),
    )
    legacy_preprocessor = getattr(artifacts, "preprocessor", None)
    if legacy_preprocessor is None:
        raise RuntimeError(f"Loaded artifacts for {spec.model_name} have no graph preprocessor")
    artifacts.preprocessor = _ReleaseGraphPreprocessor(
        legacy_preprocessor,
        n_neighbors=int(spec.graph_n_neighbors),
        cutoff=float(spec.graph_cutoff_angstrom),
    )
    return artifacts


def _dry_artifacts(spec: ModelSpec) -> SimpleNamespace:
    return SimpleNamespace(
        output_dim=int(spec.output_dim),
        device="cpu",
        output_quantile_scale=float(spec.output_quantile_scale),
    )


def _normalize_smiles_input(smiles: str | Sequence[str] | Iterable[str]) -> List[str]:
    if isinstance(smiles, str):
        items = [smiles]
    else:
        items = [str(s).strip() for s in list(smiles)]
    items = [s for s in items if s]
    if not items:
        raise ValueError("No SMILES were provided.")
    return items


def _normalize_mols_input(mols: Sequence[Mol] | Iterable[Mol]) -> List[Mol]:
    values = list(mols)
    if not values:
        raise ValueError("No molecules were provided.")
    for idx, mol in enumerate(values):
        if mol is None:
            raise ValueError(f"Molecule at index {idx} is None")
        if not isinstance(mol, Mol):
            raise TypeError(f"Molecule at index {idx} is not an RDKit Mol: {type(mol)!r}")
    return values


def _mol_to_input_smiles(mol: Mol) -> str:
    no_h = Chem.RemoveHs(mol, sanitize=False)
    smiles = Chem.MolToSmiles(no_h)
    if not smiles:
        raise RuntimeError("Failed to derive SMILES from provided molecule")
    return smiles


def _molecule_label_from_mol(mol: Mol) -> str:
    if mol.HasProp("input_smiles"):
        value = str(mol.GetProp("input_smiles")).strip()
        if value:
            return value
    if mol.HasProp("_Name"):
        value = str(mol.GetProp("_Name")).strip()
        if value:
            return value
    return _mol_to_input_smiles(mol)


def _load_molecule_file(path: str | Path) -> Tuple[List[Mol], List[str]]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"molecule_file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".mol":
        mol = Chem.MolFromMolFile(str(resolved_path), removeHs=False, sanitize=True)
        if mol is None:
            raise RuntimeError(f"Failed to parse molfile: {resolved_path}")
        return [mol], [_molecule_label_from_mol(mol)]

    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(resolved_path), removeHs=False, sanitize=True)
        mols: List[Mol] = []
        smiles: List[str] = []
        for idx, mol in enumerate(supplier, start=1):
            if mol is None:
                raise RuntimeError(f"Failed to parse molecule {idx} from SDF: {resolved_path}")
            mols.append(mol)
            smiles.append(_molecule_label_from_mol(mol))
        if not mols:
            raise ValueError(f"No molecules found in molecule_file: {resolved_path}")
        return mols, smiles

    raise ValueError(f"Unsupported molecule_file extension {suffix!r}; expected .mol, .sdf, or .sd")


def _optimize_regenerated_geometry(mol: Mol, *, conf_id: int, ff_max_iters: int) -> Tuple[float | None, str | None]:
    mmff_has = AllChem.MMFFHasAllMoleculeParams(mol)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol) if mmff_has else None
    uff_has = AllChem.UFFHasAllMoleculeParams(mol) if hasattr(AllChem, "UFFHasAllMoleculeParams") else True

    if mmff_has:
        try:
            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=int(ff_max_iters), confId=int(conf_id))
        except Exception:
            status = -1
        if status in (0, 1):
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=int(conf_id))
                return float(ff.CalcEnergy()), "MMFF"
            except Exception:
                return None, "MMFF"

    if uff_has:
        try:
            status = AllChem.UFFOptimizeMolecule(mol, maxIters=int(ff_max_iters), confId=int(conf_id))
        except Exception:
            status = -1
        if status in (0, 1):
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
                return float(ff.CalcEnergy()), "UFF"
            except Exception:
                return None, "UFF"

    raise RuntimeError("Failed to optimize regenerated geometry with MMFF or UFF")


def _regenerate_molecule_geometry(
    mol: Mol,
    *,
    molecule_label: str,
    max_embed_tries: int,
    prune_rms_thresh: float,
    ff_max_iters: int,
) -> Mol:
    regenerated = Chem.AddHs(Chem.Mol(mol))
    regenerated.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    if float(prune_rms_thresh) > 0.0:
        params.pruneRmsThresh = float(prune_rms_thresh)
    seed_material = f"{molecule_label}\0{regenerated.GetNumAtoms()}".encode("utf-8")
    base_seed = int(hashlib.md5(seed_material).hexdigest()[:8], 16) & 0x7FFFFFFF
    for attempt in range(max(1, int(max_embed_tries))):
        regenerated.RemoveAllConformers()
        params.randomSeed = int((base_seed + attempt) & 0x7FFFFFFF)
        params.useRandomCoords = attempt > 0
        if AllChem.EmbedMolecule(regenerated, params=params) == 0:
            _optimize_regenerated_geometry(
                regenerated,
                conf_id=0,
                ff_max_iters=int(ff_max_iters),
            )
            regenerated.SetProp("ConfId", "0")
            return regenerated

    raise RuntimeError(f"Failed to regenerate geometry for molecule: {molecule_label}")


def _regenerate_molecule_geometries(
    mols: Sequence[Mol],
    smiles: Sequence[str],
    *,
    max_embed_tries: int,
    prune_rms_thresh: float,
    ff_max_iters: int,
) -> List[Mol]:
    regenerated: List[Mol] = []
    for idx, (mol, smi) in enumerate(zip(mols, smiles), start=1):
        try:
            regenerated.append(
                _regenerate_molecule_geometry(
                    mol,
                    molecule_label=str(smi),
                    max_embed_tries=int(max_embed_tries),
                    prune_rms_thresh=float(prune_rms_thresh),
                    ff_max_iters=int(ff_max_iters),
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to regenerate geometry for molecule {idx} ({smi})") from exc
    return regenerated


def _boltzmann_weights(
    energies: Sequence[Any],
    *,
    temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
) -> np.ndarray:
    if float(temperature_k) <= 0.0:
        raise ValueError("Boltzmann temperature must be positive")
    energy_values = np.asarray([float(value) for value in energies], dtype=np.float64)
    if energy_values.size == 0:
        raise ValueError("Cannot Boltzmann-average an empty conformer ensemble")
    if not np.isfinite(energy_values).all():
        raise ValueError("All conformer_energy values must be finite for Boltzmann averaging")
    shifted = energy_values - float(np.min(energy_values))
    weights = np.exp(-shifted / (GAS_CONSTANT_KCAL_PER_MOL_K * float(temperature_k)))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Failed to calculate finite Boltzmann weights")
    return weights / total


def _require_conformer_energies(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        energy = record.get("energy")
        if energy is None:
            raise ValueError("Every conformer requires finite conformer_energy for Boltzmann averaging")
        try:
            energy_value = float(energy)
        except (TypeError, ValueError) as exc:
            raise ValueError("Every conformer requires finite conformer_energy for Boltzmann averaging") from exc
        if not np.isfinite(energy_value):
            raise ValueError("Every conformer requires finite conformer_energy for Boltzmann averaging")


def _boltzmann_average_prediction_rows(
    predictions: pd.DataFrame,
    *,
    temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    required_columns = {"molecule_id", "smiles", "atom_index", "shift_ppm", "conformer_energy"}
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Cannot Boltzmann-average predictions without columns: {sorted(missing)}")

    output_rows: List[Dict[str, Any]] = []
    for (_, _), group in predictions.groupby(["molecule_id", "atom_index"], sort=False):
        weights = _boltzmann_weights(group["conformer_energy"].tolist(), temperature_k=float(temperature_k))
        shift_values = group["shift_ppm"].astype(float).to_numpy(dtype=np.float64)
        averaged_shift = float(np.dot(weights, shift_values))
        first = group.iloc[0]
        output: Dict[str, Any] = {
            "molecule_id": int(first["molecule_id"]),
            "smiles": str(first["smiles"]),
            "atom_index": int(first["atom_index"]),
            "shift_ppm": round(averaged_shift, 2),
            "conformer_count": int(len(group)),
            "conformer_energy_min": round(float(np.min(group["conformer_energy"].astype(float))), 6),
            "boltzmann_temperature_k": float(temperature_k),
        }
        if "shift_q10_ppm" in group:
            output["shift_q10_ppm"] = round(
                float(np.dot(weights, group["shift_q10_ppm"].astype(float).to_numpy(dtype=np.float64))), 2
            )
        if "shift_q90_ppm" in group:
            output["shift_q90_ppm"] = round(
                float(np.dot(weights, group["shift_q90_ppm"].astype(float).to_numpy(dtype=np.float64))), 2
            )
        for key in sorted(group.columns):
            if key.startswith("shift_q") and key.endswith("_ppm") and key not in output:
                output[key] = round(
                    float(np.dot(weights, group[key].astype(float).to_numpy(dtype=np.float64))),
                    2,
                )
        if "shift_std_ppm" in group:
            std_values = group["shift_std_ppm"].astype(float).to_numpy(dtype=np.float64)
            variance = float(np.dot(weights, std_values**2 + (shift_values - averaged_shift) ** 2))
            output["shift_std_ppm"] = round(math.sqrt(max(variance, 0.0)), 4)
        output_rows.append(output)

    return pd.DataFrame(output_rows)


def _predict_from_mols(
    mols: Sequence[Mol],
    smiles: Sequence[str],
    *,
    nucleus: str,
    spec: ModelSpec,
    artifacts,
    resolved_device: str,
    batch_size: int,
    dry_run: bool,
    extra_cols: Mapping[str, Sequence[Any]] | None = None,
) -> PredictionResult:
    no_target_reason = "no_carbon" if nucleus == "13C" else "no_proton"
    failures: List[str] = []

    mols_ready: List[Mol] = []
    atom_indices: List[np.ndarray] = []
    smiles_ready: List[str] = []
    extra_cols_ready: Dict[str, List[Any]] = {}

    if extra_cols:
        extra_cols_lists: Dict[str, List[Any]] = {}
        for key, values in extra_cols.items():
            values_list = list(values)
            if len(values_list) != len(mols):
                raise ValueError(
                    f"extra_cols[{key!r}] length mismatch: got {len(values_list)}, expected {len(mols)}"
                )
            key_str = str(key)
            extra_cols_lists[key_str] = values_list
            extra_cols_ready[key_str] = []
    else:
        extra_cols_lists = {}

    for idx, (mol, smi) in enumerate(zip(mols, smiles)):
        target_indices = _atom_indices_for_nucleus(mol, nucleus)
        if target_indices.size == 0:
            failures.append(f"{no_target_reason}\t{smi}")
            continue
        if not dry_run:
            geometry_failure = _geometry_failure_reason(mol)
            if geometry_failure is not None:
                failures.append(f"{geometry_failure}\t{smi}")
                continue
        mols_ready.append(mol)
        atom_indices.append(target_indices)
        smiles_ready.append(str(smi))
        for key, values_list in extra_cols_lists.items():
            extra_cols_ready[key].append(values_list[idx])

    if not atom_indices:
        empty = pd.DataFrame(
            columns=["molecule_id", "smiles", "atom_index", "shift_ppm", "nucleus", "model_id", "model_name"]
        )
        return PredictionResult(
            predictions=empty,
            failures=failures,
            model_id=spec.model_id,
            model_name=spec.model_name,
            nucleus=nucleus,
            device=resolved_device,
            molecule_records=[],
        )

    molecule_records: List[MoleculeRecord] = []
    for molecule_id, (mol, smi) in enumerate(zip(mols_ready, smiles_ready)):
        metadata: Dict[str, Any] = {}
        for key, values in extra_cols_ready.items():
            metadata[key] = values[molecule_id]
        molecule_records.append(
            MoleculeRecord(
                molecule_id=molecule_id,
                smiles=str(smi),
                mol=mol,
                metadata=metadata,
            )
        )

    df_pred, preprocess_failures = core._predict_from_mols_ready(
        mols_ready=mols_ready,
        atom_indices=atom_indices,
        mols_smiles=smiles_ready,
        artifacts=artifacts,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=extra_cols_ready if extra_cols_ready else None,
        molecule_ids_per_mol=[record.molecule_id for record in molecule_records],
    )
    failures.extend(preprocess_failures)

    if not df_pred.empty:
        df_pred["nucleus"] = nucleus
        df_pred["model_id"] = spec.model_id
        df_pred["model_name"] = spec.model_name

    return PredictionResult(
        predictions=df_pred,
        failures=list(failures),
        model_id=spec.model_id,
        model_name=spec.model_name,
        nucleus=nucleus,
        device=resolved_device,
        molecule_records=molecule_records,
    )


def _predict_from_smiles_conformer_records(
    records: Sequence[Mapping[str, Any]],
    smiles_list: Sequence[str],
    *,
    nucleus: str,
    spec: ModelSpec,
    artifacts,
    resolved_device: str,
    batch_size: int,
    dry_run: bool,
    temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
) -> PredictionResult:
    _require_conformer_energies(records)
    no_target_reason = "no_carbon" if nucleus == "13C" else "no_proton"
    failures: List[str] = []
    records_by_input_index: Dict[int, List[Mapping[str, Any]]] = {}
    for record in records:
        if "input_index" not in record:
            raise ValueError("Conformer records must include input_index for Boltzmann averaging")
        records_by_input_index.setdefault(int(record["input_index"]), []).append(record)

    molecule_records: List[MoleculeRecord] = []
    mols_ready: List[Mol] = []
    atom_indices: List[np.ndarray] = []
    smiles_ready: List[str] = []
    molecule_ids: List[int] = []
    conformer_mol_records: List[ConformerMolRecord] = []
    extra_cols: Dict[str, List[Any]] = {
        "conformer_rank": [],
        "conformer_id": [],
        "conformer_energy": [],
        "conformer_energy_method": [],
    }

    for input_index, smi in enumerate(smiles_list):
        conformers = records_by_input_index.get(int(input_index), [])
        if not conformers:
            continue
        conformers = sorted(conformers, key=lambda record: int(record.get("conformer_rank", 0)))
        representative_mol = conformers[0]["mol"]
        representative_indices = _atom_indices_for_nucleus(representative_mol, nucleus)
        if representative_indices.size == 0:
            failures.append(f"{no_target_reason}\t{smi}")
            continue

        molecule_id = len(molecule_records)
        molecule_records.append(
            MoleculeRecord(
                molecule_id=molecule_id,
                smiles=str(smi),
                mol=representative_mol,
                metadata={
                    "conformer_count": len(conformers),
                    "boltzmann_temperature_k": float(temperature_k),
                },
            )
        )
        for conformer in conformers:
            mol = conformer["mol"]
            target_indices = _atom_indices_for_nucleus(mol, nucleus)
            if target_indices.size == 0:
                raise ValueError(f"Conformer for {smi} has no target atoms for nucleus {nucleus}")
            mols_ready.append(mol)
            atom_indices.append(target_indices)
            smiles_ready.append(str(smi))
            molecule_ids.append(molecule_id)
            conformer_metadata = {
                "conformer_rank": conformer.get("conformer_rank"),
                "conformer_id": conformer.get("conformer_id"),
                "conformer_energy": conformer.get("energy"),
                "conformer_energy_method": conformer.get("energy_method"),
            }
            conformer_mol_records.append(
                ConformerMolRecord(
                    molecule_id=molecule_id,
                    mol=mol,
                    metadata=conformer_metadata,
                )
            )
            extra_cols["conformer_rank"].append(conformer_metadata["conformer_rank"])
            extra_cols["conformer_id"].append(conformer_metadata["conformer_id"])
            extra_cols["conformer_energy"].append(conformer_metadata["conformer_energy"])
            extra_cols["conformer_energy_method"].append(conformer_metadata["conformer_energy_method"])

    if not mols_ready:
        empty = pd.DataFrame(
            columns=["molecule_id", "smiles", "atom_index", "shift_ppm", "nucleus", "model_id", "model_name"]
        )
        return PredictionResult(
            predictions=empty,
            failures=failures,
            model_id=spec.model_id,
            model_name=spec.model_name,
            nucleus=nucleus,
            device=resolved_device,
            molecule_records=[],
        )

    df_pred, preprocess_failures = core._predict_from_mols_ready(
        mols_ready=mols_ready,
        atom_indices=atom_indices,
        mols_smiles=smiles_ready,
        artifacts=artifacts,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=extra_cols,
        molecule_ids_per_mol=molecule_ids,
    )
    failures.extend(preprocess_failures)

    df_conformer = df_pred
    if not df_conformer.empty:
        df_conformer = df_conformer.copy()
        df_conformer["nucleus"] = nucleus
        df_conformer["model_id"] = spec.model_id
        df_conformer["model_name"] = spec.model_name
        df_pred = _boltzmann_average_prediction_rows(
            df_conformer,
            temperature_k=float(temperature_k),
        )
        df_pred["nucleus"] = nucleus
        df_pred["model_id"] = spec.model_id
        df_pred["model_name"] = spec.model_name

    return PredictionResult(
        predictions=df_pred,
        failures=list(failures),
        model_id=spec.model_id,
        model_name=spec.model_name,
        nucleus=nucleus,
        device=resolved_device,
        molecule_records=molecule_records,
        conformer_predictions=df_conformer,
        conformer_mol_records=conformer_mol_records,
    )


def _predict_from_structure_conformer_ensembles(
    mols: Sequence[Mol],
    smiles: Sequence[str],
    extra_cols: Mapping[str, Sequence[Any]],
    *,
    nucleus: str,
    spec: ModelSpec,
    artifacts,
    resolved_device: str,
    batch_size: int,
    dry_run: bool,
    temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
) -> PredictionResult:
    energy_values = list(extra_cols.get("conformer_energy", []))
    _require_conformer_energies([{"energy": value} for value in energy_values])

    no_target_reason = "no_carbon" if nucleus == "13C" else "no_proton"
    records_by_smiles: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        records_by_smiles.setdefault(str(smi), []).append(idx)

    molecule_records: List[MoleculeRecord] = []
    mols_ready: List[Mol] = []
    atom_indices: List[np.ndarray] = []
    smiles_ready: List[str] = []
    molecule_ids: List[int] = []
    conformer_mol_records: List[ConformerMolRecord] = []
    extra_cols_ready: Dict[str, List[Any]] = {str(key): [] for key in extra_cols}
    failures: List[str] = []

    for smi, indices in records_by_smiles.items():
        sorted_indices = sorted(
            indices,
            key=lambda idx: (
                10**9 if extra_cols.get("conformer_rank", [None] * len(mols))[idx] is None
                else int(extra_cols["conformer_rank"][idx]),
                idx,
            ),
        )
        representative_idx = sorted_indices[0]
        representative_mol = mols[representative_idx]
        representative_atom_indices = _atom_indices_for_nucleus(representative_mol, nucleus)
        if representative_atom_indices.size == 0:
            failures.append(f"{no_target_reason}\t{smi}")
            continue

        molecule_id = len(molecule_records)
        molecule_records.append(
            MoleculeRecord(
                molecule_id=molecule_id,
                smiles=str(smi),
                mol=representative_mol,
                metadata={
                    "conformer_count": len(sorted_indices),
                    "boltzmann_temperature_k": float(temperature_k),
                },
            )
        )
        for idx in sorted_indices:
            mol = mols[idx]
            target_indices = _atom_indices_for_nucleus(mol, nucleus)
            if target_indices.size == 0:
                raise ValueError(f"Conformer for {smi} has no target atoms for nucleus {nucleus}")
            mols_ready.append(mol)
            atom_indices.append(target_indices)
            smiles_ready.append(str(smi))
            molecule_ids.append(molecule_id)
            conformer_metadata: Dict[str, Any] = {}
            for key, values in extra_cols.items():
                value = list(values)[idx]
                extra_cols_ready[str(key)].append(value)
                conformer_metadata[str(key)] = value
            conformer_mol_records.append(
                ConformerMolRecord(
                    molecule_id=molecule_id,
                    mol=mol,
                    metadata=conformer_metadata,
                )
            )

    if not mols_ready:
        empty = pd.DataFrame(
            columns=["molecule_id", "smiles", "atom_index", "shift_ppm", "nucleus", "model_id", "model_name"]
        )
        return PredictionResult(
            predictions=empty,
            failures=failures,
            model_id=spec.model_id,
            model_name=spec.model_name,
            nucleus=nucleus,
            device=resolved_device,
            molecule_records=[],
        )

    df_conformer, preprocess_failures = core._predict_from_mols_ready(
        mols_ready=mols_ready,
        atom_indices=atom_indices,
        mols_smiles=smiles_ready,
        artifacts=artifacts,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=extra_cols_ready,
        molecule_ids_per_mol=molecule_ids,
    )
    failures.extend(preprocess_failures)

    if df_conformer.empty:
        df_pred = df_conformer
    else:
        df_pred = _boltzmann_average_prediction_rows(
            df_conformer,
            temperature_k=float(temperature_k),
        )
        df_pred["nucleus"] = nucleus
        df_pred["model_id"] = spec.model_id
        df_pred["model_name"] = spec.model_name
        df_conformer = df_conformer.copy()
        df_conformer["nucleus"] = nucleus
        df_conformer["model_id"] = spec.model_id
        df_conformer["model_name"] = spec.model_name

    return PredictionResult(
        predictions=df_pred,
        failures=list(failures),
        model_id=spec.model_id,
        model_name=spec.model_name,
        nucleus=nucleus,
        device=resolved_device,
        molecule_records=molecule_records,
        conformer_predictions=df_conformer,
        conformer_mol_records=conformer_mol_records,
    )


def _load_structures_dataset(
    structures_path: Path,
    *,
    use_all_conformers: bool,
    conformer_rank: int,
    limit: int,
) -> Tuple[List[Mol], List[str], Dict[str, List[Any]], List[str]]:
    import pyarrow.dataset as ds  # noqa: WPS433

    dataset_path = Path(structures_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"structures_path does not exist: {dataset_path}")

    dataset = ds.dataset(str(dataset_path), format="parquet", exclude_invalid_files=True)
    available_cols = set(dataset.schema.names)
    required = {"smiles", "molblock"}
    missing = required - available_cols
    if missing:
        raise ValueError(
            f"structures dataset missing required columns: {sorted(missing)} in {dataset_path}"
        )

    cols = ["smiles", "molblock"]
    for optional in ("conformer_rank", "conformer_id", "energy", "energy_method"):
        if optional in available_cols:
            cols.append(optional)

    mols: List[Mol] = []
    smiles: List[str] = []
    failures: List[str] = []
    extra_cols: Dict[str, List[Any]] = {
        "conformer_rank": [],
        "conformer_id": [],
        "conformer_energy": [],
        "conformer_energy_method": [],
    }
    seen_smiles: set[str] = set()

    for batch in dataset.to_batches(columns=cols):
        data = batch.to_pydict()
        smiles_col = data["smiles"]
        molblock_col = data["molblock"]
        rank_col = data.get("conformer_rank")
        conf_id_col = data.get("conformer_id")
        energy_col = data.get("energy")
        method_col = data.get("energy_method")

        for idx, (smi, molblock) in enumerate(zip(smiles_col, molblock_col)):
            if smi is None:
                failures.append("missing_smiles\t<none>")
                continue
            smi = str(smi)
            rank_val = rank_col[idx] if rank_col is not None else None
            if not use_all_conformers:
                if rank_col is not None:
                    if rank_val != int(conformer_rank):
                        continue
                else:
                    if smi in seen_smiles:
                        continue
                    seen_smiles.add(smi)

            mol = core.mol_from_molblock(molblock)
            if mol is None:
                failures.append(f"molblock\t{smi}")
                continue

            mols.append(mol)
            smiles.append(smi)
            extra_cols["conformer_rank"].append(rank_val)
            extra_cols["conformer_id"].append(conf_id_col[idx] if conf_id_col is not None else None)
            extra_cols["conformer_energy"].append(energy_col[idx] if energy_col is not None else None)
            extra_cols["conformer_energy_method"].append(method_col[idx] if method_col is not None else None)

            if int(limit) > 0 and len(mols) >= int(limit):
                return mols, smiles, extra_cols, failures

    return mols, smiles, extra_cols, failures


def predict_smiles(
    smiles: str | Sequence[str] | Iterable[str],
    *,
    nucleus: str = "13C",
    model_name: str | None = None,
    solvent: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    max_embed_tries: int = 20,
    num_conformers: int = 1,
    prune_rms_thresh: float = 0.0,
    ff_max_iters: int = 200,
    adaptive_conformers: bool = False,
    skip_heavy_atoms_gt: int = 0,
    embed_stall_timeout_s: float = 120.0,
    num_workers: int = 1,
    mp_context: str = "spawn",
    mp_chunksize: int = 20,
    boltzmann_temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
    dry_run: bool = False,
) -> PredictionResult:
    nucleus_norm = normalize_nucleus(nucleus)
    spec: ModelSpec = get_model_spec(nucleus_norm, model_name=model_name, solvent=solvent)
    smiles_list = _normalize_smiles_input(smiles)

    artifacts = _dry_artifacts(spec) if dry_run else _load_artifacts_cached(spec.model_id, device)
    resolved_device = "cpu" if artifacts is None else str(artifacts.device)

    if int(num_conformers) > 1:
        conformer_records, failures = core.embed_smiles_conformers(
            smiles_list,
            max_embed_tries=int(max_embed_tries),
            num_conformers=int(num_conformers),
            prune_rms_thresh=float(prune_rms_thresh),
            ff_max_iters=int(ff_max_iters),
            adaptive_conformers=bool(adaptive_conformers),
            skip_heavy_atoms_gt=int(skip_heavy_atoms_gt),
            embed_stall_timeout_s=float(embed_stall_timeout_s),
            num_workers=int(num_workers),
            mp_context=str(mp_context),
            mp_chunksize=int(mp_chunksize),
        )
        result = _predict_from_smiles_conformer_records(
            conformer_records,
            smiles_list,
            nucleus=nucleus_norm,
            spec=spec,
            artifacts=artifacts,
            resolved_device=resolved_device,
            batch_size=int(batch_size),
            dry_run=bool(dry_run),
            temperature_k=float(boltzmann_temperature_k),
        )
        result.failures = list(failures) + list(result.failures)
        return result

    mols_embedded, smiles_embedded, failures = core.embed_smiles(
        smiles_list,
        max_embed_tries=int(max_embed_tries),
        num_conformers=int(num_conformers),
        prune_rms_thresh=float(prune_rms_thresh),
        ff_max_iters=int(ff_max_iters),
        adaptive_conformers=bool(adaptive_conformers),
        skip_heavy_atoms_gt=int(skip_heavy_atoms_gt),
        embed_stall_timeout_s=float(embed_stall_timeout_s),
        num_workers=int(num_workers),
        mp_context=str(mp_context),
        mp_chunksize=int(mp_chunksize),
    )

    result = _predict_from_mols(
        mols_embedded,
        smiles_embedded,
        nucleus=nucleus_norm,
        spec=spec,
        artifacts=artifacts,
        resolved_device=resolved_device,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=None,
    )
    result.failures = list(failures) + list(result.failures)
    return result


def predict_mols(
    mols: Sequence[Mol] | Iterable[Mol],
    *,
    smiles: Sequence[str] | Iterable[str] | None = None,
    nucleus: str = "13C",
    model_name: str | None = None,
    solvent: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    dry_run: bool = False,
    extra_cols: Mapping[str, Sequence[Any]] | None = None,
) -> PredictionResult:
    nucleus_norm = normalize_nucleus(nucleus)
    spec: ModelSpec = get_model_spec(nucleus_norm, model_name=model_name, solvent=solvent)
    mol_values = _normalize_mols_input(mols)

    if smiles is None:
        smiles_values = [_mol_to_input_smiles(mol) for mol in mol_values]
    else:
        smiles_values = _normalize_smiles_input(smiles)
        if len(smiles_values) != len(mol_values):
            raise ValueError(
                f"mols/smiles length mismatch: got {len(mol_values)} mols and {len(smiles_values)} smiles"
            )

    artifacts = _dry_artifacts(spec) if dry_run else _load_artifacts_cached(spec.model_id, device)
    resolved_device = "cpu" if artifacts is None else str(artifacts.device)
    return _predict_from_mols(
        mol_values,
        smiles_values,
        nucleus=nucleus_norm,
        spec=spec,
        artifacts=artifacts,
        resolved_device=resolved_device,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=extra_cols,
    )


def predict_molecule_file(
    molecule_file: str | Path,
    *,
    nucleus: str = "13C",
    model_name: str | None = None,
    solvent: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    regenerate_geometry: bool = False,
    max_embed_tries: int = 20,
    prune_rms_thresh: float = 0.0,
    ff_max_iters: int = 200,
    dry_run: bool = False,
) -> PredictionResult:
    mols, smiles = _load_molecule_file(molecule_file)
    if bool(regenerate_geometry):
        mols = _regenerate_molecule_geometries(
            mols,
            smiles,
            max_embed_tries=int(max_embed_tries),
            prune_rms_thresh=float(prune_rms_thresh),
            ff_max_iters=int(ff_max_iters),
        )

    return predict_mols(
        mols,
        smiles=smiles,
        nucleus=nucleus,
        model_name=model_name,
        solvent=solvent,
        device=device,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=None,
    )


def predict_structures(
    structures_path: str | Path,
    *,
    nucleus: str = "13C",
    model_name: str | None = None,
    solvent: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    conformer_rank: int = 0,
    use_all_conformers: bool = False,
    limit: int = 0,
    boltzmann_temperature_k: float = DEFAULT_BOLTZMANN_TEMPERATURE_K,
    dry_run: bool = False,
) -> PredictionResult:
    nucleus_norm = normalize_nucleus(nucleus)
    spec: ModelSpec = get_model_spec(nucleus_norm, model_name=model_name, solvent=solvent)
    mols, smiles, extra_cols, load_failures = _load_structures_dataset(
        Path(structures_path),
        use_all_conformers=bool(use_all_conformers),
        conformer_rank=int(conformer_rank),
        limit=int(limit),
    )
    if not mols:
        empty = pd.DataFrame(
            columns=["molecule_id", "smiles", "atom_index", "shift_ppm", "nucleus", "model_id", "model_name"]
        )
        return PredictionResult(
            predictions=empty,
            failures=list(load_failures),
            model_id=spec.model_id,
            model_name=spec.model_name,
            nucleus=nucleus_norm,
            device="cpu" if dry_run else core._resolve_device(str(device), _RUNTIME_DIR),
            molecule_records=[],
        )

    artifacts = _dry_artifacts(spec) if dry_run else _load_artifacts_cached(spec.model_id, device)
    resolved_device = "cpu" if artifacts is None else str(artifacts.device)

    if bool(use_all_conformers):
        result = _predict_from_structure_conformer_ensembles(
            mols,
            smiles,
            extra_cols,
            nucleus=nucleus_norm,
            spec=spec,
            artifacts=artifacts,
            resolved_device=resolved_device,
            batch_size=int(batch_size),
            dry_run=bool(dry_run),
            temperature_k=float(boltzmann_temperature_k),
        )
        result.failures = list(load_failures) + list(result.failures)
        return result

    result = _predict_from_mols(
        mols,
        smiles,
        nucleus=nucleus_norm,
        spec=spec,
        artifacts=artifacts,
        resolved_device=resolved_device,
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        extra_cols=extra_cols,
    )
    result.failures = list(load_failures) + list(result.failures)
    return result


def predict_sdf(
    sdf_path: str | Path,
    *,
    nucleus: str = "13C",
    model_name: str | None = None,
    solvent: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    regenerate_geometry: bool = False,
    max_embed_tries: int = 20,
    prune_rms_thresh: float = 0.0,
    ff_max_iters: int = 200,
    dry_run: bool = False,
) -> PredictionResult:
    return predict_molecule_file(
        sdf_path,
        nucleus=nucleus,
        model_name=model_name,
        solvent=solvent,
        device=device,
        batch_size=int(batch_size),
        regenerate_geometry=bool(regenerate_geometry),
        max_embed_tries=int(max_embed_tries),
        prune_rms_thresh=float(prune_rms_thresh),
        ff_max_iters=int(ff_max_iters),
        dry_run=bool(dry_run),
    )
