"""Build atom-level 13C graph datasets from the v2 spectral index.

Each graph corresponds to one molecule; node targets are experimental 13C
shifts attached to carbon atoms via the SD atom reference (1-based molblock
atom index).  Only peaks with ``atom_ref >= 1`` are used as labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import CanonicalRankAtoms
from rdkit.Chem.Scaffolds import MurckoScaffold


MAX_ATOMS = 128
_ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se")


def _cache_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_cache_hash(path: Path) -> bool:
    """Return True only when a matching .sha256 sidecar exists for the cache."""

    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return False
    try:
        return sidecar.read_text(encoding="utf-8").strip() == _cache_sha256(path)
    except OSError:
        return False


def _write_cache_with_hash(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, str(path))
    Path(str(path) + ".sha256").write_text(
        _cache_sha256(path), encoding="utf-8"
    )

# Normalised solvent vocabulary (order is fixed; index 0 = unreported/other).
SOLVENT_VOCAB = (
    "unreported",
    "cdcl3",
    "dmso",
    "d2o",
    "cd3od",
    "acetone-d6",
    "pyridine-d5",
    "c6d6",
    "ccl4",
    "thf-d8",
    "cd2cl2",
    "other",
)
_N_SOLVENTS = len(SOLVENT_VOCAB)
_FIELD_NORM = 800.0


@dataclass(frozen=True)
class GraphSample:
    """One molecule graph with per-carbon shift targets."""

    sample_id: str
    inchikey: str
    scaffold: str
    canonical_smiles: str
    solvent: str | None
    field_mhz: float | None
    atom_symbols: tuple[str, ...]
    mol_condition: torch.Tensor          # (C,) solvent one-hot + normalised field
    node_feats: torch.Tensor          # (N, F)
    edge_index: torch.Tensor          # (2, E) int64
    edge_attr: torch.Tensor           # (E, B) float32
    targets: torch.Tensor             # (N,) float32, 0.0 where unlabelled
    target_mask: torch.Tensor         # (N,) bool
    env_labels: torch.Tensor           # (N,) int64: 0=sp3 1=sp2 2=aromatic 3=carbonyl

    @property
    def n_atoms(self) -> int:
        return self.node_feats.shape[0]

    def to_json(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "inchikey": self.inchikey,
            "scaffold": self.scaffold,
            "canonical_smiles": self.canonical_smiles,
            "solvent": self.solvent,
            "field_mhz": self.field_mhz,
            "n_atoms": self.n_atoms,
            "n_targets": int(self.target_mask.sum().item()),
        }


def _atom_feature(atom: Chem.Atom) -> np.ndarray:
    """Fixed-length atom feature vector (RDKit, after adding hydrogens)."""
    element = atom.GetSymbol()
    atomic = [1.0 if element == e else 0.0 for e in _ELEMENTS]
    atomic.append(1.0 if element not in _ELEMENTS else 0.0)  # other

    degree = np.zeros(7, dtype=np.float32)
    degree[min(atom.GetDegree(), 6)] = 1.0
    hcount = np.zeros(5, dtype=np.float32)
    hcount[min(atom.GetTotalNumHs(), 4)] = 1.0
    charge = np.zeros(5, dtype=np.float32)
    charge[max(0, min(atom.GetFormalCharge() + 2, 4))] = 1.0

    hyb = np.zeros(6, dtype=np.float32)
    try:
        hyb[min(int(atom.GetHybridization()), 5)] = 1.0
    except Exception:  # pragma: no cover - defensive
        hyb[0] = 1.0

    chiral = np.zeros(5, dtype=np.float32)
    chiral[min(int(atom.GetChiralTag()), 4)] = 1.0

    return np.concatenate(
        [
            np.asarray(atomic, dtype=np.float32),
            degree,
            hcount,
            charge,
            hyb,
            np.asarray([float(atom.GetIsAromatic())], dtype=np.float32),
            np.asarray([float(atom.IsInRing())], dtype=np.float32),
            chiral,
        ]
    )


def _bond_feature(bond: Chem.Bond) -> np.ndarray:
    bond_type = np.zeros(4, dtype=np.float32)
    bond_type[max(0, min(int(bond.GetBondType()) - 1, 3))] = 1.0
    stereo = np.zeros(3, dtype=np.float32)
    stereo[min(int(bond.GetStereo()), 2)] = 1.0
    return np.concatenate(
        [
            bond_type,
            np.asarray([float(bond.GetIsAromatic())], dtype=np.float32),
            np.asarray([float(bond.GetIsConjugated())], dtype=np.float32),
            np.asarray([float(bond.IsInRing())], dtype=np.float32),
            stereo,
        ]
    )


def mol_to_graph(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Convert an RDKit molecule into node/edge tensors."""
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    AllChem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n = mol.GetNumAtoms()
    if n > MAX_ATOMS:
        raise ValueError(f"molecule too large: {n} atoms > {MAX_ATOMS}")

    feats = np.stack([_atom_feature(a) for a in mol.GetAtoms()]).astype(np.float32)
    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])
        eattr.extend([_bond_feature(bond), _bond_feature(bond)])
    symbols = tuple(a.GetSymbol() for a in mol.GetAtoms())
    return (
        torch.as_tensor(feats),
        torch.as_tensor([src, dst], dtype=torch.long),
        torch.as_tensor(np.stack(eattr), dtype=torch.float32) if eattr else torch.zeros((0, 10), dtype=torch.float32),
        symbols,
    )


def _stable_id(inchikey: str, smiles: str) -> str:
    digest = hashlib.sha256(f"{inchikey}|{smiles}".encode("utf-8")).hexdigest()
    return digest[:16]


# Unicode dash variants that must collapse to ASCII "-" before matching:
# hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal
# bar, minus sign.  The previous .replace("-", "-") was a no-op.
_UNICODE_DASHES = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212")


def _normalise_solvent(solvent: str | None) -> str:
    value = (solvent or "").strip().lower()
    for dash in _UNICODE_DASHES:
        value = value.replace(dash, "-")
    if not value or value in {"unreported", "unknown", "(none)", "none"}:
        return "unreported"
    for key in SOLVENT_VOCAB:
        if key in value:
            return key
    if "chloroform" in value:
        return "cdcl3"
    if "dimethyl" in value or value == "dmso":
        return "dmso"
    if "pyridin" in value:
        return "pyridine-d5"
    if "benzene" in value:
        return "c6d6"
    if "methanol" in value:
        return "cd3od"
    if "acetone" in value:
        return "acetone-d6"
    return "other"


def _mol_condition(solvent: str | None, field_mhz: float | None) -> torch.Tensor:
    one_hot = torch.zeros(_N_SOLVENTS, dtype=torch.float32)
    index = SOLVENT_VOCAB.index(_normalise_solvent(solvent))
    one_hot[index] = 1.0
    field = 0.0 if not field_mhz else min(float(field_mhz) / _FIELD_NORM, 1.0)
    return torch.cat([one_hot, torch.as_tensor([field], dtype=torch.float32)])


def _environment_label(atom: Chem.Atom, mol: Chem.Mol) -> int:
    """0=sp3, 1=sp2, 2=aromatic, 3=carbonyl/hetero sp2 carbon."""
    if atom.GetIsAromatic():
        return 2
    hybridization = int(atom.GetHybridization())
    if hybridization == 3:  # RDKit SP2
        for neighbor in atom.GetNeighbors():
            symbol = neighbor.GetSymbol()
            if symbol in ("O", "N", "S"):
                return 3
        return 1
    if hybridization == 2:  # RDKit SP (alkyne/nitrile)
        return 1
    return 0


def _plausible_shift_range(atom: Chem.Atom, mol: Chem.Mol) -> tuple[float, float]:
    """Generous per-structure 13C shift range for label-quality filtering."""
    if atom.GetIsAromatic():
        return 95.0, 175.0
    for neighbor in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if neighbor.GetSymbol() == "O" and bond is not None and int(bond.GetBondType()) == 2:
            return 150.0, 230.0  # C=O
    hybridization = int(atom.GetHybridization())
    degree = atom.GetDegree()
    if hybridization == 2:  # RDKit SP
        return 30.0, 120.0  # sp (alkyne/nitrile-like)
    if hybridization == 3:  # RDKit SP2
        return 75.0, 175.0  # sp2 alkene/hetero
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() in ("O", "N", "S", "F", "Cl", "Br", "I"):
            return 20.0, 125.0
    if degree == 1:
        return -5.0, 60.0   # CH3
    if degree == 2:
        return 0.0, 90.0    # CH2
    if degree == 3:
        return 5.0, 110.0   # CH
    return 10.0, 130.0      # Cq


def _pick_spectrum(rows: list[tuple]) -> tuple | None:
    """Prefer explicit measured spectra, then inferred_measured."""
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r["measurement_kind"] == "measured" else 1,
            r["solvent"] is None,
        ),
    )
    return ranked[0] if ranked else None


def build_atom_dataset(
    index_path: str | Path,
    *,
    max_molecules: int | None = None,
    seed: int = 0,
    dedupe_inchikey: bool = True,
    cache_path: str | Path | None = None,
    filter_labels: bool = True,
    consensus_spectra: bool = True,
) -> list[GraphSample]:
    """Build a 13C atom-level dataset from the v2 spectral index.

    One spectrum per molecule (measured preferred).  Only carbon atoms with a
    positive atom reference receive shift targets.
    """
    if cache_path is not None and Path(cache_path).exists():
        if _verify_cache_hash(Path(cache_path)):
            print(f"[forward_v1] loading cached dataset from {cache_path}")
            return torch.load(str(cache_path), weights_only=False)
        print(
            "[forward_v1] cache hash sidecar missing or mismatch; rebuilding",
            flush=True,
        )

    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row
    molecules = conn.execute(
        """
        SELECT m.id AS molecule_id, m.molblock, m.inchi_key, m.smiles,
               m.formula, m.record_sha256
        FROM molecules m
        WHERE m.molblock IS NOT NULL AND m.molblock != ''
        """
    ).fetchall()

    spectra_by_mol: dict[int, list[sqlite3.Row]] = {}
    for row in conn.execute(
        """
        SELECT molecule_id, id AS spectrum_id, nucleus, measurement_kind,
               solvent, field_mhz
        FROM spectra
        WHERE nucleus = '13C'
          AND measurement_kind IN ('measured', 'inferred_measured')
        """
    ):
        spectra_by_mol.setdefault(row["molecule_id"], []).append(row)

    peaks_by_spectrum: dict[int, list[sqlite3.Row]] = {}
    for row in conn.execute(
        """
        SELECT spectrum_id, shift, atom_ref, multiplicity
        FROM peaks
        WHERE atom_ref IS NOT NULL AND atom_ref >= 1
        """
    ):
        peaks_by_spectrum.setdefault(row["spectrum_id"], []).append(row)

    rng = random.Random(seed)
    samples: list[GraphSample] = []
    seen_inchikeys: set[str] = set()
    for mol_row in molecules:
        if dedupe_inchikey:
            inchikey = str(mol_row["inchi_key"] or "")
            if inchikey in seen_inchikeys:
                continue
        try:
            mol = Chem.MolFromMolBlock(mol_row["molblock"])
            if mol is None:
                continue
            mol_no_h = mol
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            n_heavy = mol_no_h.GetNumAtoms()
            node_feats, edge_index, edge_attr, symbols = mol_to_graph(mol)
        except Exception:
            continue

        n_atoms_total = mol.GetNumAtoms()
        targets = torch.zeros(n_atoms_total, dtype=torch.float32)
        mask = torch.zeros(n_atoms_total, dtype=torch.bool)
        spectra = spectra_by_mol.get(mol_row["molecule_id"], [])
        if not spectra:
            continue
        preferred = _pick_spectrum(spectra)

        if consensus_spectra:
            shifts_by_ref: dict[int, list[float]] = {}
            n_assigned_refs: set[int] = set()
            for spectrum in spectra:
                for peak in peaks_by_spectrum.get(spectrum["spectrum_id"], []):
                    ref = int(peak["atom_ref"]) - 1
                    if ref < 0 or ref >= n_heavy:
                        continue
                    if symbols[ref] != "C":
                        continue
                    shift = float(peak["shift"])
                    if not math.isfinite(shift) or not (-20.0 <= shift <= 300.0):
                        continue
                    if filter_labels:
                        lo, hi = _plausible_shift_range(
                            mol.GetAtomWithIdx(ref), mol
                        )
                        if not (lo <= shift <= hi):
                            continue
                    # Count the ref only after it passes the same quality
                    # gates as the labels themselves; counting before
                    # filtering inflated the keep-threshold denominator
                    # (n_labels < max(2, len(n_assigned_refs)//2)) with shifts
                    # that were rejected anyway.  The threshold keeps its
                    # conservative direction: samples retaining fewer than
                    # half of the quality-passing carbon assignments (e.g.
                    # after the symmetry-conflict filter below) are dropped.
                    n_assigned_refs.add(ref)
                    shifts_by_ref.setdefault(ref, []).append(shift)
            pending = [
                (ref, float(sorted(values)[len(values) // 2]))
                for ref, values in shifts_by_ref.items()
                if values
            ]
        else:
            spectrum = preferred
            pending = []
            for peak in peaks_by_spectrum.get(spectrum["spectrum_id"], []):
                ref = int(peak["atom_ref"]) - 1
                if ref < 0 or ref >= n_heavy:
                    continue
                if symbols[ref] != "C":
                    continue
                shift = float(peak["shift"])
                if not math.isfinite(shift) or not (-20.0 <= shift <= 300.0):
                    continue
                if filter_labels:
                    lo, hi = _plausible_shift_range(mol.GetAtomWithIdx(ref), mol)
                    if not (lo <= shift <= hi):
                        continue
                pending.append((ref, shift))
            n_assigned_refs = {ref for ref, _ in pending}

        # Symmetry conflict filter: equivalent carbons must agree within 1 ppm.
        if filter_labels and pending:
            try:
                ranks = list(CanonicalRankAtoms(mol_no_h, breakTies=False))
                rank_values: dict[int, list[float]] = {}
                for ref, shift in pending:
                    rank_values.setdefault(ranks[ref], []).append(shift)
                keep_refs: set[int] = set()
                for ref, shift in pending:
                    values = rank_values[ranks[ref]]
                    if len(values) > 1 and max(values) - min(values) > 1.0:
                        continue
                    keep_refs.add(ref)
                pending = [(ref, shift) for ref, shift in pending if ref in keep_refs]
            except Exception:
                pass

        for ref, shift in pending:
            targets[ref] = shift
            mask[ref] = True
        n_labels = int(mask.sum().item())
        if n_labels == 0 or (
            filter_labels and n_labels < max(2, len(n_assigned_refs) // 2)
        ):
            continue

        smiles = Chem.MolToSmiles(mol)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol_no_h, includeChirality=False
            )
        except Exception:
            scaffold = smiles
        inchikey = str(mol_row["inchi_key"] or "")
        samples.append(
            GraphSample(
                sample_id=_stable_id(inchikey, smiles),
                inchikey=inchikey,
                scaffold=scaffold,
                canonical_smiles=smiles,
                solvent=preferred["solvent"] if preferred else None,
                field_mhz=preferred["field_mhz"] if preferred else None,
                atom_symbols=symbols,
                mol_condition=_mol_condition(
                    preferred["solvent"] if preferred else None,
                    preferred["field_mhz"] if preferred else None,
                ),
                node_feats=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                targets=targets,
                target_mask=mask,
                env_labels=torch.as_tensor(
                    [_environment_label(a, mol) for a in mol.GetAtoms()],
                    dtype=torch.long,
                ),
            )
        )
        if dedupe_inchikey:
            seen_inchikeys.add(inchikey)
        if max_molecules is not None and len(samples) >= max_molecules:
            break

    rng.shuffle(samples)
    if cache_path is not None:
        _write_cache_with_hash(Path(cache_path), samples)
        print(f"[forward_v1] cached {len(samples)} samples to {cache_path}")
    return samples


def build_exp22k_dataset(
    entries_pkl: str | Path,
    splits_json: str | Path,
    *,
    max_molecules: int | None = None,
    seed: int = 0,
    cache_path: str | Path | None = None,
) -> tuple[list[GraphSample], dict[str, list[int]]]:
    """Build a 13C atom-level dataset from CSP5 Exp22K assigned entries.

    The official ``CSP5-13C-scaffold-doi_split.json`` uses entry list indices
    (not the ``mol_id`` field).  Atom shift keys in ``exp_shift_dict`` refer to
    heavy-atom indices, which are preserved after removing explicit hydrogens
    from the 3D molblock.  Returns ``(samples, splits)`` where ``splits`` maps
    ``train/val/test`` to indices into ``samples``.
    """
    import json
    import pickle

    rng = random.Random(seed)
    if cache_path is not None and Path(cache_path).exists():
        if _verify_cache_hash(Path(cache_path)):
            print(f"[forward_v1] loading cached Exp22K dataset from {cache_path}")
            cached = torch.load(str(cache_path), weights_only=False)
            samples = list(cached["samples"])
            splits = {
                key: list(value) for key, value in cached["splits"].items()
            }
            if max_molecules:
                samples = samples[: int(max_molecules)]
            return samples, splits
        print(
            "[forward_v1] Exp22K cache hash sidecar missing or mismatch; "
            "rebuilding",
            flush=True,
        )

    with open(entries_pkl, "rb") as handle:
        entries = pickle.load(handle)
    with open(splits_json, "r", encoding="utf-8") as handle:
        split_doc = json.load(handle)

    if max_molecules:
        entries = entries[: int(max_molecules)]
    old_to_new: dict[int, int] = {}
    samples: list[GraphSample] = []
    for position, entry in enumerate(entries):
        try:
            mol_no_h = Chem.MolFromMolBlock(str(entry.get("mol_block") or ""))
        except Exception:
            mol_no_h = None
        if mol_no_h is None:
            continue
        try:
            Chem.SanitizeMol(mol_no_h)
        except Exception:
            continue
        canonical = Chem.MolToSmiles(mol_no_h)
        if not canonical:
            continue
        try:
            inchikey = Chem.MolToInchiKey(mol_no_h)
        except Exception:
            inchikey = ""
        scaffold = str(entry.get("scaffold") or "")
        if not scaffold:
            try:
                scaffold = Chem.MolToSmiles(
                    MurckoScaffold.GetScaffoldForMol(mol_no_h)
                )
            except Exception:
                scaffold = ""

        shifts = entry.get("exp_shift_dict") or {}
        n_heavy = mol_no_h.GetNumAtoms()
        if any(
            isinstance(key, bool)
            or not isinstance(key, (int, float))
            or int(key) < 0
            or int(key) >= n_heavy
            for key in shifts
        ):
            continue
        mol_h = Chem.AddHs(Chem.Mol(mol_no_h))
        try:
            Chem.SanitizeMol(mol_h)
            node_feats, edge_index, edge_attr, symbols = mol_to_graph(mol_h)
        except ValueError:
            continue
        n_atoms_total = mol_h.GetNumAtoms()
        targets = torch.zeros(n_atoms_total, dtype=torch.float32)
        mask = torch.zeros(n_atoms_total, dtype=torch.bool)
        for key, value in shifts.items():
            index = int(key)
            try:
                targets[index] = float(value)
                mask[index] = True
            except (TypeError, ValueError):
                continue
        if not bool(mask.any()):
            continue
        env_labels = torch.as_tensor(
            [
                _environment_label(mol_h.GetAtomWithIdx(i), mol_h)
                for i in range(n_atoms_total)
            ],
            dtype=torch.long,
        )
        samples.append(
            GraphSample(
                sample_id=f"exp22k-{entry.get('mol_id', position)}",
                inchikey=inchikey,
                scaffold=scaffold,
                canonical_smiles=canonical,
                solvent=None,
                field_mhz=None,
                atom_symbols=symbols,
                mol_condition=_mol_condition(None, None),
                node_feats=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                targets=targets,
                target_mask=mask,
                env_labels=env_labels,
            )
        )
        old_to_new[position] = len(samples) - 1

    raw_splits = split_doc.get("splits") or {}
    splits: dict[str, list[int]] = {}
    for key in ("train", "val", "test"):
        indices = []
        for old_index in raw_splits.get(key, ()):
            new_index = old_to_new.get(int(old_index))
            if new_index is not None:
                indices.append(new_index)
        if key == "train":
            rng.shuffle(indices)
        splits[key] = indices

    if cache_path is not None:
        _write_cache_with_hash(Path(cache_path), {"samples": samples, "splits": splits})
        print(f"[forward_v1] cached {len(samples)} Exp22K samples to {cache_path}")
    return samples, splits


def scaffold_group_split(
    samples: Iterable[GraphSample],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[GraphSample], list[GraphSample], list[GraphSample]]:
    """Split by Bemis-Murcko scaffold so scaffolds never cross splits."""
    samples = list(samples)
    rng = random.Random(seed)
    groups: dict[str, list[GraphSample]] = {}
    for sample in samples:
        groups.setdefault(sample.scaffold, []).append(sample)
    if len(groups) < 3:
        raise ValueError("at least 3 distinct scaffolds are required for a split")

    shuffled = sorted(groups.items(), key=lambda kv: rng.random())
    targets = {
        "train": max(1.0 - val_frac - test_frac, 1e-9),
        "val": max(val_frac, 1e-9),
        "test": max(test_frac, 1e-9),
    }
    sizes = {"train": 0, "val": 0, "test": 0}
    buckets: dict[str, list[GraphSample]] = {"train": [], "val": [], "test": []}
    for _, group in shuffled:
        # Assign to the split that is currently furthest below its target.
        split = min(
            ("train", "val", "test"),
            key=lambda s: sizes[s] / targets[s],
        )
        buckets[split].extend(group)
        sizes[split] += len(group)
    train, val, test = buckets["train"], buckets["val"], buckets["test"]
    return train, val, test


def batch_graphs(
    batch: list[GraphSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a list of graphs into one batched graph."""
    node_feats = torch.cat([g.node_feats for g in batch], dim=0)
    edge_attrs = torch.cat([g.edge_attr for g in batch], dim=0)
    targets = torch.cat([g.targets for g in batch], dim=0)
    mask = torch.cat([g.target_mask for g in batch], dim=0)
    env_labels = torch.cat([g.env_labels for g in batch], dim=0)

    offsets = torch.cumsum(
        torch.as_tensor([0] + [g.n_atoms for g in batch[:-1]], dtype=torch.long),
        dim=0,
    )
    edge_src = torch.cat(
        [g.edge_index[0] + offsets[i] for i, g in enumerate(batch)], dim=0
    )
    edge_dst = torch.cat(
        [g.edge_index[1] + offsets[i] for i, g in enumerate(batch)], dim=0
    )
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    molecule_ids = torch.cat(
        [
            torch.full((g.n_atoms,), i, dtype=torch.long)
            for i, g in enumerate(batch)
        ],
        dim=0,
    )
    mol_conditions = torch.stack([g.mol_condition for g in batch], dim=0)
    return (
        node_feats,
        edge_index,
        edge_attrs,
        targets,
        mask,
        molecule_ids,
        mol_conditions,
        env_labels,
    )
