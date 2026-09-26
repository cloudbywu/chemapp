"""RDKit-native drawing helpers for CSP5 prediction output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import html
import math

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.rdchem import Mol


DEFAULT_WIDTH = None
DEFAULT_HEIGHT = None
DEFAULT_BOND_LENGTH = 64
DEFAULT_ATOM_FONT_SIZE = 12
DEFAULT_SHIFT_FONT_SCALE = 1.00
DEFAULT_NOTE_FONT_SCALE = DEFAULT_SHIFT_FONT_SCALE
DEFAULT_PADDING = 0.06
DEFAULT_MIN_WIDTH = 260
DEFAULT_MIN_HEIGHT = 220


def _format_shift(value: Any) -> str:
    if value is None:
        return "NA"
    shift = float(value)
    if not math.isfinite(shift):
        return "NA"
    return f"{shift:.2f}"


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_shift_note(row: Mapping[str, Any]) -> str:
    shift_value = _finite_float_or_none(row.get("shift_ppm"))
    shift = _format_shift(shift_value)
    shift_std = _finite_float_or_none(row.get("shift_std_ppm"))
    if shift_value is None or shift_std is None:
        return shift
    if shift_std < 0:
        raise ValueError("shift_std_ppm must be non-negative")
    return f"{shift} +/- {shift_std:.2f}"


def _molecule_from_payload(molecule_payload: Mapping[str, Any]) -> Mol:
    mapped_smiles = str(molecule_payload.get("mapped_smiles_explicit_h") or "")
    if not mapped_smiles:
        raise ValueError("Molecule payload is missing mapped_smiles_explicit_h")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(mapped_smiles, params)
    if mol is None:
        raise ValueError("Failed to parse mapped_smiles_explicit_h")
    return mol


def _prediction_element(molecule_payload: Mapping[str, Any]) -> str | None:
    predictions = molecule_payload.get("predictions", [])
    if not predictions:
        return None
    elements = {str(row.get("element") or "") for row in predictions}
    elements.discard("")
    if len(elements) != 1:
        return None
    return next(iter(elements))


def _prepare_mol_for_drawing(mol: Mol, molecule_payload: Mapping[str, Any]) -> Mol:
    if _prediction_element(molecule_payload) == "C":
        return Chem.RemoveHs(mol, sanitize=False)
    return mol


def _annotated_mol_from_payload(
    molecule_payload: Mapping[str, Any],
) -> tuple[Mol, dict[int, str], list[str], list[str]]:
    mol = _molecule_from_payload(molecule_payload)
    mol = _prepare_mol_for_drawing(mol, molecule_payload)
    atoms_by_map = {
        int(atom.GetAtomMapNum()): atom
        for atom in mol.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }

    labels: list[str] = []
    notes: list[str] = []
    atom_labels: dict[int, str] = {}
    for row in molecule_payload.get("predictions", []):
        atom_map = int(row["atom_map"])
        atom = atoms_by_map.get(atom_map)
        if atom is None:
            raise ValueError(f"Prediction atom_map {atom_map} is absent from mapped_smiles_explicit_h")
        label = f"{atom.GetSymbol()}{atom_map}"
        note = _format_shift_note(row)
        labels.append(label)
        notes.append(note)
        atom_labels[int(atom.GetIdx())] = label
        atom.SetProp("atomLabel", label)
        atom.SetProp("atomNote", note)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    return mol, atom_labels, labels, notes


def _add_svg_metadata(svg: str, labels: Sequence[str], notes: Sequence[str]) -> str:
    escaped_labels = "; ".join(html.escape(label) for label in labels)
    escaped_notes = "; ".join(html.escape(note) for note in notes)
    metadata = (
        f"<metadata id='csp5-atom-labels'>{escaped_labels}</metadata>\n"
        f"<metadata id='csp5-shift-notes'>{escaped_notes}</metadata>\n"
    )
    if "</svg>" not in svg:
        return svg + "\n" + metadata
    return svg.replace("</svg>", metadata + "</svg>", 1)


def _payload_from_prediction(prediction) -> Mapping[str, Any]:
    if hasattr(prediction, "to_compact_mapped_payload"):
        return prediction.to_compact_mapped_payload()
    if not isinstance(prediction, Mapping):
        raise TypeError("draw_prediction expects a PredictionResult or compact prediction payload")
    return prediction


def _apply_draw_options(
    options,
    *,
    bond_length: int,
    atom_font_size: int,
    shift_font_scale: float,
    padding: float,
) -> None:
    options.fixedBondLength = int(bond_length)
    options.fixedFontSize = int(atom_font_size)
    options.annotationFontScale = float(shift_font_scale)
    options.padding = float(padding)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a median from no values")
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _coordinate_bounds(mol: Mol) -> tuple[float, float, float]:
    if mol.GetNumAtoms() == 0:
        raise ValueError("Cannot draw an empty molecule")
    conf = mol.GetConformer()
    x_values: list[float] = []
    y_values: list[float] = []
    for atom in mol.GetAtoms():
        position = conf.GetAtomPosition(int(atom.GetIdx()))
        x_values.append(float(position.x))
        y_values.append(float(position.y))

    bond_lengths: list[float] = []
    for bond in mol.GetBonds():
        begin = conf.GetAtomPosition(int(bond.GetBeginAtomIdx()))
        end = conf.GetAtomPosition(int(bond.GetEndAtomIdx()))
        length = math.hypot(float(end.x - begin.x), float(end.y - begin.y))
        if length > 0:
            bond_lengths.append(length)

    median_bond_length = _median(bond_lengths) if bond_lengths else 1.0
    return max(x_values) - min(x_values), max(y_values) - min(y_values), median_bond_length


def _auto_canvas_size(
    mol: Mol,
    *,
    bond_length: int,
    atom_font_size: int,
    shift_font_scale: float,
) -> tuple[int, int]:
    coord_width, coord_height, coord_bond_length = _coordinate_bounds(mol)
    if coord_bond_length <= 0:
        raise ValueError("Cannot auto-size drawing for molecule with zero-length bonds")
    scale = float(bond_length) / float(coord_bond_length)
    margin_px = max(
        float(bond_length) * 1.875,
        float(atom_font_size) * (5.0 + 7.0 * float(shift_font_scale)),
    )
    width = math.ceil(coord_width * scale + 2.0 * margin_px)
    height = math.ceil(coord_height * scale + 2.0 * margin_px)
    return max(DEFAULT_MIN_WIDTH, int(width)), max(DEFAULT_MIN_HEIGHT, int(height))


def _resolve_canvas_size(
    mols: Sequence[Mol],
    *,
    width: int | None,
    height: int | None,
    bond_length: int,
    atom_font_size: int,
    shift_font_scale: float,
) -> tuple[int, int]:
    if (width is None) != (height is None):
        raise ValueError("Provide both width and height, or omit both to auto-size the SVG canvas.")
    if width is not None and height is not None:
        return int(width), int(height)

    sizes = [
        _auto_canvas_size(
            mol,
            bond_length=int(bond_length),
            atom_font_size=int(atom_font_size),
            shift_font_scale=float(shift_font_scale),
        )
        for mol in mols
    ]
    return max(size[0] for size in sizes), max(size[1] for size in sizes)


def _resolve_shift_font_scale(
    shift_font_scale: float | None,
    note_font_scale: float | None,
) -> float:
    if shift_font_scale is None and note_font_scale is None:
        return float(DEFAULT_SHIFT_FONT_SCALE)
    if shift_font_scale is None:
        return float(note_font_scale)
    if note_font_scale is None:
        return float(shift_font_scale)
    if float(shift_font_scale) != float(note_font_scale):
        raise ValueError("Use either shift_font_scale or note_font_scale, not conflicting values for both.")
    return float(shift_font_scale)


def draw_prediction(
    prediction,
    *,
    width: int | None = DEFAULT_WIDTH,
    height: int | None = DEFAULT_HEIGHT,
    mols_per_row: int = 1,
    bond_length: int = DEFAULT_BOND_LENGTH,
    atom_font_size: int = DEFAULT_ATOM_FONT_SIZE,
    shift_font_scale: float | None = None,
    note_font_scale: float | None = None,
    padding: float = DEFAULT_PADDING,
) -> str:
    resolved_shift_font_scale = _resolve_shift_font_scale(shift_font_scale, note_font_scale)
    payload = _payload_from_prediction(prediction)
    molecules = payload.get("molecules")
    if not isinstance(molecules, Sequence) or isinstance(molecules, (str, bytes)):
        raise ValueError("Compact prediction payload must contain a molecules list")
    if not molecules:
        raise ValueError("Compact prediction payload contains no molecules to draw")

    mols: list[Mol] = []
    labels: list[str] = []
    notes: list[str] = []
    atom_labels_by_mol: list[dict[int, str]] = []
    for molecule in molecules:
        mol, atom_labels, molecule_labels, molecule_notes = _annotated_mol_from_payload(molecule)
        mols.append(mol)
        atom_labels_by_mol.append(atom_labels)
        labels.extend(molecule_labels)
        notes.extend(molecule_notes)
    resolved_width, resolved_height = _resolve_canvas_size(
        mols,
        width=width,
        height=height,
        bond_length=int(bond_length),
        atom_font_size=int(atom_font_size),
        shift_font_scale=resolved_shift_font_scale,
    )
    if len(mols) == 1:
        drawer = Draw.MolDraw2DSVG(int(resolved_width), int(resolved_height))
        options = drawer.drawOptions()
        _apply_draw_options(
            options,
            bond_length=int(bond_length),
            atom_font_size=int(atom_font_size),
            shift_font_scale=resolved_shift_font_scale,
            padding=float(padding),
        )
        for atom_idx, label in atom_labels_by_mol[0].items():
            options.atomLabels[int(atom_idx)] = label
        drawer.DrawMolecule(mols[0])
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    else:
        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=int(mols_per_row),
            subImgSize=(int(resolved_width), int(resolved_height)),
            useSVG=True,
        )
    if not isinstance(svg, str):
        raise RuntimeError("RDKit did not return SVG text")
    return _add_svg_metadata(svg, labels, notes)


def draw_compact_prediction_svg(
    payload: Mapping[str, Any],
    *,
    width: int | None = DEFAULT_WIDTH,
    height: int | None = DEFAULT_HEIGHT,
    mols_per_row: int = 1,
    bond_length: int = DEFAULT_BOND_LENGTH,
    atom_font_size: int = DEFAULT_ATOM_FONT_SIZE,
    shift_font_scale: float | None = None,
    note_font_scale: float | None = None,
    padding: float = DEFAULT_PADDING,
) -> str:
    return draw_prediction(
        payload,
        width=width,
        height=height,
        mols_per_row=int(mols_per_row),
        bond_length=int(bond_length),
        atom_font_size=int(atom_font_size),
        shift_font_scale=shift_font_scale,
        note_font_scale=note_font_scale,
        padding=float(padding),
    )


def draw_prediction_result_svg(
    result,
    *,
    width: int | None = DEFAULT_WIDTH,
    height: int | None = DEFAULT_HEIGHT,
    mols_per_row: int = 1,
    bond_length: int = DEFAULT_BOND_LENGTH,
    atom_font_size: int = DEFAULT_ATOM_FONT_SIZE,
    shift_font_scale: float | None = None,
    note_font_scale: float | None = None,
    padding: float = DEFAULT_PADDING,
) -> str:
    return draw_prediction(
        result,
        width=width,
        height=height,
        mols_per_row=int(mols_per_row),
        bond_length=int(bond_length),
        atom_font_size=int(atom_font_size),
        shift_font_scale=shift_font_scale,
        note_font_scale=note_font_scale,
        padding=float(padding),
    )


def write_compact_prediction_svg(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    width: int | None = DEFAULT_WIDTH,
    height: int | None = DEFAULT_HEIGHT,
    mols_per_row: int = 1,
    bond_length: int = DEFAULT_BOND_LENGTH,
    atom_font_size: int = DEFAULT_ATOM_FONT_SIZE,
    shift_font_scale: float | None = None,
    note_font_scale: float | None = None,
    padding: float = DEFAULT_PADDING,
) -> None:
    svg = draw_prediction(
        payload,
        width=width,
        height=height,
        mols_per_row=int(mols_per_row),
        bond_length=int(bond_length),
        atom_font_size=int(atom_font_size),
        shift_font_scale=shift_font_scale,
        note_font_scale=note_font_scale,
        padding=float(padding),
    )
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(svg, encoding="utf-8")
