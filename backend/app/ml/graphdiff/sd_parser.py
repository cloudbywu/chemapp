"""Streaming parser for $-$-delimited SD files from nmrshiftdb2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator


_FLOAT_RE = re.compile(r"[-+]?\d+\.?\d*")


def _parse_signal_token(token: str) -> dict[str, float | int | str | None]:
    """Parse one signal token of the form shift;intensity+mult;atomRef."""
    token = token.strip()
    if not token:
        msg = "Empty signal token"
        raise ValueError(msg)

    parts = token.split(";")
    if len(parts) != 3:
        msg = f"Invalid signal token: {token!r}"
        raise ValueError(msg)

    shift_str, intensity_mult, atom_ref_str = parts
    shift = float(shift_str)
    atom_ref = int(atom_ref_str)

    # intensity+mult: numeric prefix followed by an optional multiplicity string
    # (e.g. "0.0S", "0.0br", "0.0br s", or malformed data like "0.04.60")
    intensity_mult = intensity_mult.strip()
    match = _FLOAT_RE.match(intensity_mult)
    if not match:
        raise ValueError(f"No numeric intensity in {intensity_mult!r} of token {token!r}")
    intensity_str = match.group(0)
    mult = intensity_mult[match.end():].strip() or None
    intensity = float(intensity_str)

    return {
        "shift": shift,
        "intensity": intensity,
        "multiplicity": mult,
        "atom_ref": atom_ref,
    }


def _parse_signals(value: str) -> list[dict[str, float | int | str | None]]:
    """Parse a pipe-delimited signal string."""
    if not value:
        return []
    tokens = [t for t in value.split("|") if t]
    return [_parse_signal_token(token) for token in tokens]


def _parse_molblock(lines: list[str]) -> tuple[str, dict[str, str], list[str]]:
    """Split raw molecule lines into molblock, tag dict, and tag order.

    Returns the molblock (up to and including 'M  END'), a mapping of tag name
    to value, and the ordered list of tag names as they appear.
    """
    mol_end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "M  END":
            mol_end_idx = i
            break

    if mol_end_idx == -1:
        molblock_lines = lines
        tag_lines: list[str] = []
    else:
        molblock_lines = lines[: mol_end_idx + 1]
        tag_lines = lines[mol_end_idx + 1 :]

    molblock = "\n".join(molblock_lines)

    tags: dict[str, str] = {}
    tag_order: list[str] = []
    i = 0
    while i < len(tag_lines):
        line = tag_lines[i]
        stripped = line.strip()
        if stripped.startswith("> <") and stripped.endswith(">"):
            tag_name = stripped[3:-1]
            i += 1
            value_lines: list[str] = []
            while i < len(tag_lines):
                next_line = tag_lines[i]
                if next_line.strip().startswith("> <") and next_line.strip().endswith(">"):
                    break
                value_lines.append(next_line)
                i += 1
            value = "\n".join(value_lines).strip()
            tags[tag_name] = value
            tag_order.append(tag_name)
        else:
            i += 1

    return molblock, tags, tag_order


def _extract_element(atom_line: str) -> str | None:
    """Return the element symbol from an MDL V2000 atom line."""
    parts = atom_line.split()
    if len(parts) < 4:
        return None
    return parts[3]


def stream_sd(path: str | Path) -> Iterator[dict]:
    """Yield one dict per molecule from an SD file without loading it whole.

    Each yielded dict has keys:
      - molblock: str, the MDL molblock up to and including 'M  END'
      - tags: dict[str, str], all tag name/value pairs
      - signals_13c: list[dict], parsed signals from '<Spectrum 13C 0>'
      - signals_1h: list[dict], parsed signals from '<Spectrum 1H 0>'
        (or the first available '<Spectrum 1H N>' tag)
      - atom_count: int, number of atoms declared in the counts line
      - heavy_atom_count: int, atoms that are not hydrogen
      - elements: dict[str, int], element symbol counts in the molblock
    """
    path = Path(path)
    buffer: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("$$$$"):
                if buffer:
                    yield _build_entry(buffer)
                    buffer = []
                continue
            buffer.append(line)

        if buffer:
            yield _build_entry(buffer)


def _build_entry(lines: list[str]) -> dict:
    """Build an entry dict from the raw lines of a single molecule."""
    molblock, tags, _tag_order = _parse_molblock(lines)

    signals_13c: list[dict] = []
    if "Spectrum 13C 0" in tags:
        signals_13c = _parse_signals(tags["Spectrum 13C 0"])

    signals_1h: list[dict] = []
    if "Spectrum 1H 0" in tags:
        signals_1h = _parse_signals(tags["Spectrum 1H 0"])
    else:
        # Fall back to any available 1H spectrum tag (e.g. '<Spectrum 1H 1>')
        for name in tags:
            if name.startswith("Spectrum 1H "):
                signals_1h = _parse_signals(tags[name])
                break

    atom_count = 0
    heavy_atom_count = 0
    elements: dict[str, int] = {}

    mol_lines = molblock.split("\n")
    counts_parsed = False
    atoms_remaining = 0
    for mol_line in mol_lines:
        stripped = mol_line.strip()
        if stripped == "M  END":
            break
        if not counts_parsed:
            parts = stripped.split()
            # Require the MDL V2000 counts line: at least atom/bond counts and version.
            if len(parts) >= 2 and ("V2000" in stripped or "V3000" in stripped):
                try:
                    declared_atoms = int(parts[0])
                    bond_count = int(parts[1])
                    if declared_atoms >= 0 and bond_count >= 0:
                        atom_count = declared_atoms
                        atoms_remaining = declared_atoms
                        counts_parsed = True
                        continue
                except ValueError:
                    pass
        if counts_parsed and atoms_remaining > 0:
            element = _extract_element(mol_line)
            if element is not None:
                element = element.strip()
                elements[element] = elements.get(element, 0) + 1
                if element != "H":
                    heavy_atom_count += 1
                atoms_remaining -= 1

    return {
        "molblock": molblock,
        "tags": tags,
        "signals_13c": signals_13c,
        "signals_1h": signals_1h,
        "atom_count": atom_count,
        "heavy_atom_count": heavy_atom_count,
        "elements": elements,
    }
