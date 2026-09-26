"""LEGACY/DEMO prototype NMR dataset builder using chemical shift additivity rules.

TOMBSTONE (2026-08 ML review): this module synthesises 1H NMR-looking demo
spectra from 112 hand-written shift templates (``COMPOUND_TEMPLATES``); the
output is rule-based simulation data, not measured spectra, and the chain
(dataset -> ensemble predictor) is retained only as an archived prototype.

Scale honesty: ``build_dataset`` emits ``n_variants_per_template`` variants
per template (default in ``load_or_build_dataset``: 30), i.e.
``len(COMPOUND_TEMPLATES) x n_variants_per_template`` records (~3,360 at the
shipped default), each derived from the template's nominal shifts plus
per-peak Gaussian jitter, a solvent term and simulated coupling constants.
Do not cite counts from this module as dataset coverage.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

_DB_PATH = Path(__file__).parent / "compounds.db"
_SOLVENTS = ["CDCl3", "DMSO-d6", "CD3OD", "C6D6", "Acetone-d6"]
_FREQUENCIES = [300, 400, 500, 600]


def db_path() -> Path:
    return _DB_PATH


# ═══════════════════════════════════════════
# Compound templates with realistic NMR shifts
# Based on Pretsch/Silverstein shift tables
# ═══════════════════════════════════════════

COMPOUND_TEMPLATES = {
    # Alkanes (50 variants)
    "n-hexane": {"shifts": [(0.89, 0.1, "t", 3), (1.29, 0.15, "m", 8), (0.91, 0.1, "t", 3)]},
    "n-heptane": {"shifts": [(0.88, 0.1, "t", 3), (1.28, 0.15, "m", 10), (0.90, 0.1, "t", 3)]},
    "cyclohexane": {"shifts": [(1.44, 0.05, "s(br)", 12)]},
    "2-methylpentane": {"shifts": [(0.87, 0.1, "d", 6), (1.22, 0.15, "m", 4), (0.89, 0.1, "t", 3)]},
    "2,2-dimethylbutane": {"shifts": [(0.87, 0.05, "s", 9), (1.21, 0.1, "q", 2), (0.84, 0.1, "t", 3)]},

    # Aromatic (100 variants)
    "benzene": {"shifts": [(7.34, 0.02, "s", 6)]},
    "toluene": {"shifts": [(7.20, 0.15, "m", 5), (2.35, 0.05, "s", 3)]},
    "ethylbenzene": {"shifts": [(7.23, 0.15, "m", 5), (2.66, 0.05, "q", 2), (1.24, 0.05, "t", 3)]},
    "o-xylene": {"shifts": [(7.11, 0.1, "m", 4), (2.27, 0.05, "s", 6)]},
    "m-xylene": {"shifts": [(7.02, 0.1, "m", 3), (6.94, 0.05, "s", 1), (2.31, 0.05, "s", 6)]},
    "p-xylene": {"shifts": [(7.05, 0.02, "s", 4), (2.30, 0.05, "s", 6)]},
    "mesitylene": {"shifts": [(6.78, 0.02, "s", 3), (2.26, 0.05, "s", 9)]},
    "naphthalene": {"shifts": [(7.81, 0.1, "m", 4), (7.46, 0.1, "m", 4)]},
    "biphenyl": {"shifts": [(7.58, 0.1, "m", 4), (7.43, 0.1, "m", 4), (7.34, 0.1, "m", 2)]},
    "styrene": {"shifts": [(7.40, 0.1, "m", 5), (6.72, 0.05, "dd", 1), (5.74, 0.05, "d", 1), (5.24, 0.05, "d", 1)]},

    # Substituted benzenes
    "chlorobenzene": {"shifts": [(7.28, 0.15, "m", 5)]},
    "bromobenzene": {"shifts": [(7.50, 0.1, "d", 2), (7.30, 0.1, "t", 2), (7.22, 0.1, "t", 1)]},
    "iodobenzene": {"shifts": [(7.83, 0.1, "d", 2), (7.37, 0.1, "t", 2), (7.13, 0.1, "t", 1)]},
    "phenol": {"shifts": [(7.25, 0.1, "t", 2), (6.95, 0.1, "d", 2), (6.84, 0.1, "t", 1), (5.40, 0.5, "s(br)", 1)]},
    "anisole": {"shifts": [(7.28, 0.1, "t", 2), (6.95, 0.05, "d", 2), (6.90, 0.05, "t", 1), (3.80, 0.05, "s", 3)]},
    "aniline": {"shifts": [(7.15, 0.1, "t", 2), (6.75, 0.1, "d", 2), (6.65, 0.1, "t", 1), (3.50, 0.5, "s(br)", 2)]},
    "nitrobenzene": {"shifts": [(8.23, 0.1, "d", 2), (7.72, 0.1, "t", 1), (7.56, 0.1, "t", 2)]},
    "benzonitrile": {"shifts": [(7.63, 0.1, "d", 2), (7.60, 0.1, "t", 1), (7.49, 0.1, "t", 2)]},
    "benzaldehyde": {"shifts": [(10.00, 0.05, "s", 1), (7.87, 0.1, "d", 2), (7.62, 0.1, "t", 1), (7.52, 0.1, "t", 2)]},
    "benzoic_acid": {"shifts": [(12.0, 0.5, "s(br)", 1), (8.13, 0.1, "d", 2), (7.62, 0.1, "t", 1), (7.48, 0.1, "t", 2)]},
    "acetophenone": {"shifts": [(7.95, 0.1, "d", 2), (7.55, 0.1, "t", 1), (7.45, 0.1, "t", 2), (2.60, 0.05, "s", 3)]},
    "methyl_benzoate": {"shifts": [(8.04, 0.1, "d", 2), (7.55, 0.1, "t", 1), (7.44, 0.1, "t", 2), (3.90, 0.05, "s", 3)]},
    "benzamide": {"shifts": [(7.85, 0.1, "d", 2), (7.52, 0.1, "t", 1), (7.44, 0.1, "t", 2), (6.3, 0.5, "s(br)", 2)]},
    "4-nitrotoluene": {"shifts": [(8.11, 0.05, "d", 2), (7.31, 0.05, "d", 2), (2.45, 0.05, "s", 3)]},
    "4-chloroaniline": {"shifts": [(7.10, 0.05, "d", 2), (6.61, 0.05, "d", 2), (3.60, 0.5, "s(br)", 2)]},
    "salicylic_acid": {"shifts": [(10.5, 0.5, "s(br)", 1), (7.92, 0.05, "dd", 1), (7.45, 0.1, "t", 1), (6.98, 0.05, "d", 1), (6.92, 0.05, "t", 1)]},
    "para-hydroxybenzoic_acid": {"shifts": [(12.4, 0.5, "s(br)", 1), (10.2, 0.5, "s(br)", 1), (7.79, 0.05, "d", 2), (6.84, 0.05, "d", 2)]},

    # Heterocyclic aromatics (40 variants)
    "pyridine": {"shifts": [(8.60, 0.05, "d", 2), (7.68, 0.05, "t", 1), (7.27, 0.05, "t", 2)]},
    "2-picoline": {"shifts": [(8.46, 0.05, "d", 1), (7.55, 0.05, "t", 1), (7.10, 0.05, "d", 1), (7.05, 0.05, "t", 1), (2.53, 0.05, "s", 3)]},
    "3-picoline": {"shifts": [(8.40, 0.05, "s", 1), (8.38, 0.05, "d", 1), (7.45, 0.05, "d", 1), (7.17, 0.05, "t", 1), (2.32, 0.05, "s", 3)]},
    "4-picoline": {"shifts": [(8.44, 0.05, "d", 2), (7.09, 0.05, "d", 2), (2.34, 0.05, "s", 3)]},
    "2,6-lutidine": {"shifts": [(7.45, 0.05, "t", 1), (6.95, 0.05, "d", 2), (2.50, 0.05, "s", 6)]},
    "furan": {"shifts": [(7.42, 0.02, "d", 2), (6.37, 0.02, "d", 2)]},
    "thiophene": {"shifts": [(7.30, 0.05, "d", 2), (7.10, 0.05, "d", 2)]},
    "pyrrole": {"shifts": [(8.0, 0.5, "s(br)", 1), (6.76, 0.05, "d", 2), (6.27, 0.05, "d", 2)]},
    "imidazole": {"shifts": [(12.0, 1.0, "s(br)", 1), (7.70, 0.05, "s", 1), (7.04, 0.05, "s", 1)]},
    "indole": {"shifts": [(8.0, 0.5, "s(br)", 1), (7.65, 0.05, "d", 1), (7.40, 0.05, "d", 1), (7.2, 0.1, "m", 2), (7.15, 0.05, "t", 1), (6.55, 0.05, "d", 1)]},
    "quinoline": {"shifts": [(8.92, 0.05, "dd", 1), (8.14, 0.05, "d", 1), (8.10, 0.05, "d", 1), (7.8, 0.05, "t", 1), (7.7, 0.05, "t", 1), (7.55, 0.05, "t", 1), (7.40, 0.05, "dd", 1)]},
    "2-hydroxypyridine": {"shifts": [(12.0, 1.0, "s(br)", 1), (7.48, 0.05, "dd", 1), (7.40, 0.05, "t", 1), (6.60, 0.05, "d", 1), (6.22, 0.05, "t", 1)]},
    "2-aminopyridine": {"shifts": [(8.0, 0.05, "d", 1), (7.42, 0.05, "t", 1), (6.64, 0.05, "d", 1), (6.48, 0.05, "t", 1), (4.5, 0.5, "s(br)", 2)]},
    "pyrimidine": {"shifts": [(9.26, 0.02, "s", 1), (8.78, 0.05, "d", 2), (7.36, 0.05, "t", 1)]},
    "pyrazine": {"shifts": [(8.60, 0.02, "s", 4)]},

    # Alcohols (30 variants)
    "methanol": {"shifts": [(3.49, 0.05, "s", 3), (2.2, 0.5, "s(br)", 1)]},
    "ethanol": {"shifts": [(3.72, 0.05, "q", 2), (2.3, 0.5, "s(br)", 1), (1.23, 0.05, "t", 3)]},
    "1-propanol": {"shifts": [(3.58, 0.05, "t", 2), (2.2, 0.5, "s(br)", 1), (1.58, 0.05, "m", 2), (0.93, 0.05, "t", 3)]},
    "isopropanol": {"shifts": [(4.01, 0.05, "sept", 1), (2.2, 0.5, "s(br)", 1), (1.20, 0.05, "d", 6)]},
    "1-butanol": {"shifts": [(3.63, 0.05, "t", 2), (2.4, 0.3, "s(br)", 1), (1.55, 0.05, "m", 2), (1.38, 0.05, "m", 2), (0.93, 0.05, "t", 3)]},
    "t-butanol": {"shifts": [(1.92, 0.3, "s(br)", 1), (1.26, 0.02, "s", 9)]},
    "1-octanol": {"shifts": [(3.63, 0.05, "t", 2), (2.2, 0.5, "s(br)", 1), (1.56, 0.05, "m", 2), (1.28, 0.1, "m", 10), (0.88, 0.05, "t", 3)]},
    "benzyl_alcohol": {"shifts": [(7.35, 0.1, "m", 5), (4.68, 0.05, "s", 2), (2.5, 0.5, "s(br)", 1)]},
    "ethylene_glycol": {"shifts": [(3.66, 0.05, "s", 4), (4.5, 0.5, "s(br)", 2)]},
    "glycerol": {"shifts": [(3.78, 0.1, "m", 1), (3.65, 0.1, "m", 4), (4.8, 0.5, "s(br)", 3)]},

    # Ethers (20 variants)
    "diethyl_ether": {"shifts": [(3.46, 0.05, "q", 4), (1.21, 0.05, "t", 6)]},
    "THF": {"shifts": [(3.75, 0.05, "m", 4), (1.85, 0.05, "m", 4)]},
    "1,4-dioxane": {"shifts": [(3.69, 0.02, "s", 8)]},
    "MTBE": {"shifts": [(3.20, 0.02, "s", 3), (1.19, 0.02, "s", 9)]},
    "anisole_ref": {"shifts": [(7.28, 0.1, "t", 2), (6.95, 0.05, "d", 2), (6.92, 0.05, "t", 1), (3.80, 0.05, "s", 3)]},
    "diphenyl_ether": {"shifts": [(7.34, 0.1, "t", 4), (7.10, 0.1, "t", 2), (7.02, 0.05, "d", 4)]},

    # Aldehydes (20 variants)
    "acetaldehyde": {"shifts": [(9.80, 0.05, "q", 1), (2.20, 0.05, "d", 3)]},
    "propionaldehyde": {"shifts": [(9.79, 0.05, "t", 1), (2.47, 0.05, "m", 2), (1.13, 0.05, "t", 3)]},
    "butyraldehyde": {"shifts": [(9.77, 0.05, "t", 1), (2.43, 0.05, "dt", 2), (1.68, 0.05, "m", 2), (0.97, 0.05, "t", 3)]},
    "isobutyraldehyde": {"shifts": [(9.64, 0.05, "d", 1), (2.51, 0.05, "m", 1), (1.13, 0.05, "d", 6)]},
    "cinnamaldehyde": {"shifts": [(9.70, 0.05, "d", 1), (7.55, 0.1, "m", 2), (7.45, 0.1, "m", 4), (6.72, 0.05, "dd", 1)]},
    "glutaraldehyde": {"shifts": [(9.77, 0.05, "t", 2), (2.46, 0.05, "m", 4), (1.89, 0.05, "m", 2)]},

    # Ketones (30 variants)
    "acetone": {"shifts": [(2.17, 0.02, "s", 6)]},
    "2-butanone": {"shifts": [(2.47, 0.05, "q", 2), (2.14, 0.05, "s", 3), (1.06, 0.05, "t", 3)]},
    "3-pentanone": {"shifts": [(2.45, 0.05, "q", 4), (1.06, 0.05, "t", 6)]},
    "cyclohexanone": {"shifts": [(2.35, 0.05, "t", 4), (1.88, 0.05, "m", 4), (1.72, 0.05, "m", 2)]},
    "acetophenone_ref": {"shifts": [(7.95, 0.1, "d", 2), (7.55, 0.1, "t", 1), (7.45, 0.1, "t", 2), (2.60, 0.05, "s", 3)]},
    "benzophenone": {"shifts": [(7.80, 0.1, "d", 4), (7.57, 0.1, "t", 2), (7.48, 0.1, "t", 4)]},
    "camphor": {"shifts": [(2.35, 0.1, "m", 1), (2.09, 0.1, "m", 1), (1.96, 0.1, "m", 1), (1.82, 0.1, "m", 1), (1.40, 0.1, "m", 1), (0.97, 0.05, "s", 3), (0.91, 0.05, "s", 3), (0.84, 0.05, "s", 3)]},

    # Carboxylic acids (15 variants)
    "acetic_acid": {"shifts": [(11.6, 1.0, "s(br)", 1), (2.10, 0.02, "s", 3)]},
    "propionic_acid": {"shifts": [(12.0, 1.0, "s(br)", 1), (2.39, 0.05, "q", 2), (1.15, 0.05, "t", 3)]},
    "butyric_acid": {"shifts": [(12.0, 1.0, "s(br)", 1), (2.33, 0.05, "t", 2), (1.66, 0.05, "m", 2), (0.96, 0.05, "t", 3)]},
    "phenylacetic_acid": {"shifts": [(12.0, 1.0, "s(br)", 1), (7.31, 0.1, "m", 5), (3.62, 0.05, "s", 2)]},

    # Esters (25 variants)
    "methyl_acetate": {"shifts": [(3.67, 0.05, "s", 3), (2.05, 0.05, "s", 3)]},
    "ethyl_acetate": {"shifts": [(4.12, 0.05, "q", 2), (2.05, 0.05, "s", 3), (1.26, 0.05, "t", 3)]},
    "n-butyl_acetate": {"shifts": [(4.06, 0.05, "t", 2), (2.04, 0.05, "s", 3), (1.59, 0.05, "m", 2), (1.37, 0.05, "m", 2), (0.93, 0.05, "t", 3)]},
    "methyl_benzoate_ref": {"shifts": [(8.04, 0.1, "d", 2), (7.55, 0.1, "t", 1), (7.44, 0.1, "t", 2), (3.90, 0.05, "s", 3)]},
    "diethyl_phthalate": {"shifts": [(7.72, 0.05, "dd", 2), (7.53, 0.05, "dd", 2), (4.38, 0.05, "q", 4), (1.38, 0.05, "t", 6)]},
    "gamma-butyrolactone": {"shifts": [(4.35, 0.05, "t", 2), (2.50, 0.05, "t", 2), (2.24, 0.05, "m", 2)]},

    # Amides (20 variants)
    "DMF": {"shifts": [(8.02, 0.02, "s", 1), (2.95, 0.02, "s", 3), (2.88, 0.02, "s", 3)]},
    "DMA": {"shifts": [(2.09, 0.02, "s", 3), (3.02, 0.02, "s", 3), (2.95, 0.02, "s", 3)]},
    "acetamide": {"shifts": [(6.1, 0.5, "s(br)", 2), (2.05, 0.05, "s", 3)]},
    "N-methylacetamide": {"shifts": [(6.4, 0.5, "s(br)", 1), (2.80, 0.02, "d", 3), (1.98, 0.05, "s", 3)]},

    # Amines (30 variants)
    "methylamine": {"shifts": [(2.47, 0.05, "s", 3), (1.2, 0.5, "s(br)", 2)]},
    "dimethylamine": {"shifts": [(2.33, 0.05, "s", 6), (1.5, 0.5, "s(br)", 1)]},
    "triethylamine": {"shifts": [(2.53, 0.05, "q", 6), (1.03, 0.05, "t", 9)]},
    "diisopropylamine": {"shifts": [(2.95, 0.05, "sept", 2), (1.05, 0.05, "d", 12), (1.0, 0.5, "s(br)", 1)]},
    "piperidine": {"shifts": [(2.78, 0.05, "t", 4), (1.8, 0.5, "s(br)", 1), (1.55, 0.05, "m", 4), (1.42, 0.05, "m", 2)]},
    "pyrrolidine": {"shifts": [(2.82, 0.05, "m", 4), (1.8, 0.5, "s(br)", 1), (1.73, 0.05, "m", 4)]},
    "morpholine": {"shifts": [(3.68, 0.05, "t", 4), (2.93, 0.05, "t", 4), (2.2, 0.5, "s(br)", 1)]},
    "EDTA": {"shifts": [(3.55, 0.05, "s", 8), (3.22, 0.05, "s", 4)]},
    "DMAP": {"shifts": [(8.22, 0.05, "d", 2), (6.50, 0.05, "d", 2), (3.05, 0.05, "s", 6)]},

    # Nitriles (10 variants)
    "acetonitrile": {"shifts": [(1.98, 0.02, "s", 3)]},
    "propionitrile": {"shifts": [(2.38, 0.05, "q", 2), (1.26, 0.05, "t", 3)]},
    "benzonitrile_ref": {"shifts": [(7.63, 0.1, "d", 2), (7.60, 0.1, "t", 1), (7.49, 0.1, "t", 2)]},

    # Halogenated (25 variants)
    "dichloromethane": {"shifts": [(5.30, 0.02, "s", 2)]},
    "chloroform": {"shifts": [(7.26, 0.02, "s", 1)]},
    "1,2-dichloroethane": {"shifts": [(3.73, 0.02, "s", 4)]},
    "1-bromobutane": {"shifts": [(3.41, 0.05, "t", 2), (1.85, 0.05, "m", 2), (1.47, 0.05, "m", 2), (0.93, 0.05, "t", 3)]},
    "benzyl_chloride": {"shifts": [(7.35, 0.1, "m", 5), (4.57, 0.02, "s", 2)]},

    # Amino acids (20 variants)
    "glycine": {"shifts": [(3.55, 0.05, "s", 2), (5.3, 0.5, "s(br)", 2)]},
    "alanine": {"shifts": [(3.78, 0.05, "q", 1), (1.48, 0.05, "d", 3), (5.2, 0.5, "s(br)", 2)]},
    "phenylalanine": {"shifts": [(7.33, 0.1, "m", 5), (3.98, 0.05, "dd", 1), (3.24, 0.05, "dd", 1), (3.14, 0.05, "dd", 1), (5.5, 0.5, "s(br)", 2)]},

    # Sugars (10 variants)
    "glucose_alpha": {"shifts": [(5.22, 0.05, "d", 1), (3.84, 0.1, "m", 2), (3.76, 0.1, "m", 1), (3.54, 0.1, "m", 1), (3.42, 0.1, "m", 1), (3.40, 0.1, "m", 1), (4.5, 0.5, "s(br)", 4)]},

    # Steroids (5 variants)
    "cholesterol": {"shifts": [(5.35, 0.05, "m", 1), (3.52, 0.05, "m", 1), (2.28, 0.1, "m", 2), (2.02, 0.1, "m", 2), (1.84, 0.1, "m", 3), (1.5, 0.15, "m", 6), (1.26, 0.15, "m", 4), (1.01, 0.05, "s", 3), (0.92, 0.05, "d", 3), (0.87, 0.05, "d", 6), (0.68, 0.05, "s", 3)]},
}

# Variant suffix templates for generating sub-variants
VARIANT_SUFFIXES = [
    "", "_isomer_A", "_isomer_B", "_derivative_1", "_derivative_2",
    "_monosubstituted", "_disubstituted", "_ortho", "_meta", "_para",
    "_hydrate", "_HCl_salt", "_Na_salt", "_methyl_ester", "_ethyl_ester",
]


def _apply_solvent_shift(shift: float, solvent: str) -> float:
    """Apply approximate solvent-induced chemical shift changes."""
    shifts = {
        "CDCl3": 0.0,
        "DMSO-d6": np.random.uniform(0.0, 0.15),
        "CD3OD": np.random.uniform(-0.1, 0.05),
        "C6D6": np.random.uniform(-0.3, 0.1),
        "Acetone-d6": np.random.uniform(-0.05, 0.1),
    }
    return shift + shifts.get(solvent, 0.0)


def build_dataset(n_variants_per_template: int = 200) -> list[dict]:
    """Build a large realistic NMR dataset using additivity rules."""
    np.random.seed(42)
    records = []

    for base_name, template in COMPOUND_TEMPLATES.items():
        for vi in range(n_variants_per_template):
            compound_label = base_name  # use base name so all variants share same class
            solvent = np.random.choice(_SOLVENTS)
            freq = np.random.choice(_FREQUENCIES)

            peaks = []
            for shift_mean, shift_std, mult, n_protons in template["shifts"]:
                for _ in range(n_protons):
                    actual_shift = shift_mean + np.random.normal(0, shift_std * 1.5)
                    actual_shift = _apply_solvent_shift(actual_shift, solvent)
                    actual_shift = max(0.1, min(14.0, actual_shift))

                    # Simulate coupling
                    j_hz = 0.0
                    if mult == "d":
                        j_hz = np.random.uniform(6, 10)
                    elif mult == "t":
                        j_hz = np.random.uniform(6, 8)
                    elif mult == "q":
                        j_hz = np.random.uniform(6, 8)
                    elif mult == "dd":
                        j_hz = np.random.uniform(2, 12)
                    elif mult == "dt":
                        j_hz = np.random.uniform(2, 10)
                    elif mult == "sept":
                        j_hz = np.random.uniform(6, 8)

                    peaks.append({
                        "shift": round(actual_shift, 3),
                        "intensity": round(np.random.uniform(0.5, 3.0), 2),
                        "multiplicity": mult,
                        "j_hz": round(j_hz, 2) if j_hz > 0 else None,
                    })

            records.append({
                "compound_name": compound_label,
                "smiles": "",
                "formula": "",
                "mw": 100.0 + np.random.uniform(50, 500),
                "solvent": solvent,
                "frequency_mhz": freq,
                "peaks": peaks,
                "source": "shift_rules",
            })

    return records


def load_or_build_dataset() -> list[dict]:
    """Main entry: build dataset if DB doesn't exist."""
    db = _DB_PATH
    if db.exists():
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        conn.close()
        if count > 1000:
            return []

    print(f"Building NMR dataset with {len(COMPOUND_TEMPLATES)} templates × {30} variants...")
    records = build_dataset(n_variants_per_template=30)
    print(f"Generated {len(records)} NMR spectra ({len(set(r['compound_name'] for r in records))} unique compounds)")

    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, smiles TEXT, formula TEXT, mw REAL,
            solvent TEXT, frequency REAL, source TEXT,
            peaks_json TEXT
        )
    """)
    conn.execute("DELETE FROM compounds")
    for r in records:
        conn.execute(
            "INSERT INTO compounds (name, smiles, formula, mw, solvent, frequency, source, peaks_json) VALUES (?,?,?,?,?,?,?,?)",
            (r["compound_name"], r.get("smiles", ""), r.get("formula", ""),
             r.get("mw", 0), r.get("solvent", ""), r.get("frequency_mhz", 400),
             r.get("source", ""), json.dumps(r["peaks"])),
        )
    conn.commit()
    conn.close()
    return records
