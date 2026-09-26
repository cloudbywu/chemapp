"""Chemical ontology: compound class hierarchy for NMR interpretation."""

FUNCTIONAL_GROUP_REGIONS = {
    "alkyl_CH3": {"range": (0.7, 1.2), "mult": "m"},
    "alkyl_CH2": {"range": (1.1, 1.5), "mult": "m"},
    "alkyl_CH": {"range": (1.5, 2.0), "mult": "m"},
    "alpha_to_carbonyl": {"range": (2.0, 2.5), "mult": "m"},
    "acetyl_CH3": {"range": (2.0, 2.2), "mult": "s"},
    "alpha_to_aromatic": {"range": (2.3, 2.8), "mult": "m"},
    "terminal_alkyne": {"range": (2.5, 3.0), "mult": "s"},
    "methoxy": {"range": (3.3, 3.5), "mult": "s"},
    "alpha_to_oxygen": {"range": (3.5, 4.5), "mult": "m"},
    "alpha_to_nitrogen": {"range": (2.5, 3.5), "mult": "m"},
    "hydroxyl": {"range": (1.0, 5.0), "mult": "s(br)"},
    "olefin": {"range": (4.5, 6.5), "mult": "m"},
    "aromatic": {"range": (6.5, 8.5), "mult": "m"},
    "heteroaromatic": {"range": (6.0, 9.0), "mult": "d,d,t"},
    "aldehyde": {"range": (9.5, 10.2), "mult": "s"},
    "carboxylic_acid": {"range": (10.5, 12.5), "mult": "s(br)"},
    "phenol": {"range": (4.5, 8.0), "mult": "s(br)"},
    "amide_NH": {"range": (5.0, 8.5), "mult": "s(br)"},
}

COMPOUND_CLASSES = {
    "alkanes": {
        "parent": "hydrocarbons",
        "features": {"alkyl_CH3": True, "alkyl_CH2": True},
    },
    "alkenes": {
        "parent": "hydrocarbons",
        "features": {"olefin": True},
    },
    "aromatic_hydrocarbons": {
        "parent": "hydrocarbons",
        "features": {"aromatic": True},
    },
    "alcohols": {
        "parent": "oxygen_compounds",
        "features": {"hydroxyl": True, "alpha_to_oxygen": True},
    },
    "ethers": {
        "parent": "oxygen_compounds",
        "features": {"alpha_to_oxygen": True, "methoxy": False},
    },
    "aldehydes": {
        "parent": "carbonyl_compounds",
        "features": {"aldehyde": True},
    },
    "ketones": {
        "parent": "carbonyl_compounds",
        "features": {"alpha_to_carbonyl": True},
    },
    "esters": {
        "parent": "carbonyl_compounds",
        "features": {"alpha_to_oxygen": True, "alpha_to_carbonyl": True},
    },
    "carboxylic_acids": {
        "parent": "carbonyl_compounds",
        "features": {"carboxylic_acid": True},
    },
    "amides": {
        "parent": "nitrogen_compounds",
        "features": {"amide_NH": True, "alpha_to_carbonyl": True},
    },
    "amines": {
        "parent": "nitrogen_compounds",
        "features": {"alpha_to_nitrogen": True},
    },
    "pyridines": {
        "parent": "heterocycles",
        "features": {"heteroaromatic": True},
    },
    "furans": {
        "parent": "heterocycles",
        "features": {"heteroaromatic": True},
    },
}


def detect_functional_groups(peaks: list[dict]) -> list[dict]:
    """Detect functional groups from peak patterns."""
    results = []
    for group_name, params in FUNCTIONAL_GROUP_REGIONS.items():
        lo, hi = params["range"]
        matched = [p for p in peaks if lo <= p.get("shift", 0) < hi]
        confidence = min(len(matched) / 3.0, 1.0)
        if confidence > 0.2:
            results.append({
                "group": group_name,
                "confidence": round(confidence, 3),
                "matched_peaks": len(matched),
            })
    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:10]


def classify_compound_class(peaks: list[dict]) -> list[dict]:
    """Classify compound into chemical classes using rule-based heuristics."""
    regions_found = set()
    for p in peaks:
        shift = p.get("shift", 0)
        for group_name, params in FUNCTIONAL_GROUP_REGIONS.items():
            if params["range"][0] <= shift < params["range"][1]:
                regions_found.add(group_name)

    results = []
    for class_name, info in COMPOUND_CLASSES.items():
        score = 0
        total = 0
        for feat, required in info["features"].items():
            total += 1
            if feat in regions_found:
                score += 1 if required else 0
            elif not required:
                score += 1
        if total > 0:
            confidence = score / total
            if confidence > 0.3:
                results.append({
                    "class_name": class_name,
                    "parent_class": info["parent"],
                    "confidence": round(confidence, 3),
                })
    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:5]
