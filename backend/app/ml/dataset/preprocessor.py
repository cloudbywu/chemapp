"""Feature extraction: 1H NMR spectrum → 140-dim vector + peak sequence."""

import numpy as np

_N_BINS = 120
_BIN_RANGE = (0.0, 12.0)
_REGIONS = [
    ("alkyl", 0.0, 4.5),
    ("hetero_attached", 3.0, 5.0),
    ("olefin", 4.5, 7.0),
    ("aromatic", 6.5, 9.0),
    ("aldehyde", 9.0, 10.5),
    ("carboxylic", 10.5, 12.5),
    ("full_range", 0.0, 12.0),
]


def spectrum_to_bins(peaks: list[dict], solvent: str = "CDCl3") -> np.ndarray:
    """Convert peak list to 120-bin histogram (0-12 ppm, 0.1 ppm bins)."""
    hist = np.zeros(_N_BINS, dtype=np.float32)
    for p in peaks:
        shift = p.get("shift", 0)
        intensity = p.get("intensity", 1.0)
        idx = int((shift - _BIN_RANGE[0]) / 0.1)
        if 0 <= idx < _N_BINS:
            hist[idx] += float(intensity)
    if hist.max() > 0:
        hist = hist / hist.max()
    return hist


def spectrum_to_region_integrals(peaks: list[dict]) -> np.ndarray:
    """Calculate relative integration per chemical shift region."""
    features = np.zeros(len(_REGIONS), dtype=np.float32)
    total = 0.0
    for p in peaks:
        shift = p.get("shift", 0)
        intensity = p.get("intensity", 1.0)
        for i, (_, lo, hi) in enumerate(_REGIONS):
            if lo <= shift < hi:
                features[i] += float(intensity)
        total += float(intensity)
    if total > 0:
        features = features / total
    return features


def spectrum_to_multiplet_features(peaks: list[dict]) -> np.ndarray:
    """Count multiplet types and average coupling by region."""
    mult_map = {
        "s": 0, "d": 1, "t": 2, "q": 3,
        "m": 4, "dd": 5, "dt": 6, "td": 7,
        "s(br)": 0,
    }
    features = np.zeros(8, dtype=np.float32)
    region_counts = np.zeros(4, dtype=np.float32)
    region_j = np.zeros(4, dtype=np.float32)
    region_j_count = np.zeros(4, dtype=np.float32)

    for p in peaks:
        mult = str(p.get("multiplicity", "s")).lower()
        features[mult_map.get(mult, 4)] += 1

        shift = p.get("shift", 0)
        if 0 <= shift < 3:
            ri = 0
        elif 3 <= shift < 6:
            ri = 1
        elif 6 <= shift < 8:
            ri = 2
        else:
            ri = 3
        region_counts[ri] += 1

        j = p.get("j_hz", 0) or 0
        if j > 0:
            region_j[ri] += j
            region_j_count[ri] += 1

    total = max(len(peaks), 1)
    features = features / total

    for i in range(4):
        if region_j_count[i] > 0:
            region_j[i] = region_j[i] / region_j_count[i]

    return np.concatenate([features, region_counts / max(region_counts.max(), 1), region_j])


def extract_features(peaks: list[dict], solvent: str = "CDCl3") -> np.ndarray:
    """Extract full 140-dim feature vector from peak list."""
    bins = spectrum_to_bins(peaks, solvent)
    integrals = spectrum_to_region_integrals(peaks)
    multiplets = spectrum_to_multiplet_features(peaks)
    return np.concatenate([bins, integrals, multiplets])


def peaks_to_sequence(peaks: list[dict], max_len: int = 20) -> np.ndarray:
    """Convert peak list to fixed-length sequence for transformer input."""
    seq = np.zeros((max_len, 4), dtype=np.float32)
    sorted_peaks = sorted(peaks, key=lambda p: p.get("shift", 0))
    for i, p in enumerate(sorted_peaks[:max_len]):
        mult_map = {"s": 0, "d": 0.25, "t": 0.5, "q": 0.75, "m": 0.5, "dd": 0.3, "dt": 0.6, "td": 0.7}
        seq[i, 0] = p.get("shift", 0) / 12.0
        seq[i, 1] = min(p.get("intensity", 1.0) / 5.0, 1.0)
        seq[i, 2] = mult_map.get(str(p.get("multiplicity", "s")).lower(), 0.5)
        seq[i, 3] = min((p.get("j_hz", 0) or 0) / 20.0, 1.0)
    return seq
