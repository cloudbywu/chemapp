"""Data augmentation for NMR spectra (legacy prototype chain).

All randomness is drawn from ``numpy.random.default_rng(seed)`` so callers
can reproduce augmentations exactly; passing ``seed=None`` keeps the legacy
non-deterministic behaviour without ever touching numpy global RNG state.
"""

import numpy as np


def _add_noise_rng(spectrum: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, noise_level, spectrum.shape).astype(np.float32)
    return np.clip(spectrum + noise, 0, 1)


def _scale_spectrum_rng(spectrum: np.ndarray, factor_range: tuple, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(*factor_range)
    return np.clip(spectrum * factor, 0, 1)


def _shift_peaks_rng(peaks: list[dict], max_shift: float, rng: np.random.Generator) -> list[dict]:
    shifted = []
    for p in peaks:
        new_shift = p["shift"] + rng.uniform(-max_shift, max_shift)
        shifted.append({**p, "shift": round(max(0, min(12, new_shift)), 3)})
    return shifted


def _remove_peaks_rng(peaks: list[dict], drop_prob: float, rng: np.random.Generator) -> list[dict]:
    return [p for p in peaks if rng.random() > drop_prob]


def add_noise(spectrum: np.ndarray, noise_level: float = 0.02, seed: int | None = None) -> np.ndarray:
    return _add_noise_rng(spectrum, noise_level, np.random.default_rng(seed))


def scale_spectrum(spectrum: np.ndarray, factor_range: tuple = (0.9, 1.1), seed: int | None = None) -> np.ndarray:
    return _scale_spectrum_rng(spectrum, factor_range, np.random.default_rng(seed))


def shift_peaks(peaks: list[dict], max_shift: float = 0.05, seed: int | None = None) -> list[dict]:
    return _shift_peaks_rng(peaks, max_shift, np.random.default_rng(seed))


def remove_peaks(peaks: list[dict], drop_prob: float = 0.05, seed: int | None = None) -> list[dict]:
    return _remove_peaks_rng(peaks, drop_prob, np.random.default_rng(seed))


def generate_augmented(peaks: list[dict], n_variants: int = 4, seed: int | None = None) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    variants = []
    for _ in range(n_variants):
        variant = _shift_peaks_rng(peaks, max_shift=0.02, rng=rng)
        variant = _remove_peaks_rng(variant, drop_prob=0.03, rng=rng)
        variants.append(variant)
    return variants
