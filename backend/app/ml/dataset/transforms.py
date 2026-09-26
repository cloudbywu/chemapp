"""PyTorch Dataset and DataLoader for NMR data (legacy prototype chain).

Split hygiene: records are partitioned by compound class with a seeded
``torch.Generator`` BEFORE any augmentation is applied, and augmented
variants are generated only for the training subset.  The previous
implementation augmented inside the dataset constructor and then shuffled
all records into random splits, leaking augmented copies of val/test
compounds into every partition.
"""

import json
import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.ml.dataset.downloader import db_path
from app.ml.dataset.preprocessor import extract_features, peaks_to_sequence
from app.ml.dataset.augment import generate_augmented

DEFAULT_SPLIT_SEED = 42
AUGMENT_VARIANTS_PER_RECORD = 1


class NMRCompoundDataset(Dataset):
    """Un-augmented compound records with deterministic class indexing.

    Class indices are assigned from ``sorted(set(names))`` so the mapping is
    stable regardless of database row order.  ``augment`` is accepted for
    backward compatibility and ignored: augmentation is applied by
    ``create_dataloaders`` to the training subset only, after the split.
    """

    def __init__(self, augment: bool = False):
        self.records: list = []
        self.peak_records: list = []
        db = db_path()
        if not db.exists():
            from app.ml.dataset.downloader import load_or_build_dataset
            load_or_build_dataset()

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT name, peaks_json FROM compounds").fetchall()
        conn.close()

        self.class_names = sorted({name for name, _ in rows})
        class_labels = {name: i for i, name in enumerate(self.class_names)}
        self.n_classes = len(self.class_names)

        for name, peaks_json in rows:
            peaks = json.loads(peaks_json)
            label = class_labels[name]
            self.peak_records.append((peaks, label))
            self.records.append(
                (extract_features(peaks), peaks_to_sequence(peaks), label)
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        feat, seq, label = self.records[idx]
        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class _RecordDataset(Dataset):
    """Materialised (feat, seq, label) records; same item format as
    ``NMRCompoundDataset``."""

    def __init__(self, records: list):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        feat, seq, label = self.records[idx]
        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def _class_grouped_split(dataset: NMRCompoundDataset, train_ratio: float, seed: int):
    """Partition record indices by compound class with a seeded generator.

    Returns (train_indices, val_indices, test_indices); every record of one
    compound class lands in exactly one split, so no class appears in both
    training and validation/test.
    """
    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.peak_records):
        by_class.setdefault(label, []).append(idx)

    class_ids = sorted(by_class)
    n_classes = len(class_ids)
    perm = torch.randperm(n_classes, generator=generator).tolist()
    n_train = int(round(n_classes * train_ratio))
    n_val = int(round(n_classes * 0.1))
    if n_train + n_val > n_classes:
        n_val = max(n_classes - n_train, 0)

    train_classes = [class_ids[i] for i in perm[:n_train]]
    val_classes = [class_ids[i] for i in perm[n_train:n_train + n_val]]
    test_classes = [class_ids[i] for i in perm[n_train + n_val:]]

    train_indices = [i for c in train_classes for i in by_class[c]]
    val_indices = [i for c in val_classes for i in by_class[c]]
    test_indices = [i for c in test_classes for i in by_class[c]]
    return train_indices, val_indices, test_indices


def create_dataloaders(
    batch_size: int = 64,
    train_ratio: float = 0.8,
    augment: bool = True,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = NMRCompoundDataset(augment=False)
    train_indices, val_indices, test_indices = _class_grouped_split(
        dataset, train_ratio, seed
    )

    train_records = [dataset.records[i] for i in train_indices]
    if augment and train_indices:
        # Augment the training subset only, with an independent seeded stream
        # derived from the split seed so the whole pipeline is reproducible.
        rng = np.random.default_rng(seed)
        augmented = []
        for idx in train_indices:
            peaks, label = dataset.peak_records[idx]
            for variant in generate_augmented(
                peaks,
                n_variants=AUGMENT_VARIANTS_PER_RECORD,
                seed=int(rng.integers(0, 2**31 - 1)),
            ):
                augmented.append(
                    (extract_features(variant), peaks_to_sequence(variant), label)
                )
        train_records = train_records + augmented

    train_ds = _RecordDataset(train_records)
    val_ds = _RecordDataset([dataset.records[i] for i in val_indices])
    test_ds = _RecordDataset([dataset.records[i] for i in test_indices])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.n_classes