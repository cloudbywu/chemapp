"""Hygiene and correctness tests for the legacy ML prototype chain.

Covers the 2026-08 ML review fixes: masked-mean pooling in the transformer,
weight-directory consistency, class-grouped seeded splits with augmentation
restricted to the training subset, seeded augmentation, best-checkpoint
restore before test evaluation, honest downloader documentation, solvent
dash normalisation, post-filter n_assigned_refs counting, retrieval
dead-code removal and predictor error handling.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch


class TestTransformerMaskedMean:
    def test_forward_with_mask_pools_only_valid_positions(self):
        from app.ml.models.transformer_nmr import NMRTransformerClassifier

        torch.manual_seed(0)
        model = NMRTransformerClassifier(n_classes=3, d_model=32, nhead=4, n_layers=1)
        model.eval()
        x = torch.randn(2, 8, 4)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 5:] = True
        mask[1, 3:] = True

        with torch.no_grad():
            out = model(x, mask)
            hidden = model.encoder(
                model.pos_encoder(model.input_proj(x)),
                src_key_padding_mask=mask,
            )
            valid = (~mask).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            ref = model.classifier(pooled)

        assert out.shape == (2, 3)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_padded_positions_do_not_change_output(self):
        from app.ml.models.transformer_nmr import NMRTransformerClassifier

        torch.manual_seed(1)
        model = NMRTransformerClassifier(n_classes=3, d_model=32, nhead=4, n_layers=1)
        model.eval()
        x = torch.randn(1, 8, 4)
        x_pad = x.clone()
        x_pad[0, 6:] = 100.0  # garbage inside padded region
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 6:] = True

        with torch.no_grad():
            out_a = model(x, mask)
            out_b = model(x_pad, mask)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_forward_without_mask_matches_plain_mean(self):
        from app.ml.models.transformer_nmr import NMRTransformerClassifier

        torch.manual_seed(2)
        model = NMRTransformerClassifier(n_classes=3, d_model=32, nhead=4, n_layers=1)
        model.eval()
        x = torch.randn(2, 6, 4)
        with torch.no_grad():
            out = model(x)
            hidden = model.encoder(model.pos_encoder(model.input_proj(x)))
            ref = model.classifier(hidden.mean(dim=1))
        assert torch.allclose(out, ref, atol=1e-5)


class TestWeightsDirConsistency:
    def test_all_weight_paths_resolve_to_app_ml_pretrained(self):
        import app.ml.models.ensemble as ensemble
        import app.ml.predictor as predictor
        import app.ml.retrieval as retrieval

        expected = Path(predictor.__file__).resolve().parent / "pretrained"
        assert expected.is_dir()
        assert ensemble._WEIGHTS_DIR == expected
        assert retrieval.CKPT_DIR == expected
        assert predictor._WEIGHTS_DIR == expected

    def test_retrieval_dead_code_removed(self):
        import app.ml.retrieval as retrieval

        assert not hasattr(retrieval, "NUCLEUS_MAP")
        assert not hasattr(retrieval, "MULTIPLICITY_MAP")
        source = inspect.getsource(retrieval)
        assert "cumsum" not in source
        assert "NUCLEUS_VOCAB" in source


def _make_compound_db(db_file: Path, entries: list[tuple[str, list[dict]]]) -> None:
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE compounds ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, smiles TEXT, formula TEXT, "
        "mw REAL, solvent TEXT, frequency REAL, source TEXT, peaks_json TEXT)"
    )
    for name, peaks in entries:
        conn.execute(
            "INSERT INTO compounds (name, smiles, formula, mw, solvent, frequency, "
            "source, peaks_json) VALUES (?,?,?,?,?,?,?,?)",
            (name, "", "", 0.0, "", 400.0, "test", json.dumps(peaks)),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def compound_db(tmp_path, monkeypatch):
    # Eight classes with one record each: grouped split gives
    # train=6 / val=1 / test=1 classes with the shipped ratios.
    entries = [
        (name, [{"shift": 1.0 + i, "intensity": 1.0, "multiplicity": "s"}])
        for i, name in enumerate([
            "delta", "alpha", "charlie", "bravo",
            "echo", "golf", "foxtrot", "hotel",
        ])
    ]
    db_file = tmp_path / "compounds.db"
    _make_compound_db(db_file, entries)
    monkeypatch.setattr("app.ml.dataset.downloader._DB_PATH", db_file)
    return db_file


def _loader_label_sets(loader):
    labels: set[int] = set()
    for _, _, batch_labels in loader:
        labels.update(int(v) for v in batch_labels.tolist())
    return labels


class TestTransformsSplitHygiene:
    def test_class_names_sorted_and_deterministic(self, compound_db):
        from app.ml.dataset.transforms import NMRCompoundDataset

        ds = NMRCompoundDataset(augment=False)
        assert ds.class_names == sorted(ds.class_names)
        assert ds.class_names == [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        ]
        assert ds.n_classes == 8
        assert len(ds) == 8

    def test_split_is_seeded_grouped_and_reproducible(self, compound_db):
        from app.ml.dataset import transforms

        def split_snapshot(seed: int):
            loaders = transforms.create_dataloaders(batch_size=8, augment=False, seed=seed)
            train_loader, val_loader, test_loader, _ = loaders
            return (
                _loader_label_sets(train_loader),
                _loader_label_sets(val_loader),
                _loader_label_sets(test_loader),
                len(train_loader.dataset),
                len(val_loader.dataset),
                len(test_loader.dataset),
            )

        snap_a = split_snapshot(123)
        snap_b = split_snapshot(123)
        assert snap_a == snap_b
        train_labels, val_labels, test_labels, n_train, n_val, n_test = snap_a
        assert train_labels.isdisjoint(val_labels)
        assert train_labels.isdisjoint(test_labels)
        assert val_labels.isdisjoint(test_labels)
        assert train_labels | val_labels | test_labels == set(range(8))
        assert (n_train, n_val, n_test) == (6, 1, 1)

    def test_augmentation_applies_to_train_subset_only(self, compound_db):
        from app.ml.dataset import transforms

        loaders = transforms.create_dataloaders(batch_size=8, augment=True, seed=123)
        train_loader, val_loader, test_loader, _ = loaders
        # 6 training records, each with AUGMENT_VARIANTS_PER_RECORD extra copy.
        assert len(train_loader.dataset) == 6 * (1 + transforms.AUGMENT_VARIANTS_PER_RECORD)
        assert len(val_loader.dataset) == 1
        assert len(test_loader.dataset) == 1

    def test_augmented_records_are_reproducible(self, compound_db):
        from app.ml.dataset import transforms

        def train_records(seed: int):
            loaders = transforms.create_dataloaders(batch_size=8, augment=True, seed=seed)
            return loaders[0].dataset.records

        rec_a = train_records(7)
        rec_b = train_records(7)
        assert len(rec_a) == len(rec_b)
        for (feat_a, seq_a, label_a), (feat_b, seq_b, label_b) in zip(rec_a, rec_b):
            assert label_a == label_b
            assert np.array_equal(feat_a, feat_b)
            assert np.array_equal(seq_a, seq_b)


class TestAugmentSeeded:
    def test_same_seed_is_reproducible(self):
        from app.ml.dataset.augment import generate_augmented, remove_peaks, shift_peaks

        peaks = [{"shift": float(i), "intensity": 1.0, "multiplicity": "s"} for i in range(10)]
        assert generate_augmented(peaks, n_variants=3, seed=7) == generate_augmented(
            peaks, n_variants=3, seed=7
        )
        assert shift_peaks(peaks, seed=3) == shift_peaks(peaks, seed=3)
        assert remove_peaks(peaks, drop_prob=0.5, seed=3) == remove_peaks(
            peaks, drop_prob=0.5, seed=3
        )

    def test_different_seed_differs(self):
        from app.ml.dataset.augment import generate_augmented

        peaks = [{"shift": float(i), "intensity": 1.0, "multiplicity": "s"} for i in range(20)]
        assert generate_augmented(peaks, n_variants=4, seed=1) != generate_augmented(
            peaks, n_variants=4, seed=2
        )

    def test_global_numpy_rng_is_not_touched(self):
        from app.ml.dataset import augment

        np.random.seed(2024)
        expected = np.random.random(5)
        np.random.seed(2024)
        augment.add_noise(np.zeros(8, dtype=np.float32), seed=11)
        augment.scale_spectrum(np.ones(6, dtype=np.float32), seed=12)
        augment.shift_peaks([{"shift": 1.0}], seed=13)
        augment.remove_peaks([{"shift": 1.0}], seed=14)
        augment.generate_augmented([{"shift": 1.0}], seed=15)
        assert np.array_equal(expected, np.random.random(5))

    def test_unseeded_call_still_works(self):
        from app.ml.dataset.augment import generate_augmented

        variants = generate_augmented([{"shift": 1.0, "intensity": 1.0, "multiplicity": "s"}])
        assert len(variants) == 4


class TestBestCheckpointRestore:
    def test_restore_overwrites_last_epoch_weights(self, tmp_path):
        from app.ml.forward_v1.gnn import ForwardGNN13C
        from app.ml.forward_v1.train import _restore_best_checkpoint

        model = ForwardGNN13C(in_dim=9, hidden=16, n_layers=1, condition_dim=13)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(
            {"state_dict": model.state_dict(), "hidden": 16, "layers": 1},
            tmp_path / "forward_gnn_13c_v1.pt",
        )
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        assert _restore_best_checkpoint(model, tmp_path, torch.device("cpu")) is True
        for key, value in model.state_dict().items():
            assert torch.equal(value, saved[key])

    def test_missing_checkpoint_returns_false(self, tmp_path):
        from app.ml.forward_v1.gnn import ForwardGNN13C
        from app.ml.forward_v1.train import _restore_best_checkpoint

        model = ForwardGNN13C(in_dim=9, hidden=16, n_layers=1, condition_dim=13)
        assert _restore_best_checkpoint(model, tmp_path, torch.device("cpu")) is False


class TestDownloaderDocumentation:
    def test_template_count_and_honest_docstring(self):
        import app.ml.dataset.downloader as downloader

        assert len(downloader.COMPOUND_TEMPLATES) == 112
        doc = downloader.__doc__ or ""
        assert "LEGACY" in doc
        assert "112" in doc
        assert "synthes" in doc.lower()
        assert "15,000" not in doc
        assert "400+" not in doc

    def test_predictor_marked_legacy(self):
        import app.ml.predictor as predictor

        assert "LEGACY" in (predictor.__doc__ or "")


class TestSolventNormalisation:
    def test_unicode_dashes_collapse_to_ascii(self):
        from app.ml.forward_v1.graph_data import _normalise_solvent

        assert _normalise_solvent("DMSO\u2013d6") == "dmso"  # en dash
        assert _normalise_solvent("DMSO\u2014d6") == "dmso"  # em dash
        assert _normalise_solvent("DMSO\u2212d6") == "dmso"  # minus sign
        assert _normalise_solvent("DMSO\u2010d6") == "dmso"  # unicode hyphen
        assert _normalise_solvent("DMSO-d6") == "dmso"  # ascii hyphen
        assert _normalise_solvent(None) == "unreported"
        assert _normalise_solvent("CDCl3") == "cdcl3"


def _make_index_db(db_file: Path) -> None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1")
    molblock = Chem.MolToMolBlock(mol)
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE molecules (
            id INTEGER PRIMARY KEY, molblock TEXT, inchi_key TEXT, smiles TEXT,
            formula TEXT, record_sha256 TEXT
        );
        CREATE TABLE spectra (
            id INTEGER PRIMARY KEY, molecule_id INTEGER, nucleus TEXT,
            measurement_kind TEXT, solvent TEXT, field_mhz REAL
        );
        CREATE TABLE peaks (
            id INTEGER PRIMARY KEY, spectrum_id INTEGER, shift REAL,
            atom_ref INTEGER, multiplicity TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO molecules (id, molblock, inchi_key, smiles, formula, record_sha256)"
        " VALUES (1, ?, 'KEY1', 'c1ccccc1', 'C6H6', 'sha')",
        (molblock,),
    )
    conn.execute(
        "INSERT INTO spectra (id, molecule_id, nucleus, measurement_kind, solvent, field_mhz)"
        " VALUES (1, 1, '13C', 'measured', 'CDCl3', 400.0)"
    )
    # Benzene: 6 aromatic carbons.  Two plausible shifts (kept) and four
    # implausible shifts that fail the 95-175 ppm aromatic gate (rejected).
    for atom_ref, shift in enumerate([128.0, 128.3, 200.0, 220.0, 250.0, 280.0], start=1):
        conn.execute(
            "INSERT INTO peaks (spectrum_id, shift, atom_ref, multiplicity) VALUES (1, ?, ?, 'S')",
            (shift, atom_ref),
        )
    conn.commit()
    conn.close()


class TestConsensusAssignedRefs:
    def test_assigned_refs_counted_after_quality_filter(self, tmp_path):
        from app.ml.forward_v1.graph_data import build_atom_dataset

        db_file = tmp_path / "index.sqlite"
        _make_index_db(db_file)
        samples = build_atom_dataset(db_file, max_molecules=10, seed=0, cache_path=None)
        # Post-filter counting keeps this molecule: 2 labels vs denominator
        # max(2, 2//2)=2 -> kept.  Pre-filter counting used denominator 6 and
        # dropped it (2 < max(2, 6//2)=3).
        assert len(samples) == 1
        assert int(samples[0].target_mask.sum().item()) == 2


class TestPredictorErrorHandling:
    def test_h1_failure_reported_out_of_band(self, monkeypatch):
        import app.ml.predictor as predictor

        def _raise():
            raise RuntimeError("weights missing")

        monkeypatch.setattr(predictor, "_ensure_model", _raise)
        result = predictor.predict_dual([{"shift": 1.0, "intensity": 1.0}], None)
        assert result["h1"] == []
        assert "weights missing" in result.get("h1_error", "")
        for candidate in result["combined"]:
            assert not str(candidate.get("compound_name", "")).startswith("error")

    def test_c13_failure_is_marked(self, monkeypatch):
        import app.ml.predictor as predictor

        class _Stub:
            def predict_top_k(self, feat, seq, k=5):
                return [0.9, 0.1], [0, 1]

        monkeypatch.setattr(predictor, "_ensure_model", _Stub)
        monkeypatch.setattr(predictor, "_CLASS_NAMES", ["a", "b"])

        def _raise():
            raise RuntimeError("13C weights missing")

        monkeypatch.setattr(predictor, "_ensure_model_13c", _raise)
        result = predictor.predict_dual(
            [{"shift": 1.0, "intensity": 1.0}],
            [{"shift": 20.0, "intensity": 1.0}],
        )
        assert result["h1"]
        assert result["c13"] == []
        assert "13C weights missing" in result.get("c13_error", "")


class TestEnsembleLazyDevice:
    def test_device_resolved_at_call_time(self, monkeypatch):
        import app.ml.models.ensemble as ensemble

        monkeypatch.setattr(ensemble, "_DEVICE", None)

        def _forbidden():
            raise AssertionError("CUDA probed")

        monkeypatch.setattr(torch.cuda, "is_available", _forbidden)
        with pytest.raises(AssertionError, match="CUDA probed"):
            ensemble.get_device()
        assert ensemble._DEVICE is None

    def test_classifier_constructs_on_cpu(self):
        import app.ml.models.ensemble as ensemble

        model = ensemble.NMREnsembleClassifier(2)
        assert next(model.parameters()).device.type == "cpu"