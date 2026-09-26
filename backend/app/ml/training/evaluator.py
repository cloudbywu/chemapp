"""Model evaluation utilities."""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from app.ml.models.ensemble import NMREnsembleClassifier, get_device


def evaluate_model(
    model: NMREnsembleClassifier,
    test_loader: DataLoader,
    class_names: list[str],
) -> dict:
    device = get_device()
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feat, seq, labels in test_loader:
            feat, seq = feat.to(device), seq.to(device)
            outputs = model(feat, seq)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    cm = confusion_matrix(all_labels, all_preds)
    top_errors = []
    if cm.shape[0] > 1:
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        for _ in range(min(5, cm_copy.size)):
            idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
            top_errors.append({
                "true_class": class_names[idx[0]] if idx[0] < len(class_names) else "?",
                "pred_class": class_names[idx[1]] if idx[1] < len(class_names) else "?",
                "count": int(cm_copy[idx]),
            })
            cm_copy[idx] = 0

    return {
        "accuracy": round(float(acc), 4),
        "n_samples": len(all_labels),
        "top_misclassifications": top_errors,
    }
