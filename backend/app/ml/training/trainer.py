"""Training loop with GPU support, AMP, and callbacks."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.models.ensemble import NMREnsembleClassifier, get_device


class Trainer:
    def __init__(
        self,
        model: NMREnsembleClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        epochs: int = 50,
        patience: int = 10,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_amp and self.device.type == "cuda"

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0

    def train(self, callback=None) -> dict[str, Any]:
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            train_loss = self._train_epoch()

            val_loss, val_acc = self._validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = self.current_epoch
                self.model.save("nmr_ensemble_best")

            if callback:
                callback(self)

            if self.current_epoch - self.best_epoch >= self.patience:
                break

        return self.get_results()

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for feat, seq, labels in self.train_loader:
            feat, seq, labels = feat.to(self.device), seq.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(feat, seq)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(feat, seq)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for feat, seq, labels in self.val_loader:
            feat, seq, labels = feat.to(self.device), seq.to(self.device), labels.to(self.device)
            outputs = self.model(feat, seq)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(self.val_loader), correct / total if total > 0 else 0.0

    def get_results(self) -> dict[str, Any]:
        return {
            "best_val_acc": round(self.best_val_acc, 4),
            "best_epoch": self.best_epoch,
            "epochs_trained": self.current_epoch,
            "train_loss_final": round(self.history["train_loss"][-1], 6) if self.history["train_loss"] else 0,
            "val_loss_final": round(self.history["val_loss"][-1], 6) if self.history["val_loss"] else 0,
            "history": self.history,
            "device": str(self.device),
        }
