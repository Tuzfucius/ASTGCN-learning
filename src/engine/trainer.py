"""训练流程封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class Trainer:
    """ASTGCN 训练器。

    本类只组织训练流程，不实现模型数学细节。
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        train_loader: Any,
        val_loader: Any,
        output_dir: str | Path,
        device: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(device)
        self.model.to(self.device)

    def fit(self, epochs: int, start_epoch: int = 0) -> dict[str, Any]:
        """执行完整训练流程。"""
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "best_epoch": None, "best_val_loss": best_val_loss}

        for epoch in range(start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_epoch"] = epoch
                history["best_val_loss"] = val_loss
                self.save_checkpoint(epoch, val_loss)

        return history

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_size = batch_x.shape[0]

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.loss_fn(predictions, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("训练集 DataLoader 为空。")
        return total_loss / total_samples

    def validate(self, epoch: int) -> float:
        """在验证集上计算 loss。"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size = batch_x.shape[0]
                predictions = self.model(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("验证集 DataLoader 为空。")
        return total_loss / total_samples

    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """保存模型权重。"""
        checkpoint_path = self.output_dir / "best.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """解析训练设备，CUDA 不可用时回退到 CPU。"""
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA 不可用，自动使用 CPU。")
            return torch.device("cpu")
        return torch.device(device)
