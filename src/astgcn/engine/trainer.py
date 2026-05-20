"""模型训练循环。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from astgcn.engine.checkpoint import save_checkpoint
from astgcn.losses import get_loss
from astgcn.metrics import compute_metrics


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _forward_model(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    output = model(batch["recent"], batch["daily"], batch["weekly"], return_components=False)
    if isinstance(output, dict):
        return output.get("prediction", output.get("pred"))
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _average_dict(items: list[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


class Trainer:
    """ASTGCN 训练器。

    参数:
        model: 待训练模型，前向接口为 ``forward(recent, daily, weekly, return_components=False)``。
        optimizer: PyTorch 优化器。
        train_loader: 训练 DataLoader，batch 需包含 ``recent``、``daily``、``weekly``、``target``、``t0``。
        val_loader: 验证 DataLoader，可为 ``None``。
        criterion: 损失函数对象；为 ``None`` 时由 ``loss_name`` 创建。
        device: 训练设备。
        loss_name: 损失名称，传给 ``get_loss``。
        mask_value: 指标和 masked 损失忽略的目标值；为 ``None`` 时指标不做掩码。
        logger: 可选 logger。
        checkpoint_dir: checkpoint 保存目录；为 ``None`` 时不自动保存。
        config: 训练配置，保存 checkpoint 时写入。
        scaler: 标准化器，保存 checkpoint 时写入 mean/std。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        val_loader: Any | None = None,
        criterion: nn.Module | None = None,
        device: str | torch.device = "cpu",
        loss_name: str = "mae",
        mask_value: float | None = None,
        logger: Any | None = None,
        checkpoint_dir: str | Path | None = None,
        config: Dict[str, Any] | None = None,
        scaler: Any | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.criterion = criterion if criterion is not None else get_loss(loss_name, mask_value=mask_value or 0.0)
        self.mask_value = mask_value
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.config = config
        self.scaler = scaler
        self.model.to(self.device)

    def train_one_epoch(self, max_batches: int | None = None) -> Dict[str, float]:
        """训练一个 epoch。

        参数:
            max_batches: smoke test 使用的最大 batch 数；为 ``None`` 时遍历完整训练集。

        返回:
            包含 ``loss``、``mae``、``rmse``、``mape`` 的平均指标字典。
        """
        self.model.train()
        logs: list[Dict[str, float]] = []
        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch(batch, self.device)
            target = batch["target"].float()

            self.optimizer.zero_grad(set_to_none=True)
            pred = _forward_model(self.model, batch).float()
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            metrics = compute_metrics(pred.detach(), target.detach(), mask_value=self.mask_value)
            metrics["loss"] = float(loss.detach().cpu().item())
            logs.append(metrics)
        return _average_dict(logs)

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> Dict[str, float]:
        """在验证集上评估模型。

        参数:
            max_batches: smoke test 使用的最大 batch 数；为 ``None`` 时遍历完整验证集。

        返回:
            包含 ``loss``、``mae``、``rmse``、``mape`` 的平均指标字典；没有验证集时返回空字典。
        """
        if self.val_loader is None:
            return {}
        self.model.eval()
        logs: list[Dict[str, float]] = []
        for batch_idx, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch(batch, self.device)
            target = batch["target"].float()
            pred = _forward_model(self.model, batch).float()
            loss = self.criterion(pred, target)
            metrics = compute_metrics(pred, target, mask_value=self.mask_value)
            metrics["loss"] = float(loss.detach().cpu().item())
            logs.append(metrics)
        return _average_dict(logs)

    def fit(
        self,
        epochs: int,
        patience: int | None = None,
        monitor: str = "mae",
        max_batches: int | None = None,
    ) -> Dict[str, Any]:
        """执行完整训练流程。

        参数:
            epochs: 最大训练轮数。
            patience: early stopping 容忍轮数；为 ``None`` 时不启用。
            monitor: 用于选择最优模型的验证指标，默认 ``mae``，越小越好。
            max_batches: smoke test 使用的每个 epoch 最大 batch 数。

        返回:
            训练历史字典，包含每轮 train/val 指标、best_epoch、best_metric。
        """
        history: Dict[str, Any] = {"train": [], "val": [], "best_epoch": None, "best_metric": None}
        best_metric = float("inf")
        best_epoch = 0
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(max_batches=max_batches)
            val_metrics = self.validate(max_batches=max_batches)
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            current = val_metrics.get(monitor, train_metrics.get(monitor))
            if current is not None and current < best_metric:
                best_metric = current
                best_epoch = epoch
                bad_epochs = 0
                if self.checkpoint_dir is not None:
                    save_checkpoint(
                        self.checkpoint_dir / "best.pt",
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=best_metric,
                        config=self.config,
                        scaler=self.scaler,
                    )
            else:
                bad_epochs += 1

            if self.logger is not None:
                self.logger.info(
                    "epoch=%s train=%s val=%s best_%s=%.6f",
                    epoch,
                    train_metrics,
                    val_metrics,
                    monitor,
                    best_metric,
                )

            if patience is not None and bad_epochs >= patience:
                if self.logger is not None:
                    self.logger.info("early stopping at epoch=%s", epoch)
                break

        history["best_epoch"] = best_epoch
        history["best_metric"] = None if best_metric == float("inf") else best_metric
        return history
