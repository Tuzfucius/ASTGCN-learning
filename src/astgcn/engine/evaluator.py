"""模型评估器。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import nn

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


def _inverse_target(data: np.ndarray, scaler: Any | None, target_dim: int) -> np.ndarray:
    if scaler is None:
        return data
    if hasattr(scaler, "inverse_transform_target"):
        return scaler.inverse_transform_target(data, target_dim=target_dim)
    mean = getattr(scaler, "mean", None)
    std = getattr(scaler, "std", None)
    if mean is None or std is None:
        return data
    return data * float(std[0, 0, target_dim]) + float(mean[0, 0, target_dim])


class Evaluator:
    """在反标准化尺度上评估模型。

    参数:
        model: 待评估模型，前向接口为 ``forward(recent, daily, weekly, return_components=False)``。
        data_loader: 测试或验证 DataLoader。
        scaler: 标准化器，需支持 ``inverse_transform_target`` 或暴露 ``mean/std``。
        device: 评估设备。
        target_dim: 目标特征维度。
        mask_value: 指标忽略的目标值；为 ``None`` 时不做掩码。
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Any,
        scaler: Any | None = None,
        device: str | torch.device = "cpu",
        target_dim: int = 0,
        mask_value: float | None = None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.scaler = scaler
        self.device = torch.device(device)
        self.target_dim = target_dim
        self.mask_value = mask_value
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> Dict[str, float]:
        """执行评估。

        参数:
            max_batches: smoke test 使用的最大 batch 数；为 ``None`` 时遍历完整数据集。

        返回:
            反标准化后的 ``mae``、``rmse``、``mape`` 指标字典。
        """
        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for batch_idx, batch in enumerate(self.data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch(batch, self.device)
            pred = _forward_model(self.model, batch)
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())

        if not predictions:
            return {}

        pred_all = _inverse_target(np.concatenate(predictions, axis=0), self.scaler, self.target_dim)
        target_all = _inverse_target(np.concatenate(targets, axis=0), self.scaler, self.target_dim)
        return compute_metrics(pred_all, target_all, mask_value=self.mask_value)
