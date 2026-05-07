"""训练损失函数。"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


def masked_mae(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 MAE。

    mask 会除以自身均值，避免有效点比例影响 loss 尺度。
    """
    mask = _build_mask(labels, null_val)
    loss = torch.abs(preds - labels)
    return _apply_mask(loss, mask)


def masked_mse(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 MSE。"""
    mask = _build_mask(labels, null_val)
    loss = (preds - labels) ** 2
    return _apply_mask(loss, mask)


def masked_rmse(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 RMSE。"""
    return torch.sqrt(masked_mse(preds, labels, null_val))


def get_loss_function(loss_name: str, missing_value: float) -> Callable[[Any, Any], Any]:
    """根据配置返回训练损失函数。

    支持 `mse`、`mae`、`masked_mse`、`masked_mae`、`masked_rmse`。
    """
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "masked_mse":
        return lambda preds, labels: masked_mse(preds, labels, missing_value)
    if loss_name == "masked_mae":
        return lambda preds, labels: masked_mae(preds, labels, missing_value)
    if loss_name == "masked_rmse":
        return lambda preds, labels: masked_rmse(preds, labels, missing_value)
    raise ValueError(f"不支持的损失函数: {loss_name}")


def _build_mask(labels: Any, null_val: float) -> Any:
    """根据缺失值构造有效位置 mask。"""
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask_mean = mask.mean()
    if mask_mean > 0:
        mask = mask / mask_mean
    return torch.nan_to_num(mask)


def _apply_mask(loss: Any, mask: Any) -> Any:
    """应用 mask 并返回平均 loss。"""
    loss = loss * mask
    loss = torch.nan_to_num(loss)
    return loss.mean()
