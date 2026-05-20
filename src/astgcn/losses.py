"""损失函数选择器。"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from astgcn.metrics import masked_mae, masked_mape, masked_rmse


class MaskedLoss(nn.Module):
    """将 masked 指标包装为 PyTorch 损失模块。

    参数:
        metric_fn: 接收 ``pred``、``target``、``mask_value`` 并返回 Tensor 的指标函数。
        mask_value: 需要忽略的目标值。

    返回:
        前向调用返回一个可反向传播的标量 Tensor。
    """

    def __init__(self, metric_fn: Callable[..., torch.Tensor], mask_value: float = 0.0):
        super().__init__()
        self.metric_fn = metric_fn
        self.mask_value = mask_value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 masked 损失。

        参数:
            pred: 模型预测，形状通常为 ``[B, N, T]``。
            target: 真实值，形状与 ``pred`` 一致。

        返回:
            标量损失 Tensor。
        """
        return self.metric_fn(pred, target, self.mask_value)


def get_loss(name: str, mask_value: float = 0.0) -> nn.Module:
    """按名称创建损失函数。

    参数:
        name: 损失名称，支持 ``mae``、``mse``、``rmse``、``huber``、``smooth_l1``、
            ``masked_mae``、``masked_rmse``、``masked_mape``。
        mask_value: masked 损失忽略的目标值。

    返回:
        PyTorch ``nn.Module`` 损失对象。
    """
    key = name.lower()
    if key in {"mae", "l1"}:
        return nn.L1Loss()
    if key in {"mse", "l2"}:
        return nn.MSELoss()
    if key in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss()
    if key == "rmse":
        return RMSELoss()
    if key == "masked_mae":
        return MaskedLoss(masked_mae, mask_value=mask_value)
    if key == "masked_rmse":
        return MaskedLoss(masked_rmse, mask_value=mask_value)
    if key == "masked_mape":
        return MaskedLoss(masked_mape, mask_value=mask_value)
    raise ValueError(f"未知损失函数: {name}")


class RMSELoss(nn.Module):
    """均方根误差损失。

    输入:
        pred: 模型预测，形状通常为 ``[B, N, T]``。
        target: 真实值，形状与 ``pred`` 一致。

    输出:
        标量 RMSE Tensor。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 RMSE 损失。

        参数:
            pred: 模型预测，形状通常为 ``[B, N, T]``。
            target: 真实值，形状与 ``pred`` 一致。

        返回:
            标量 RMSE Tensor。
        """
        return torch.sqrt(torch.mean((pred - target) ** 2) + self.eps)
