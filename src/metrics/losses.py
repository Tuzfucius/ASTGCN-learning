"""训练损失函数。"""

from __future__ import annotations

from typing import Any, Callable


def masked_mae(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 MAE。

    TODO:
    - 根据 labels 构造 mask。
    - 计算 `abs(preds - labels)`。
    - 只统计 mask 有效位置。
    """
    raise NotImplementedError("TODO: 实现 masked MAE。")


def masked_mse(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 MSE。"""
    raise NotImplementedError("TODO: 实现 masked MSE。")


def masked_rmse(preds: Any, labels: Any, null_val: float) -> Any:
    """忽略缺失值的 RMSE。"""
    raise NotImplementedError("TODO: 实现 masked RMSE。")


def get_loss_function(loss_name: str, missing_value: float) -> Callable[[Any, Any], Any]:
    """根据配置返回训练损失函数。

    TODO:
    - 支持 mse、mae、masked_mse、masked_mae。
    - 普通 mse/mae 可以使用 torch.nn 中的损失。
    """
    raise NotImplementedError("TODO: 实现损失函数选择逻辑。")
