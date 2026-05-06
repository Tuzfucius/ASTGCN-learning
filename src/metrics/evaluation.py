"""测试评估指标。"""

from __future__ import annotations

from typing import Any


def mae(y_true: Any, y_pred: Any) -> float:
    """平均绝对误差。"""
    raise NotImplementedError("TODO: 实现 MAE。")


def mse(y_true: Any, y_pred: Any) -> float:
    """均方误差。"""
    raise NotImplementedError("TODO: 实现 MSE。")


def rmse(y_true: Any, y_pred: Any) -> float:
    """均方根误差。"""
    raise NotImplementedError("TODO: 实现 RMSE。")


def mape(y_true: Any, y_pred: Any, missing_value: float | None = None) -> float:
    """平均绝对百分比误差。

    TODO:
    - 注意 y_true 为 0 时的除零问题。
    - 如果启用 mask，应忽略缺失值位置。
    """
    raise NotImplementedError("TODO: 实现 MAPE。")


def evaluate_prediction(
    y_true: Any,
    y_pred: Any,
    metric_method: str,
    missing_value: float,
) -> dict[str, float]:
    """汇总测试指标。

    TODO:
    - 根据 metric_method 决定是否 mask。
    - 输出 MAE、RMSE、MAPE。
    - 后续可以增加按 horizon 统计。
    """
    raise NotImplementedError("TODO: 实现测试指标汇总。")
