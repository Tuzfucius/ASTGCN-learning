"""测试评估指标。"""

from __future__ import annotations

from typing import Any

import numpy as np


def mae(y_true: Any, y_pred: Any) -> float:
    """平均绝对误差。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def mse(y_true: Any, y_pred: Any) -> float:
    """均方误差。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    """均方根误差。"""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Any, y_pred: Any, missing_value: float | None = None) -> float:
    """平均绝对百分比误差。

    返回值沿用官方实现，不额外乘以 100。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.abs(y_true) > 1e-5
        if missing_value is not None:
            mask = mask & _valid_mask(y_true, missing_value)
        percentage_error = np.abs((y_pred - y_true) / y_true)
        return _masked_mean(percentage_error, mask)


def evaluate_prediction(
    y_true: Any,
    y_pred: Any,
    metric_method: str,
    missing_value: float,
) -> dict[str, float]:
    """汇总测试指标。

    `metric_method` 为 `mask` 时忽略缺失值位置，否则只在 MAPE 中规避除零。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"预测和标签形状不一致: y_pred={y_pred.shape}, y_true={y_true.shape}")

    if metric_method == "mask":
        mask = _valid_mask(y_true, missing_value)
        return {
            "MAE": _masked_mean(np.abs(y_pred - y_true), mask),
            "RMSE": float(np.sqrt(_masked_mean((y_pred - y_true) ** 2, mask))),
            "MAPE": mape(y_true, y_pred, missing_value),
        }

    if metric_method == "unmask":
        return {
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        }

    raise ValueError(f"不支持的评估方式: {metric_method}")


def _valid_mask(y_true: Any, missing_value: float) -> Any:
    """根据缺失值生成有效位置 mask。"""
    if np.isnan(missing_value):
        return ~np.isnan(y_true)
    return y_true != missing_value


def _masked_mean(values: Any, mask: Any) -> float:
    """计算 mask 后的均值，并保持与官方实现一致的尺度。"""
    mask = mask.astype(np.float32)
    mask_mean = mask.mean()
    if mask_mean == 0:
        return 0.0
    mask = mask / mask_mean
    values = np.nan_to_num(values * mask)
    return float(np.mean(values))
