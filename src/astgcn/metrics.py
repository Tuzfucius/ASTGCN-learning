"""训练与评估指标。"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Union

import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray]


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def mae(pred: ArrayLike, target: ArrayLike) -> ArrayLike:
    """计算平均绝对误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]`` 或可广播形状。
        target: 真实值，形状与 ``pred`` 一致或可广播。

    返回:
        标量 MAE；输入为 Tensor 时返回 Tensor，输入为 ndarray 时返回 float。
    """
    if _is_torch(pred) or _is_torch(target):
        return torch.mean(torch.abs(_to_tensor(pred) - _to_tensor(target)))
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def rmse(pred: ArrayLike, target: ArrayLike) -> ArrayLike:
    """计算均方根误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]`` 或可广播形状。
        target: 真实值，形状与 ``pred`` 一致或可广播。

    返回:
        标量 RMSE；输入为 Tensor 时返回 Tensor，输入为 ndarray 时返回 float。
    """
    if _is_torch(pred) or _is_torch(target):
        diff = _to_tensor(pred) - _to_tensor(target)
        return torch.sqrt(torch.mean(diff * diff))
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.sqrt(np.mean(diff * diff)))


def mape(pred: ArrayLike, target: ArrayLike, eps: float = 1e-5) -> ArrayLike:
    """计算平均绝对百分比误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]`` 或可广播形状。
        target: 真实值，形状与 ``pred`` 一致或可广播。
        eps: 分母下限，用于避免目标值接近 0 时除零。

    返回:
        标量 MAPE，单位为百分比；输入为 Tensor 时返回 Tensor，输入为 ndarray 时返回 float。
    """
    if _is_torch(pred) or _is_torch(target):
        pred_t = _to_tensor(pred)
        target_t = _to_tensor(target)
        denom = torch.clamp(torch.abs(target_t), min=eps)
        return torch.mean(torch.abs((pred_t - target_t) / denom)) * 100.0
    pred_a = np.asarray(pred)
    target_a = np.asarray(target)
    denom = np.maximum(np.abs(target_a), eps)
    return float(np.mean(np.abs((pred_a - target_a) / denom)) * 100.0)


def masked_mae(pred: ArrayLike, target: ArrayLike, mask_value: float = 0.0) -> ArrayLike:
    """按掩码计算平均绝对误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]``。
        target: 真实值，形状与 ``pred`` 一致。
        mask_value: 需要忽略的目标值，默认忽略 0。

    返回:
        标量 masked MAE；若没有有效元素，返回 0。
    """
    if _is_torch(pred) or _is_torch(target):
        pred_t = _to_tensor(pred)
        target_t = _to_tensor(target).to(device=pred_t.device, dtype=pred_t.dtype)
        mask = target_t != mask_value
        if not torch.any(mask):
            return pred_t.new_tensor(0.0)
        return torch.mean(torch.abs(pred_t[mask] - target_t[mask]))
    pred_a = np.asarray(pred)
    target_a = np.asarray(target)
    mask = target_a != mask_value
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred_a[mask] - target_a[mask])))


def masked_rmse(pred: ArrayLike, target: ArrayLike, mask_value: float = 0.0) -> ArrayLike:
    """按掩码计算均方根误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]``。
        target: 真实值，形状与 ``pred`` 一致。
        mask_value: 需要忽略的目标值，默认忽略 0。

    返回:
        标量 masked RMSE；若没有有效元素，返回 0。
    """
    if _is_torch(pred) or _is_torch(target):
        pred_t = _to_tensor(pred)
        target_t = _to_tensor(target).to(device=pred_t.device, dtype=pred_t.dtype)
        mask = target_t != mask_value
        if not torch.any(mask):
            return pred_t.new_tensor(0.0)
        diff = pred_t[mask] - target_t[mask]
        return torch.sqrt(torch.mean(diff * diff))
    pred_a = np.asarray(pred)
    target_a = np.asarray(target)
    mask = target_a != mask_value
    if not np.any(mask):
        return 0.0
    diff = pred_a[mask] - target_a[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def masked_mape(
    pred: ArrayLike,
    target: ArrayLike,
    mask_value: float = 0.0,
    eps: float = 1e-5,
) -> ArrayLike:
    """按掩码计算平均绝对百分比误差。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]``。
        target: 真实值，形状与 ``pred`` 一致。
        mask_value: 需要忽略的目标值，默认忽略 0。
        eps: 分母下限，用于避免目标值接近 0 时除零。

    返回:
        标量 masked MAPE，单位为百分比；若没有有效元素，返回 0。
    """
    if _is_torch(pred) or _is_torch(target):
        pred_t = _to_tensor(pred)
        target_t = _to_tensor(target).to(device=pred_t.device, dtype=pred_t.dtype)
        mask = target_t != mask_value
        if not torch.any(mask):
            return pred_t.new_tensor(0.0)
        denom = torch.clamp(torch.abs(target_t[mask]), min=eps)
        return torch.mean(torch.abs((pred_t[mask] - target_t[mask]) / denom)) * 100.0
    pred_a = np.asarray(pred)
    target_a = np.asarray(target)
    mask = target_a != mask_value
    if not np.any(mask):
        return 0.0
    denom = np.maximum(np.abs(target_a[mask]), eps)
    return float(np.mean(np.abs((pred_a[mask] - target_a[mask]) / denom)) * 100.0)


METRIC_REGISTRY: Mapping[str, Callable[..., ArrayLike]] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "masked_mae": masked_mae,
    "masked_rmse": masked_rmse,
    "masked_mape": masked_mape,
}


def get_metric(name: str) -> Callable[..., ArrayLike]:
    """按名称获取指标函数。

    参数:
        name: 指标名称，支持 ``mae``、``rmse``、``mape`` 及对应 ``masked_*`` 版本。

    返回:
        可调用指标函数。
    """
    key = name.lower()
    if key not in METRIC_REGISTRY:
        raise ValueError(f"未知指标: {name}")
    return METRIC_REGISTRY[key]


def compute_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    mask_value: float | None = None,
) -> Dict[str, float]:
    """一次性计算 MAE、RMSE、MAPE。

    参数:
        pred: 预测值，形状通常为 ``[B, N, T]``。
        target: 真实值，形状与 ``pred`` 一致。
        mask_value: 非 ``None`` 时按该目标值做掩码。

    返回:
        包含 ``mae``、``rmse``、``mape`` 的字典，值为 Python float。
    """
    funcs = (masked_mae, masked_rmse, masked_mape) if mask_value is not None else (mae, rmse, mape)
    values = {
        "mae": funcs[0](pred, target, mask_value) if mask_value is not None else funcs[0](pred, target),
        "rmse": funcs[1](pred, target, mask_value) if mask_value is not None else funcs[1](pred, target),
        "mape": funcs[2](pred, target, mask_value) if mask_value is not None else funcs[2](pred, target),
    }
    return {name: float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value) for name, value in values.items()}
