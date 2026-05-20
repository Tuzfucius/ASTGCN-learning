"""模型 checkpoint 保存与加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def _scaler_state(scaler: Any | None) -> Dict[str, Any]:
    if scaler is None:
        return {"scaler_mean": None, "scaler_std": None}
    return {
        "scaler_mean": getattr(scaler, "mean", None),
        "scaler_std": getattr(scaler, "std", None),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    config: Dict[str, Any] | None = None,
    scaler: Any | None = None,
    **extra: Any,
) -> Path:
    """保存训练 checkpoint。

    参数:
        path: 保存路径。
        model: 需要保存参数的模型。
        optimizer: 优化器；为 ``None`` 时不保存优化器状态。
        epoch: 当前 epoch 编号。
        best_metric: 当前最优验证指标。
        config: 训练配置字典。
        scaler: 标准化器，读取其中的 ``mean`` 和 ``std``。
        extra: 额外需要写入 checkpoint 的键值。

    返回:
        实际保存的 checkpoint 路径。
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "config": config,
    }
    state.update(_scaler_state(scaler))
    state.update(extra)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """加载训练 checkpoint。

    参数:
        path: checkpoint 文件路径。
        model: 需要载入参数的模型；为 ``None`` 时只返回 checkpoint 字典。
        optimizer: 需要载入状态的优化器；checkpoint 中没有优化器状态时跳过。
        map_location: ``torch.load`` 的设备映射参数。
        strict: 传给 ``model.load_state_dict`` 的严格匹配开关。

    返回:
        checkpoint 原始字典，包含模型参数、优化器参数、epoch、best_metric、config、scaler_mean、scaler_std。
    """
    checkpoint = torch.load(Path(path), map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
