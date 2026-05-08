"""数据加载模块。

本文件负责把预处理后的 `.npz` 文件转换为 PyTorch DataLoader。
不要在这里重新切片原始时间序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_processed_dataset(processed_path: str | Path) -> dict[str, Any]:
    """读取预处理后的 `.npz` 文件。

    返回值保留 `.npz` 中的字段名，便于和预处理模块对齐。
    """
    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"预处理数据文件不存在: {processed_path}")

    file_data = np.load(processed_path)
    required_keys = (
        "train_x",
        "train_target",
        "val_x",
        "val_target",
        "test_x",
        "test_target",
        "mean",
        "std",
    )
    missing_keys = [key for key in required_keys if key not in file_data]
    if missing_keys:
        raise KeyError(f"预处理数据缺少字段: {missing_keys}")

    dataset = {key: file_data[key] for key in file_data.files}
    for key in ("train_x", "val_x", "test_x"):
        if dataset[key].ndim != 4:
            raise ValueError(f"{key} 应为 (B, N, F, T)，实际形状: {dataset[key].shape}")

    for key in ("train_target", "val_target", "test_target"):
        if dataset[key].ndim != 3:
            raise ValueError(f"{key} 应为 (B, N, T_pred)，实际形状: {dataset[key].shape}")

    return dataset


def build_dataloader(x: Any, y: Any, batch_size: int, shuffle: bool) -> Any:
    """构造单个 DataLoader。

    这里不把张量提前移动到 GPU，训练循环负责 device 管理。
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("构造 DataLoader 需要先安装 PyTorch。") from exc

    x_tensor = torch.from_numpy(np.asarray(x)).float()
    y_tensor = torch.from_numpy(np.asarray(y)).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_all_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置构造 train/val/test DataLoader。

    同时返回原始数组形式的 target、mean、std，供评估阶段使用。
    """
    dataset = load_processed_dataset(config["data"]["processed_dataset_filename"])
    batch_size = int(config["training"]["batch_size"])

    return {
        "train": build_dataloader(dataset["train_x"], dataset["train_target"], batch_size, shuffle=True),
        "val": build_dataloader(dataset["val_x"], dataset["val_target"], batch_size, shuffle=False),
        "test": build_dataloader(dataset["test_x"], dataset["test_target"], batch_size, shuffle=False),
        "train_target": dataset["train_target"],
        "val_target": dataset["val_target"],
        "test_target": dataset["test_target"],
        "mean": dataset["mean"],
        "std": dataset["std"],
    }
