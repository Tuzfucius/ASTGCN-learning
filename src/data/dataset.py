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
        "train_timestamp",
        "val_x",
        "val_target",
        "val_timestamp",
        "test_x",
        "test_target",
        "test_timestamp",
        "mean",
        "std",
    )
    missing_keys = [key for key in required_keys if key not in file_data]
    if missing_keys:
        raise KeyError(f"预处理数据缺少字段: {missing_keys}")

    dataset = {key: file_data[key] for key in file_data.files}
    for split in ("train", "val", "test"):
        _validate_split_arrays(dataset, split)

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

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，实际为 {batch_size}")
    _validate_xy_arrays(np.asarray(x), np.asarray(y), split_name="DataLoader")

    x_tensor = torch.from_numpy(np.asarray(x)).float()
    y_tensor = torch.from_numpy(np.asarray(y)).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    if len(dataset) == 0:
        raise ValueError("DataLoader 数据集为空。")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_all_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置构造 train/val/test DataLoader。

    同时返回原始数组形式的 target、mean、std，供评估阶段使用。
    """
    dataset = load_processed_dataset(config["data"]["processed_dataset_filename"])
    validate_processed_dataset(dataset, config)
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


def validate_processed_dataset(dataset: dict[str, Any], config: dict[str, Any]) -> None:
    """按配置校验预处理 `.npz` 的数据契约。"""
    expected_x_shape = (
        int(config["data"]["num_of_vertices"]),
        int(config["task"]["in_channels"]),
        int(config["task"]["len_input"]),
    )
    expected_target_shape = (
        int(config["data"]["num_of_vertices"]),
        int(config["task"]["num_for_predict"]),
    )

    for split in ("train", "val", "test"):
        x_key = f"{split}_x"
        target_key = f"{split}_target"
        x_shape = dataset[x_key].shape
        target_shape = dataset[target_key].shape

        if x_shape[0] == 0:
            raise ValueError(f"{x_key} 样本数为 0，请重新检查预处理结果。")
        if x_shape[1:] != expected_x_shape:
            raise ValueError(
                f"{x_key} 形状与配置不一致: "
                f"实际 N/F/T={x_shape[1:]}, 配置 N/F/T={expected_x_shape}"
            )
        if target_shape[1:] != expected_target_shape:
            raise ValueError(
                f"{target_key} 形状与配置不一致: "
                f"实际 N/T_pred={target_shape[1:]}, 配置 N/T_pred={expected_target_shape}"
            )

    mean_shape = dataset["mean"].shape
    std_shape = dataset["std"].shape
    expected_stats_shape = (1, 1, int(config["task"]["in_channels"]), 1)
    if mean_shape != expected_stats_shape:
        raise ValueError(f"mean 应为 {expected_stats_shape}，实际形状: {mean_shape}")
    if std_shape != expected_stats_shape:
        raise ValueError(f"std 应为 {expected_stats_shape}，实际形状: {std_shape}")


def _validate_split_arrays(dataset: dict[str, Any], split: str) -> None:
    """校验单个数据划分中的基础数组形状和样本数。"""
    x_key = f"{split}_x"
    target_key = f"{split}_target"
    timestamp_key = f"{split}_timestamp"

    x = dataset[x_key]
    target = dataset[target_key]
    timestamp = dataset[timestamp_key]

    if x.ndim != 4:
        raise ValueError(f"{x_key} 应为 (B, N, F, T)，实际形状: {x.shape}")
    if target.ndim != 3:
        raise ValueError(f"{target_key} 应为 (B, N, T_pred)，实际形状: {target.shape}")
    if timestamp.ndim != 2:
        raise ValueError(f"{timestamp_key} 应为 (B, 1)，实际形状: {timestamp.shape}")
    if x.shape[0] != target.shape[0]:
        raise ValueError(f"{x_key} 与 {target_key} 样本数不一致: {x.shape[0]} != {target.shape[0]}")
    if x.shape[0] != timestamp.shape[0]:
        raise ValueError(f"{x_key} 与 {timestamp_key} 样本数不一致: {x.shape[0]} != {timestamp.shape[0]}")
    if x.shape[1] != target.shape[1]:
        raise ValueError(f"{x_key} 与 {target_key} 节点数不一致: {x.shape[1]} != {target.shape[1]}")


def _validate_xy_arrays(x: Any, y: Any, split_name: str) -> None:
    """校验准备交给 TensorDataset 的输入和目标。"""
    if x.ndim != 4:
        raise ValueError(f"{split_name} x 应为 (B, N, F, T)，实际形状: {x.shape}")
    if y.ndim != 3:
        raise ValueError(f"{split_name} y 应为 (B, N, T_pred)，实际形状: {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split_name} x 与 y 样本数不一致: {x.shape[0]} != {y.shape[0]}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"{split_name} x 与 y 节点数不一致: {x.shape[1]} != {y.shape[1]}")
