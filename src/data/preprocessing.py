"""数据预处理模块。

本文件负责从原始 PEMS04 数据中生成监督学习样本。
核心目标是得到:

- x: (B, N, F, T)
- y: (B, N, T_pred)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def search_data(
    sequence_length: int,
    num_of_depend: int,
    label_start_idx: int,
    num_for_predict: int,
    units: int,
    points_per_hour: int,
) -> list[tuple[int, int]]:
    """查找历史依赖窗口。

    返回值中的每个元素是左闭右开的时间索引区间。
    """
    if points_per_hour <= 0:
        raise ValueError("points_per_hour 必须大于 0。")

    if num_of_depend <= 0:
        return []

    if label_start_idx + num_for_predict > sequence_length:
        return []

    indices = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx < 0:
            return []
        indices.append((start_idx, end_idx))

    return indices[::-1]


def get_sample_indices(
    data_sequence: Any,
    num_of_weeks: int,
    num_of_days: int,
    num_of_hours: int,
    label_start_idx: int,
    num_for_predict: int,
    points_per_hour: int,
) -> tuple[Any, Any, Any, Any]:
    """生成单个监督样本。

    输入序列形状应为 `(T, N, F)`。
    """
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(
            data_sequence.shape[0],
            num_of_weeks,
            label_start_idx,
            num_for_predict,
            7 * 24,
            points_per_hour,
        )
        if not week_indices:
            return None, None, None, None
        week_sample = np.concatenate([data_sequence[start:end] for start, end in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(
            data_sequence.shape[0],
            num_of_days,
            label_start_idx,
            num_for_predict,
            24,
            points_per_hour,
        )
        if not day_indices:
            return None, None, None, None
        day_sample = np.concatenate([data_sequence[start:end] for start, end in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(
            data_sequence.shape[0],
            num_of_hours,
            label_start_idx,
            num_for_predict,
            1,
            points_per_hour,
        )
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[start:end] for start, end in hour_indices], axis=0)

    target = data_sequence[label_start_idx : label_start_idx + num_for_predict]
    return week_sample, day_sample, hour_sample, target


def generate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置生成 train/val/test 数据。

    第一阶段只保留 `target_channel` 一个特征作为输入和预测目标。
    """
    data_config = config["data"]
    task_config = config["task"]

    data_path = Path(data_config["graph_signal_matrix_filename"])
    if not data_path.exists():
        raise FileNotFoundError(f"原始数据文件不存在: {data_path}")

    raw_file = np.load(data_path)
    if "data" not in raw_file:
        raise KeyError(f"{data_path} 中缺少 data 数组。")

    data_sequence = raw_file["data"]
    if data_sequence.ndim != 3:
        raise ValueError(f"原始数据应为 (T, N, F)，实际形状: {data_sequence.shape}")

    num_of_vertices = int(data_config["num_of_vertices"])
    if data_sequence.shape[1] != num_of_vertices:
        raise ValueError(f"节点数不匹配: 配置为 {num_of_vertices}，数据为 {data_sequence.shape[1]}")

    target_channel = int(task_config["target_channel"])
    if target_channel < 0 or target_channel >= data_sequence.shape[2]:
        raise ValueError(f"target_channel 超出范围: {target_channel}")

    data_sequence = data_sequence[:, :, target_channel : target_channel + 1]
    all_samples = []

    for label_start_idx in range(data_sequence.shape[0]):
        week_sample, day_sample, hour_sample, target = get_sample_indices(
            data_sequence=data_sequence,
            num_of_weeks=int(task_config["num_of_weeks"]),
            num_of_days=int(task_config["num_of_days"]),
            num_of_hours=int(task_config["num_of_hours"]),
            label_start_idx=label_start_idx,
            num_for_predict=int(task_config["num_for_predict"]),
            points_per_hour=int(data_config["points_per_hour"]),
        )
        if week_sample is None and day_sample is None and hour_sample is None:
            continue

        sample_parts = []
        for history_sample in (week_sample, day_sample, hour_sample):
            if history_sample is not None:
                sample_parts.append(np.expand_dims(history_sample, axis=0).transpose((0, 2, 3, 1)))

        x = np.concatenate(sample_parts, axis=-1)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        timestamp = np.array([[label_start_idx]], dtype=np.int64)
        all_samples.append((x, target, timestamp))

    if not all_samples:
        raise ValueError("没有生成任何样本，请检查时间窗口配置。")

    x_all, target_all, timestamp_all = [np.concatenate(items, axis=0) for items in zip(*all_samples)]

    expected_len_input = int(task_config["len_input"])
    if x_all.shape[-1] != expected_len_input:
        raise ValueError(f"输入时间步不匹配: 配置为 {expected_len_input}，实际为 {x_all.shape[-1]}")

    split_line1 = int(x_all.shape[0] * 0.6)
    split_line2 = int(x_all.shape[0] * 0.8)
    if split_line1 == 0 or split_line2 == split_line1 or split_line2 == x_all.shape[0]:
        raise ValueError(f"样本数量过少，无法按 60/20/20 划分: {x_all.shape[0]}")

    train_x = x_all[:split_line1]
    val_x = x_all[split_line1:split_line2]
    test_x = x_all[split_line2:]

    stats, train_x, val_x, test_x = standardize(train_x, val_x, test_x)

    return {
        "train_x": train_x,
        "train_target": target_all[:split_line1],
        "train_timestamp": timestamp_all[:split_line1],
        "val_x": val_x,
        "val_target": target_all[split_line1:split_line2],
        "val_timestamp": timestamp_all[split_line1:split_line2],
        "test_x": test_x,
        "test_target": target_all[split_line2:],
        "test_timestamp": timestamp_all[split_line2:],
        "mean": stats["mean"],
        "std": stats["std"],
    }


def standardize(train_x: Any, val_x: Any, test_x: Any) -> tuple[dict[str, Any], Any, Any, Any]:
    """使用训练集统计量标准化输入。

    mean/std 的形状为 `(1, 1, F, 1)`，可广播到 `(B, N, F, T)`。
    """
    if train_x.ndim != 4 or val_x.ndim != 4 or test_x.ndim != 4:
        raise ValueError("train_x、val_x、test_x 都应为 (B, N, F, T)。")

    if train_x.shape[1:] != val_x.shape[1:] or train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError("train/val/test 的 N、F、T 维度必须一致。")

    mean = train_x.mean(axis=(0, 1, 3), keepdims=True)
    std = train_x.std(axis=(0, 1, 3), keepdims=True)
    std = np.where(std == 0, 1.0, std)

    return (
        {"mean": mean, "std": std},
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
    )


def save_dataset(dataset: dict[str, Any], output_path: str | Path) -> None:
    """保存预处理后的数据集。

    保存字段与 `docs/数据说明.md` 保持一致。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_x=dataset["train_x"],
        train_target=dataset["train_target"],
        train_timestamp=dataset["train_timestamp"],
        val_x=dataset["val_x"],
        val_target=dataset["val_target"],
        val_timestamp=dataset["val_timestamp"],
        test_x=dataset["test_x"],
        test_target=dataset["test_target"],
        test_timestamp=dataset["test_timestamp"],
        mean=dataset["mean"],
        std=dataset["std"],
    )
