"""数据预处理模块。

本文件负责从原始 PEMS04 数据中生成监督学习样本。
核心目标是得到:

- x: (B, N, F, T)
- y: (B, N, T_pred)

含义： B时间切片数量，N传感器数量，F输入特征数量，T历史时间步数量，T_pred预测时间步数量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def search_data(
    sequence_length: int, # 全部历史数据的长度
    num_of_depend: int, # 需要回溯的依赖片段数量
    label_start_idx: int, # 预测目标的起始时间索引
    num_for_predict: int, # 每个样本需要预测的时间点数量
    units: int, # 依赖跨度单位，周为 7 * 24，日为 24，最近小时为 1
    points_per_hour: int, # 每小时包含的数据点数量，由数据集采样频率决定。
) -> list[tuple[int, int]]: # 返回：按时间先后排列的历史片段索引范围列表
    """查找历史依赖窗口。

    TODO:
    - 参考官方 `prepareData.py` 中的 `search_data`。
    - 根据 `label_start_idx` 向前查找周、天或小时依赖。
    - 返回历史窗口的起止索引列表。
    """
    if points_per_hour <= 0:
        raise ValueError("points_per_hour 必须为正整数。")
    if label_start_idx + num_for_predict > sequence_length:
        return None  # 无法生成样本，越界了
    
    x_idx = [] # 存储历史窗口的索引范围
    for i in range(1, num_of_depend + 1): # 依次查找依赖窗口，因为0是当前窗口，所以从1开始
        start_idx = label_start_idx - points_per_hour * units * i # 计算历史窗口的起始索引
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    if len(x_idx) != num_of_depend:
        return None  # 没有找到足够的历史窗口
    
    return x_idx[::-1]  # 先去找最近的依赖，再去找远的依赖，为了保证时间顺序正确，最后需要反转列表
    


def get_sample_indices(
    data_sequence: Any, # 原始数据序列np.ndarray，形状为 (时间切片数量, 传感器数量, 特征数量)
    num_of_weeks: int, # 依赖切片数量
    num_of_days: int, # 依赖切片数量
    num_of_hours: int, # 依赖切片数量
    label_start_idx: int, # 预测目标的起始时间索引
    num_for_predict: int, # 每个样本需要预测的时间点数量
    points_per_hour: int = 12, # 每小时包含的数据点数量，由数据集采样频率决定。
) -> tuple[Any, Any, Any, Any]:
    """
    week_sample: np.ndarray
                 形状为 (num_of_weeks * points_per_hour,
                        num_of_vertices, num_of_features)。
    day_sample: np.ndarray
                形状为 (num_of_days * points_per_hour,
                       num_of_vertices, num_of_features)。
    hour_sample: np.ndarray
                 形状为 (num_of_hours * points_per_hour,
                        num_of_vertices, num_of_features)。
    target: np.ndarray
            形状为 (num_for_predict, num_of_vertices, num_of_features)。
    """
    """生成单个监督样本。

    TODO:
    - 调用 `search_data` 获取 week/day/hour 片段。
    - 第一阶段只启用 hour 片段。
    - 目标 `target` 应取 `label_start_idx : label_start_idx + num_for_predict`。
    """
    week_sample, day_sample, hour_sample = None, None, None
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None
        # 越界了，无法生成样本，直接返回 None
    if num_of_hours > 0:
        hour_indices = search_data(
            sequence_length=data_sequence.shape[0],
            num_of_depend=num_of_hours,
            label_start_idx=label_start_idx,
            num_for_predict=num_for_predict,
            units=1,  # 小时依赖
            points_per_hour=points_per_hour,
        )
        if hour_indices is not None:
            hour_sample = np.concatenate(
                [data_sequence[start:end] for start, end in hour_indices], axis=0
            )
        else:
            return None, None, None, None
    if num_of_days > 0:
        day_indices = search_data(
            sequence_length=data_sequence.shape[0],
            num_of_depend=num_of_days,
            label_start_idx=label_start_idx,
            num_for_predict=num_for_predict,
            units=24,  # 天依赖
            points_per_hour=points_per_hour,
        )
        if day_indices is not None:
            day_sample = np.concatenate(
                [data_sequence[start:end] for start, end in day_indices], axis=0
            )
        else:
            return None, None, None, None
    if num_of_weeks > 0:
        week_indices = search_data(
            sequence_length=data_sequence.shape[0],
            num_of_depend=num_of_weeks,
            label_start_idx=label_start_idx,
            num_for_predict=num_for_predict,
            units=7 * 24,  # 周依赖
            points_per_hour=points_per_hour,
        )
        if week_indices is not None:
            week_sample = np.concatenate(
                [data_sequence[start:end] for start, end in week_indices], axis=0
            )
        else:
            return None, None, None, None  # 没有找到足够的周依赖，无法生成样本，直接返回 None
    target = data_sequence[label_start_idx : label_start_idx + num_for_predict]
    return week_sample, day_sample, hour_sample, target
    # raise NotImplementedError("TODO: 实现单样本切片逻辑。")


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
