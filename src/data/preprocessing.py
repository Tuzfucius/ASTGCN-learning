"""数据预处理模块。

本文件负责从原始 PEMS04 数据中生成监督学习样本。
核心目标是得到:

- x: (B, N, F, T)
- y: (B, N, T_pred)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def search_data(
    sequence_length: int,
    num_of_depend: int,
    label_start_idx: int,
    num_for_predict: int,
    units: int,
    points_per_hour: int,
) -> list[tuple[int, int]]:
    """查找历史依赖窗口。

    TODO:
    - 参考官方 `prepareData.py` 中的 `search_data`。
    - 根据 `label_start_idx` 向前查找周、天或小时依赖。
    - 返回历史窗口的起止索引列表。
    """
    raise NotImplementedError("TODO: 实现历史窗口索引搜索逻辑。")


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

    TODO:
    - 调用 `search_data` 获取 week/day/hour 片段。
    - 第一阶段只启用 hour 片段。
    - 目标 `target` 应取 `label_start_idx : label_start_idx + num_for_predict`。
    """
    raise NotImplementedError("TODO: 实现单样本切片逻辑。")


def generate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置生成 train/val/test 数据。

    TODO:
    - 读取 `graph_signal_matrix_filename`。
    - 遍历时间索引，生成全部样本。
    - 按 60/20/20 划分 train/val/test。
    - 只保留第 `target_channel` 个特征作为预测目标。
    """
    raise NotImplementedError("TODO: 实现完整数据集生成逻辑。")


def standardize(train_x: Any, val_x: Any, test_x: Any) -> tuple[dict[str, Any], Any, Any, Any]:
    """使用训练集统计量标准化输入。

    TODO:
    - mean 只从 train_x 计算。
    - std 只从 train_x 计算。
    - val_x/test_x 必须复用 train_x 的 mean/std。
    """
    raise NotImplementedError("TODO: 实现标准化逻辑。")


def save_dataset(dataset: dict[str, Any], output_path: str | Path) -> None:
    """保存预处理后的数据集。

    TODO:
    - 使用 `np.savez_compressed` 保存。
    - 保存 train_x、train_target、val_x、val_target、test_x、test_target、mean、std。
    """
    raise NotImplementedError("TODO: 实现预处理数据保存逻辑。")
