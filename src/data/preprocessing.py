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

    TODO:
    - 读取 `graph_signal_matrix_filename`。
    - 遍历时间索引，生成全部样本。
    - 按 60/20/20 划分 train/val/test。
    - 只保留第 `target_channel` 个特征作为预测目标。
    """
    graph_signal_matrix_filename = config["graph_signal_matrix_filename"]
    num_of_weeks = config["num_of_weeks"]
    num_of_days = config["num_of_days"]
    num_of_hours = config["num_of_hours"]
    num_for_predict = config["num_for_predict"]
    data_seq = np.load(graph_signal_matrix_filename)["data"]  # 形状 (时间切片数量, 传感器数量, 特征数量)
    all_samples = []
    
    # raise NotImplementedError("TODO: 实现完整数据集生成逻辑。")


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
