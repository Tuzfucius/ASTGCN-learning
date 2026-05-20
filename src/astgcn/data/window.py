"""ASTGCN 时间窗口构造函数。

本模块只处理 NumPy 数组，输入原始数据形状固定为 ``[T, N, F]``。
``T`` 表示时间步，``N`` 表示交通传感器节点数，``F`` 表示特征数。
"""

from __future__ import annotations

import numpy as np


def get_segment_data(data: np.ndarray, start: int, end: int) -> np.ndarray:
    """获取连续时间片段。

    :param data: 原始数据，形状为 ``[T, N, F]``。
    :param start: 起始时间步，包含该位置。
    :param end: 结束时间步，不包含该位置。
    :return: 连续片段，形状为 ``[end - start, N, F]``。
    """
    if data.ndim != 3:
        raise ValueError(f"data 必须是 [T, N, F] 三维数组，当前 shape={data.shape}")
    if start < 0:
        raise ValueError("start 必须是非负整数")
    if end <= start:
        raise ValueError("end 必须大于 start")
    if end > data.shape[0]:
        raise ValueError("片段结束位置超出数据范围")
    return data[start:end]


def get_recent_data(data: np.ndarray, t0: int, num_recent: int) -> np.ndarray:
    """获取当前时间点之前的近期片段。

    :param data: 原始数据，形状为 ``[T, N, F]``。
    :param t0: 当前时间步位置，预测目标从 ``t0 + 1`` 开始。
    :param num_recent: 近期片段长度。
    :return: 近期片段，形状为 ``[num_recent, N, F]``。
    """
    if num_recent <= 0:
        raise ValueError("num_recent 必须是正整数")
    return get_segment_data(data, t0 - num_recent + 1, t0 + 1)


def get_days_data(
    data: np.ndarray,
    t0: int,
    num_days: int,
    pred_len: int,
    points_per_day: int = 288,
) -> np.ndarray:
    """获取过去若干天相同时间段的数据。

    数据集 5 分钟采样一次时，1 天有 288 个时间步。每一天取与未来预测窗口
    长度相同的 ``pred_len`` 个时间步。

    :param data: 原始数据，形状为 ``[T, N, F]``。
    :param t0: 当前时间步位置。
    :param num_days: 使用过去多少天。
    :param pred_len: 预测长度。
    :param points_per_day: 每天的时间步数。
    :return: 日周期片段，形状为 ``[num_days * pred_len, N, F]``。
    """
    if num_days <= 0:
        raise ValueError("num_days 必须是正整数")
    if pred_len <= 0:
        raise ValueError("pred_len 必须是正整数")

    segments = []
    for day in range(num_days, 0, -1):
        start = t0 + 1 - day * points_per_day
        end = start + pred_len
        segments.append(get_segment_data(data, start, end))
    return np.concatenate(segments, axis=0)


def get_weeks_data(
    data: np.ndarray,
    t0: int,
    num_weeks: int,
    pred_len: int,
    points_per_day: int = 288,
) -> np.ndarray:
    """获取过去若干周相同星期和时间段的数据。

    :param data: 原始数据，形状为 ``[T, N, F]``。
    :param t0: 当前时间步位置。
    :param num_weeks: 使用过去多少周。
    :param pred_len: 预测长度。
    :param points_per_day: 每天的时间步数。
    :return: 周周期片段，形状为 ``[num_weeks * pred_len, N, F]``。
    """
    if num_weeks <= 0:
        raise ValueError("num_weeks 必须是正整数")
    if pred_len <= 0:
        raise ValueError("pred_len 必须是正整数")

    segments = []
    for week in range(num_weeks, 0, -1):
        start = t0 + 1 - 7 * week * points_per_day
        end = start + pred_len
        segments.append(get_segment_data(data, start, end))
    return np.concatenate(segments, axis=0)


def get_target_data(
    data: np.ndarray,
    t0: int,
    pred_len: int,
    target_dim: int = 0,
) -> np.ndarray:
    """获取未来预测目标。

    :param data: 原始数据，形状为 ``[T, N, F]``。
    :param t0: 当前时间步位置。
    :param pred_len: 预测长度。
    :param target_dim: 目标特征编号，PEMS04 中默认第 0 个特征为流量。
    :return: 预测目标，形状为 ``[N, pred_len]``。
    """
    if pred_len <= 0:
        raise ValueError("pred_len 必须是正整数")
    if target_dim < 0 or target_dim >= data.shape[2]:
        raise ValueError("target_dim 超出特征维度范围")

    target = get_segment_data(data, t0 + 1, t0 + pred_len + 1)
    return target[:, :, target_dim].T.astype(np.float32)


def build_t0_list(
    num_samples: int,
    num_recent: int,
    num_days: int,
    num_weeks: int,
    pred_len: int,
    points_per_day: int = 288,
) -> np.ndarray:
    """构造所有合法的当前时间步 ``t0``。

    :param num_samples: 总时间步数 ``T``。
    :param num_recent: recent 片段长度。
    :param num_days: day 分支使用的天数。
    :param num_weeks: week 分支使用的周数。
    :param pred_len: 预测长度。
    :param points_per_day: 每天时间步数。
    :return: 一维 ``t0`` 数组。
    """
    min_t0 = max(
        num_recent - 1,
        num_days * points_per_day - 1,
        num_weeks * 7 * points_per_day - 1,
    )
    max_t0 = num_samples - pred_len - 1
    if min_t0 > max_t0:
        raise ValueError("数据长度不足，无法构造 ASTGCN 样本")
    return np.arange(min_t0, max_t0 + 1, dtype=np.int64)


def split_t0_list(
    t0_list: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序划分训练、验证和测试 ``t0``。

    :param t0_list: 合法 ``t0`` 一维数组。
    :param train_ratio: 训练集比例。
    :param val_ratio: 验证集比例。
    :return: ``train_t0, val_t0, test_t0``。
    """
    if t0_list.ndim != 1:
        raise ValueError("t0_list 必须是一维数组")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    train_end = int(len(t0_list) * train_ratio)
    val_end = int(len(t0_list) * (train_ratio + val_ratio))
    return t0_list[:train_end], t0_list[train_end:val_end], t0_list[val_end:]
