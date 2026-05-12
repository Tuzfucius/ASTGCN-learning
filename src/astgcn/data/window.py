import pandas as pd
import numpy as np

def get_segment_data(
    data : np.ndarray,
    start : int,
    end : int
):
    """
    获取数据的一个片段
    :param data: 原始数据，形状为 (num_samples, num_nodes, num_features)
    :param start: 片段的起始位置
    :param end: 片段的结束位置
    :return: 数据片段，形状为 (end - start, num_nodes, num_features)
    """
    if start < 0:
        raise ValueError("起始位置必须是非负整数")
    if end is None:
        end = data.shape[0]
    if end > data.shape[0]:
        raise ValueError("结束位置超出了数据范围")
    
    return data[start:end]

def get_recent_data(
    data : np.ndarray,
    t0 : int, 
    num_recent : int
):
    """
    获取最近的 num_recent 个时间步的数据
    :param data: 原始数据，形状为 (num_samples, num_nodes, num_features)
    :param num_recent: 最近的时间步数
    :param t0: 当前时间步的位置
    :return: 最近的数据，形状为 (num_recent, num_nodes, num_features)
    """
    if t0 < 0:
        raise ValueError("当前时间步的位置必须是非负整数")
    if num_recent <= 0:
        raise ValueError("最近的时间步数必须是正整数")
    if t0 - num_recent + 1 < 0:
        raise ValueError("最近的时间步数超出了数据范围")
    if t0 >= data.shape[0]:
        raise ValueError("当前时间步的位置超出了数据范围")
    return get_segment_data(data, t0 - num_recent + 1, t0 + 1)

def get_days_data(
    data: np.ndarray,
    t0: int,
    num_days: int,
    pred_len: int,
    points_per_day: int = 288,
):
    """
    获取 num_days 天的当前时间的数据，数据集 5min 采样一次，1 天有 288 个时间步
    :param data: 原始数据，形状为 (num_samples, num_nodes, num_features)
    :param t0: 当前时间步的位置
    :param num_days: 天数
    :param pred_len: 预测长度
    :param points_per_day: 每天的时间步数
    :return: 天的数据，形状为 (num_days * pred_len, num_nodes, num_features)
    """
    if t0 < 0:
        raise ValueError("当前时间步的位置必须是非负整数")
    if t0 >= data.shape[0]:
        raise ValueError("当前时间步的位置超出了数据范围")
    if num_days <= 0:
        raise ValueError("天数必须是正整数")
    if t0 + 1 - num_days * points_per_day < 0:
        raise ValueError("天数超出了数据范围")
    
    segments = []

    for i in range(num_days, 0, -1):
        start = t0 + 1 - i * points_per_day
        end = start + pred_len

        if start < 0 or end > data.shape[0]:
            raise ValueError("天周期数据超出范围")

        segments.append(data[start:end])  # [pred_len, N, F]

    return np.concatenate(segments, axis=0)  # [num_days * pred_len, N, F]

def get_weeks_data(
    data: np.ndarray,
    t0: int,
    num_weeks: int,
    pred_len: int,
    points_per_day: int = 288,
):
    """
    获取 num_weeks 周的当前时间的数据，数据集 5min 采样一次，1 周有 2016 个时间步
    :param data: 原始数据，形状为 (num_samples, num_nodes, num_features)
    :param t0: 当前时间步的位置
    :param num_weeks: 周数
    :return: 周的数据，形状为 (num_weeks, num_nodes, num_features)
    """
    if t0 < 0:
        raise ValueError("当前时间步的位置必须是非负整数")
    if t0 >= data.shape[0]:
        raise ValueError("当前时间步的位置超出了数据范围")
    if num_weeks <= 0:
        raise ValueError("周数必须是正整数")
    if t0 + 1 - 7 * num_weeks * points_per_day < 0:
        raise ValueError("周数超出了数据范围")

    segments = []

    for i in range(num_weeks, 0, -1):
        start = t0 + 1 - 7 * i * points_per_day
        end = start + pred_len

        if start < 0 or end > data.shape[0]:
            raise ValueError("周周期数据超出范围")

        segments.append(data[start:end])  # [pred_len, N, F]

    return np.concatenate(segments, axis=0)  # [num_weeks * pred_len, N, F]

def get_target_data(
    data: np.ndarray,
    t0: int,
    pred_len: int,
    target_dim: int = 0,
):
    """
    获取预测目标。

    :param data: 原始数据，形状为 (num_samples, num_nodes, num_features)
    :param t0: 当前时间步
    :param pred_len: 预测长度
    :param target_dim: 预测的特征维度，通常 flow 是 0
    :return: 形状为 (num_nodes, pred_len)
    """
    start = t0 + 1
    end = t0 + pred_len + 1

    if start < 0:
        raise ValueError("目标起始位置非法")
    if end > data.shape[0]:
        raise ValueError("目标数据超出范围")

    y = data[start:end, :, target_dim]  # [pred_len, N]
    return y.T                          # [N, pred_len]