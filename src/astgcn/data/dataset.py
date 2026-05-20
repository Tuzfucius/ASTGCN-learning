"""PyTorch Dataset 封装。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from astgcn.data.window import (
    get_days_data,
    get_recent_data,
    get_target_data,
    get_weeks_data,
)


def _time_first_to_node_feature_time(segment: np.ndarray) -> np.ndarray:
    """把 ``[T, N, F]`` 片段转换为 ``[N, F, T]``。

    :param segment: 时间优先的窗口片段。
    :return: 节点、特征、时间顺序的窗口片段。
    """
    return np.transpose(segment, (1, 2, 0)).astype(np.float32)


class ASTGCNDataset(Dataset):
    """ASTGCN 三组件数据集。

    单个样本返回字典：

    - ``recent``: ``[N, F, T_h]``
    - ``daily``: ``[N, F, T_d]``
    - ``weekly``: ``[N, F, T_w]``
    - ``target``: ``[N, T_p]``
    - ``t0``: 当前时间步索引
    """

    def __init__(
        self,
        data: np.ndarray,
        t0_list: np.ndarray,
        num_recent: int,
        num_days: int,
        num_weeks: int,
        pred_len: int,
        points_per_day: int = 288,
        target_dim: int = 0,
    ) -> None:
        """初始化数据集。

        :param data: 已标准化或原始数据，形状为 ``[T, N, F]``。
        :param t0_list: 样本当前时间步列表。
        :param num_recent: recent 分支输入长度。
        :param num_days: daily 分支使用天数。
        :param num_weeks: weekly 分支使用周数。
        :param pred_len: 预测长度。
        :param points_per_day: 每天时间步数。
        :param target_dim: 目标特征编号。
        """
        if data.ndim != 3:
            raise ValueError(f"data 必须是 [T, N, F]，当前 shape={data.shape}")
        self.data = data.astype(np.float32)
        self.t0_list = np.asarray(t0_list, dtype=np.int64)
        self.num_recent = num_recent
        self.num_days = num_days
        self.num_weeks = num_weeks
        self.pred_len = pred_len
        self.points_per_day = points_per_day
        self.target_dim = target_dim

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.t0_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """按索引读取一个 ASTGCN 样本。

        :param idx: 样本编号。
        :return: 包含三类输入、目标值和 ``t0`` 的字典。
        """
        t0 = int(self.t0_list[idx])
        recent = get_recent_data(self.data, t0, self.num_recent)
        daily = get_days_data(self.data, t0, self.num_days, self.pred_len, self.points_per_day)
        weekly = get_weeks_data(self.data, t0, self.num_weeks, self.pred_len, self.points_per_day)
        target = get_target_data(self.data, t0, self.pred_len, self.target_dim)

        return {
            "recent": torch.from_numpy(_time_first_to_node_feature_time(recent)),
            "daily": torch.from_numpy(_time_first_to_node_feature_time(daily)),
            "weekly": torch.from_numpy(_time_first_to_node_feature_time(weekly)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
