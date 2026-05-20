"""历史平均基线模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoricalAverage(nn.Module):
    """使用历史输入窗口的目标特征均值作为未来预测。

    该基线不包含可学习参数，适合验证数据加载、反标准化和指标计算链路。
    """

    def __init__(self, pred_len: int, target_dim: int = 0) -> None:
        """初始化历史平均模型。

        :param pred_len: 预测长度。
        :param target_dim: 目标特征编号。
        """
        super().__init__()
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, recent: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        """根据 recent 输入生成预测。

        :param recent: recent 输入，形状为 ``[B, N, F, T]``。
        :return: 预测值，形状为 ``[B, N, pred_len]``。
        """
        if recent.ndim != 4:
            raise ValueError(f"recent 必须是 [B, N, F, T]，当前 shape={tuple(recent.shape)}")
        if self.target_dim >= recent.shape[2]:
            raise ValueError("target_dim 超出 recent 的特征维度范围")
        value = recent[:, :, self.target_dim, :].mean(dim=-1, keepdim=True)
        return value.repeat(1, 1, self.pred_len)


__all__ = ["HistoricalAverage"]
