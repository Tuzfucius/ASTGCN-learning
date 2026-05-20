"""GRU 基线模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """逐节点共享参数的 GRU 预测基线。

    模型把每个节点的 recent 时间序列视为一个样本，使用同一套 GRU 参数预测
    未来 ``pred_len`` 个时间步。输入输出接口与 ``LSTMBaseline`` 保持一致。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        pred_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """初始化 GRU 基线。

        :param in_channels: 输入特征数 ``F``。
        :param hidden_size: GRU 隐状态维度。
        :param pred_len: 预测长度。
        :param num_layers: GRU 层数。
        :param dropout: GRU 层间 dropout，单层时自动置为 0。
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, pred_len)

    def forward(self, recent: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        """根据 recent 输入生成预测。

        :param recent: recent 输入，形状为 ``[B, N, F, T]``。
        :return: 预测值，形状为 ``[B, N, pred_len]``。
        """
        if recent.ndim != 4:
            raise ValueError(f"recent 必须是 [B, N, F, T]，当前 shape={tuple(recent.shape)}")
        if recent.shape[2] != self.in_channels:
            raise ValueError(f"输入特征数必须为 {self.in_channels}，实际为 {recent.shape[2]}")
        batch_size, num_nodes, _, num_timesteps = recent.shape
        x = recent.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, num_timesteps, self.in_channels)
        _, hidden = self.gru(x)
        pred = self.projection(hidden[-1])
        return pred.reshape(batch_size, num_nodes, self.pred_len)


__all__ = ["GRUBaseline"]
