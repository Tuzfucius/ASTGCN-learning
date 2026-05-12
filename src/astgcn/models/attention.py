# TemporalAttention 时间注意力
# 模型内部输入使用 x.shape == [B, N, C, T]
# B：batch size
# N：节点数
# C：特征通道数
# T：时间长度 
# 注意dataloader输出是[B, T, N, F]，需要进行转化

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    时间注意力层。

    输入:
        x: [B, N, C, T]

    输出:
        E: [B, T, T]

    E[i, j] 表示时间片 i 对时间片 j 的相关强度。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        num_timesteps: int,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps

        self.U1 = nn.Parameter(torch.empty(num_nodes))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.empty(in_channels))

        self.be = nn.Parameter(torch.empty(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.empty(num_timesteps, num_timesteps))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U1.unsqueeze(0))
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.U3.unsqueeze(0))
        nn.init.xavier_uniform_(self.be)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, C, T]
        :return: [B, T, T]
        """
        if x.ndim != 4:
            raise ValueError(f"x 必须是四维张量 [B, N, C, T]，当前 shape={x.shape}")

        # [B, N, C, T] -> [B, T, C, N]
        x_perm = x.permute(0, 3, 2, 1)

        # [B, T, C, N] @ [N] -> [B, T, C]
        lhs = torch.matmul(x_perm, self.U1)

        # [B, T, C] @ [C, N] -> [B, T, N]
        lhs = torch.matmul(lhs, self.U2)

        # [C] @ [B, N, C, T] -> [B, N, T]
        rhs = torch.matmul(self.U3, x)

        # [B, T, N] @ [B, N, T] -> [B, T, T]
        product = torch.matmul(lhs, rhs)

        # 加偏置、激活、线性变换
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))

        # 对最后一维 softmax，使每个时间片对其他时间片的权重和为 1
        E = F.softmax(E, dim=-1)

        return E