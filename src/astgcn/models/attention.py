import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    时间注意力层。

    输入:
        x: 形状为 [B, N, F, T]，其中 B 为批量大小，N 为节点数，
           F 为输入特征数，T 为时间步数。

    输出:
        E: 形状为 [B, T, T]，表示每个样本内不同时间片之间的注意力权重。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        num_timesteps: int,
    ) -> None:
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

    def reset_parameters(self) -> None:
        """初始化时间注意力层参数。"""
        nn.init.xavier_uniform_(self.U1.unsqueeze(0))
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.U3.unsqueeze(0))
        nn.init.xavier_uniform_(self.be)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算时间注意力矩阵。

        输入:
            x: [B, N, F, T]

        输出:
            E: [B, T, T]
        """
        _check_4d_input(x, self.num_nodes, self.in_channels, self.num_timesteps)

        lhs = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
        lhs = torch.matmul(lhs, self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)

        attention = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(attention, dim=-1)


class SpatialAttention(nn.Module):
    """
    空间注意力层。

    输入:
        x: 形状为 [B, N, F, T]，其中 B 为批量大小，N 为节点数，
           F 为输入特征数，T 为时间步数。

    输出:
        S: 形状为 [B, N, N]，表示每个样本内节点之间的注意力权重。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        num_timesteps: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps

        self.W1 = nn.Parameter(torch.empty(num_timesteps))
        self.W2 = nn.Parameter(torch.empty(in_channels, num_timesteps))
        self.W3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.empty(num_nodes, num_nodes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化空间注意力层参数。"""
        nn.init.xavier_uniform_(self.W1.unsqueeze(0))
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3.unsqueeze(0))
        nn.init.xavier_uniform_(self.bs)
        nn.init.xavier_uniform_(self.Vs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算空间注意力矩阵。

        输入:
            x: [B, N, F, T]

        输出:
            S: [B, N, N]
        """
        _check_4d_input(x, self.num_nodes, self.in_channels, self.num_timesteps)

        lhs = torch.matmul(x, self.W1)
        lhs = torch.matmul(lhs, self.W2)
        rhs = torch.matmul(self.W3, x).transpose(1, 2)
        product = torch.matmul(lhs, rhs)

        attention = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(attention, dim=-1)


def apply_temporal_attention(
    x: torch.Tensor,
    temporal_attention: torch.Tensor,
) -> torch.Tensor:
    """
    将时间注意力作用到输入张量。

    输入:
        x: [B, N, F, T]
        temporal_attention: [B, T, T]

    输出:
        out: [B, N, F, T]
    """
    if x.ndim != 4:
        raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")
    if temporal_attention.ndim != 3:
        raise ValueError(
            "temporal_attention 必须是 [B, T, T]，"
            f"当前 shape={tuple(temporal_attention.shape)}"
        )
    if x.shape[0] != temporal_attention.shape[0] or x.shape[-1] != temporal_attention.shape[1]:
        raise ValueError(
            "x 与 temporal_attention 的批量大小或时间长度不匹配："
            f"x={tuple(x.shape)}, temporal_attention={tuple(temporal_attention.shape)}"
        )
    return torch.einsum("bnft,btu->bnfu", x, temporal_attention)


def _check_4d_input(
    x: torch.Tensor,
    num_nodes: int,
    in_channels: int,
    num_timesteps: int,
) -> None:
    if x.ndim != 4:
        raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")
    _, nodes, channels, timesteps = x.shape
    expected = (num_nodes, in_channels, num_timesteps)
    actual = (nodes, channels, timesteps)
    if actual != expected:
        raise ValueError(
            "x 的 [N, F, T] 与模块初始化参数不匹配："
            f"期望 {expected}，实际 {actual}"
        )


__all__ = ["TemporalAttention", "SpatialAttention", "apply_temporal_attention"]
