from typing import Union

import numpy as np
import torch
import torch.nn as nn


ArrayLike = Union[np.ndarray, torch.Tensor]


class ChebGraphConv(nn.Module):
    """
    Chebyshev 图卷积层。

    输入:
        x: [B, N, F_in, T]

    输出:
        out: [B, N, F_out, T]

    其中 Chebyshev 多项式 buffer 的形状为 [K, N, N]。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cheb_polynomials: ArrayLike,
    ) -> None:
        super().__init__()
        cheb_tensor = _as_cheb_tensor(cheb_polynomials)
        k_order = cheb_tensor.shape[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_order = k_order
        self.num_nodes = cheb_tensor.shape[1]

        self.theta = nn.Parameter(torch.empty(k_order, in_channels, out_channels))
        self.register_buffer("cheb_polynomials", cheb_tensor, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化 Chebyshev 图卷积参数。"""
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 Chebyshev 图卷积。

        输入:
            x: [B, N, F_in, T]

        输出:
            out: [B, N, F_out, T]
        """
        _check_x(x, self.num_nodes, self.in_channels)
        outputs = []
        for k in range(self.k_order):
            support = self.cheb_polynomials[k]
            graph_signal = torch.einsum("nm,bmft->bnft", support, x)
            outputs.append(torch.einsum("bnft,fo->bnot", graph_signal, self.theta[k]))
        return torch.stack(outputs, dim=0).sum(dim=0)


class ChebGraphConvWithSAtt(nn.Module):
    """
    带空间注意力的 Chebyshev 图卷积层。

    输入:
        x: [B, N, F_in, T]
        spatial_attention: [B, N, N]

    输出:
        out: [B, N, F_out, T]

    其中 Chebyshev 多项式 buffer 的形状为 [K, N, N]，并通过
    register_buffer(..., persistent=False) 注册。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cheb_polynomials: ArrayLike,
    ) -> None:
        super().__init__()
        cheb_tensor = _as_cheb_tensor(cheb_polynomials)
        k_order = cheb_tensor.shape[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_order = k_order
        self.num_nodes = cheb_tensor.shape[1]

        self.theta = nn.Parameter(torch.empty(k_order, in_channels, out_channels))
        self.register_buffer("cheb_polynomials", cheb_tensor, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化带空间注意力的 Chebyshev 图卷积参数。"""
        nn.init.xavier_uniform_(self.theta)

    def forward(
        self,
        x: torch.Tensor,
        spatial_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行带空间注意力的 Chebyshev 图卷积。

        输入:
            x: [B, N, F_in, T]
            spatial_attention: [B, N, N]

        输出:
            out: [B, N, F_out, T]
        """
        _check_x(x, self.num_nodes, self.in_channels)
        if spatial_attention.shape != (x.shape[0], self.num_nodes, self.num_nodes):
            raise ValueError(
                "spatial_attention 必须是 [B, N, N]，"
                f"期望 {(x.shape[0], self.num_nodes, self.num_nodes)}，"
                f"实际 {tuple(spatial_attention.shape)}"
            )

        outputs = []
        for k in range(self.k_order):
            support = self.cheb_polynomials[k].unsqueeze(0) * spatial_attention
            graph_signal = torch.einsum("bnm,bmft->bnft", support, x)
            outputs.append(torch.einsum("bnft,fo->bnot", graph_signal, self.theta[k]))
        return torch.stack(outputs, dim=0).sum(dim=0)


def _as_cheb_tensor(cheb_polynomials: ArrayLike) -> torch.Tensor:
    if isinstance(cheb_polynomials, np.ndarray):
        cheb_tensor = torch.from_numpy(cheb_polynomials)
    elif isinstance(cheb_polynomials, torch.Tensor):
        cheb_tensor = cheb_polynomials.detach().clone()
    else:
        raise TypeError("cheb_polynomials 必须是 numpy.ndarray 或 torch.Tensor")

    cheb_tensor = cheb_tensor.float()
    if cheb_tensor.ndim != 3:
        raise ValueError(
            "cheb_polynomials 必须是 [K, N, N]，"
            f"当前 shape={tuple(cheb_tensor.shape)}"
        )
    if cheb_tensor.shape[1] != cheb_tensor.shape[2]:
        raise ValueError("cheb_polynomials 的后两维必须是方阵 [N, N]")
    return cheb_tensor


def _check_x(x: torch.Tensor, num_nodes: int, in_channels: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")
    if x.shape[1] != num_nodes or x.shape[2] != in_channels:
        raise ValueError(
            "x 的节点数或特征数与模块初始化参数不匹配："
            f"期望 N={num_nodes}, F={in_channels}，"
            f"实际 N={x.shape[1]}, F={x.shape[2]}"
        )


__all__ = ["ChebGraphConv", "ChebGraphConvWithSAtt"]
