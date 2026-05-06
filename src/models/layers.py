"""ASTGCN 基础层。

本文件只写模型层，不读取配置、不读取数据、不管理训练流程。
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # 允许在未安装 PyTorch 时阅读代码骨架。
    torch = None

    class _FallbackModule:
        """未安装 PyTorch 时的占位基类。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()


class TemporalAttention(nn.Module):
    """时间注意力层。

    输入:
        x: (B, N, F, T)
    输出:
        temporal_attention: (B, T, T)
    """

    def __init__(self, num_of_vertices: int, in_channels: int, num_of_timesteps: int) -> None:
        super().__init__()
        # TODO: 定义时间注意力所需的可学习参数。
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self.num_of_timesteps = num_of_timesteps

    def forward(self, x: Any) -> Any:
        """计算时间注意力矩阵。"""
        # TODO: 按官方 ASTGCN 公式实现时间注意力。
        raise NotImplementedError("TODO: 实现 TemporalAttention.forward。")


class SpatialAttention(nn.Module):
    """空间注意力层。

    输入:
        x: (B, N, F, T)
    输出:
        spatial_attention: (B, N, N)
    """

    def __init__(self, num_of_vertices: int, in_channels: int, num_of_timesteps: int) -> None:
        super().__init__()
        # TODO: 定义空间注意力所需的可学习参数。
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self.num_of_timesteps = num_of_timesteps

    def forward(self, x: Any) -> Any:
        """计算空间注意力矩阵。"""
        # TODO: 按官方 ASTGCN 公式实现空间注意力。
        raise NotImplementedError("TODO: 实现 SpatialAttention.forward。")


class ChebGraphConvWithAttention(nn.Module):
    """带空间注意力的 Chebyshev 图卷积。

    输入:
        x: (B, N, F, T)
        spatial_attention: (B, N, N)
    输出:
        output: (B, N, nb_chev_filter, T)
    """

    def __init__(self, k: int, cheb_polynomials: list[Any], in_channels: int, out_channels: int) -> None:
        super().__init__()
        # TODO: 定义每一阶 Chebyshev 多项式对应的 Theta 参数。
        self.k = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Any, spatial_attention: Any) -> Any:
        """执行带注意力的 Chebyshev 图卷积。"""
        # TODO: 遍历每个时间步和每个 Chebyshev 阶数，完成图卷积。
        raise NotImplementedError("TODO: 实现 ChebGraphConvWithAttention.forward。")


class ASTGCNBlock(nn.Module):
    """ASTGCN 基础块。

    组成:
    - 时间注意力
    - 空间注意力
    - 带空间注意力的图卷积
    - 时间卷积
    - 残差连接
    - LayerNorm
    """

    def __init__(
        self,
        in_channels: int,
        k: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        cheb_polynomials: list[Any],
        num_of_vertices: int,
        num_of_timesteps: int,
    ) -> None:
        super().__init__()
        # TODO: 初始化 TemporalAttention、SpatialAttention、图卷积、时间卷积、残差卷积、LayerNorm。
        self.in_channels = in_channels
        self.k = k
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.cheb_polynomials = cheb_polynomials
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps

    def forward(self, x: Any) -> Any:
        """执行一个 ASTGCN block 的前向传播。"""
        # TODO: 依次执行时间注意力、空间注意力、图卷积、时间卷积、残差和归一化。
        raise NotImplementedError("TODO: 实现 ASTGCNBlock.forward。")
