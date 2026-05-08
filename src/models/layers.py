"""ASTGCN 基础层。

本文件只写模型层，不读取配置、不读取数据、不管理训练流程。
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
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
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self.num_of_timesteps = num_of_timesteps
        self.u1 = nn.Parameter(torch.empty(num_of_vertices))
        self.u2 = nn.Parameter(torch.empty(in_channels, num_of_vertices))
        self.u3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        self.ve = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))

    def forward(self, x: Any) -> Any:
        """计算时间注意力矩阵。"""
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.u1), self.u2)
        rhs = torch.matmul(self.u3, x)
        product = torch.matmul(lhs, rhs)
        attention = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        return F.softmax(attention, dim=1)


class SpatialAttention(nn.Module):
    """空间注意力层。

    输入:
        x: (B, N, F, T)
    输出:
        spatial_attention: (B, N, N)
    """

    def __init__(self, num_of_vertices: int, in_channels: int, num_of_timesteps: int) -> None:
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self.num_of_timesteps = num_of_timesteps
        self.w1 = nn.Parameter(torch.empty(num_of_timesteps))
        self.w2 = nn.Parameter(torch.empty(in_channels, num_of_timesteps))
        self.w3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices))
        self.vs = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))

    def forward(self, x: Any) -> Any:
        """计算空间注意力矩阵。"""
        lhs = torch.matmul(torch.matmul(x, self.w1), self.w2)
        rhs = torch.matmul(self.w3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        attention = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        return F.softmax(attention, dim=1)


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
        if k <= 0:
            raise ValueError("k 必须大于 0。")
        if len(cheb_polynomials) < k:
            raise ValueError(f"Chebyshev 多项式数量不足: 需要 {k}，实际 {len(cheb_polynomials)}")

        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.registered_polynomial_names = []
        for index, polynomial in enumerate(cheb_polynomials[:k]):
            tensor = torch.as_tensor(polynomial, dtype=torch.float32)
            name = f"cheb_polynomial_{index}"
            self.register_buffer(name, tensor)
            self.registered_polynomial_names.append(name)
        self.theta = nn.ParameterList([nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(k)])

    def forward(self, x: Any, spatial_attention: Any) -> Any:
        """执行带注意力的 Chebyshev 图卷积。"""
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = x.new_zeros(batch_size, num_of_vertices, self.out_channels)

            for index, theta_k in enumerate(self.theta):
                t_k = getattr(self, self.registered_polynomial_names[index])
                t_k_with_attention = t_k * spatial_attention
                rhs = t_k_with_attention.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


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
        self.in_channels = in_channels
        self.k = k
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.cheb_polynomials = cheb_polynomials
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps
        self.temporal_attention = TemporalAttention(num_of_vertices, in_channels, num_of_timesteps)
        self.spatial_attention = SpatialAttention(num_of_vertices, in_channels, num_of_timesteps)
        self.cheb_graph_conv = ChebGraphConvWithAttention(k, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.layer_norm = nn.LayerNorm(nb_time_filter)

    def forward(self, x: Any) -> Any:
        """执行一个 ASTGCN block 的前向传播。"""
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_attention = self.temporal_attention(x)
        x_temporal = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps),
            temporal_attention,
        ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        spatial_attention = self.spatial_attention(x_temporal)
        spatial_gcn = self.cheb_graph_conv(x, spatial_attention)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        residual_output = self.residual_conv(x.permute(0, 2, 1, 3))
        output = F.relu(residual_output + time_conv_output)
        output = self.layer_norm(output.permute(0, 3, 2, 1))
        return output.permute(0, 2, 3, 1)
