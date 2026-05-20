import torch
import torch.nn as nn
import torch.nn.functional as F

from astgcn.models.attention import (
    SpatialAttention,
    TemporalAttention,
    apply_temporal_attention,
)
from astgcn.models.cheb_conv import ChebGraphConvWithSAtt
from astgcn.models.temporal_conv import TemporalConv


class STBlock(nn.Module):
    """
    ASTGCN 时空块。

    输入:
        x: [B, N, F_in, T]

    输出:
        out: [B, N, F_out, T]

    数据流为 temporal attention -> spatial attention -> Chebyshev graph
    convolution -> temporal convolution -> residual + layer norm。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        num_timesteps: int,
        cheb_polynomials,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        self.temporal_attention = TemporalAttention(
            num_nodes=num_nodes,
            in_channels=in_channels,
            num_timesteps=num_timesteps,
        )
        self.spatial_attention = SpatialAttention(
            num_nodes=num_nodes,
            in_channels=in_channels,
            num_timesteps=num_timesteps,
        )
        self.cheb_conv_satt = ChebGraphConvWithSAtt(
            in_channels=in_channels,
            out_channels=out_channels,
            cheb_polynomials=cheb_polynomials,
        )
        self.temporal_conv = TemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel_size,
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行一个时空块。

        输入:
            x: [B, N, F_in, T]

        输出:
            out: [B, N, F_out, T]
        """
        if x.ndim != 4:
            raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")

        temporal_attention = self.temporal_attention(x)
        x_tatt = apply_temporal_attention(x, temporal_attention)

        spatial_attention = self.spatial_attention(x_tatt)
        x_gcn = self.cheb_conv_satt(x_tatt, spatial_attention)
        x_time = self.temporal_conv(x_gcn)

        residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = F.relu(x_time + residual)
        out = self.layer_norm(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return out


__all__ = ["STBlock"]
