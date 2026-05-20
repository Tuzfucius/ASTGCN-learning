import torch
import torch.nn as nn


class TemporalConv(nn.Module):
    """
    沿时间维度执行二维卷积的时间卷积层。

    输入:
        x: [B, N, F_in, T]

    输出:
        out: [B, N, F_out, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数，以保持时间长度不变")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行时间卷积。

        输入:
            x: [B, N, F_in, T]

        输出:
            out: [B, N, F_out, T]
        """
        if x.ndim != 4:
            raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"x 的特征数必须为 {self.in_channels}，实际为 {x.shape[2]}"
            )
        out = self.conv(x.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3)


__all__ = ["TemporalConv"]
