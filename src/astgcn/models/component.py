import torch
import torch.nn as nn

from astgcn.models.st_block import STBlock


class ASTGCNComponent(nn.Module):
    """
    ASTGCN 单个时间依赖组件。

    输入:
        x: [B, N, F, T]

    输出:
        y: [B, N, pred_len]

    该组件可用于 recent、daily、weekly 三个分支中的任意一个。
    """

    def __init__(
        self,
        num_blocks: int,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        num_timesteps: int,
        pred_len: int,
        cheb_polynomials,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks 必须是正整数")
        self.num_blocks = num_blocks
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_timesteps = num_timesteps
        self.pred_len = pred_len

        blocks = []
        current_channels = in_channels
        for _ in range(num_blocks):
            blocks.append(
                STBlock(
                    num_nodes=num_nodes,
                    in_channels=current_channels,
                    out_channels=hidden_channels,
                    num_timesteps=num_timesteps,
                    cheb_polynomials=cheb_polynomials,
                    temporal_kernel_size=temporal_kernel_size,
                )
            )
            current_channels = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(hidden_channels * num_timesteps, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算单个 ASTGCN 组件的预测。

        输入:
            x: [B, N, F, T]

        输出:
            y: [B, N, pred_len]
        """
        if x.ndim != 4:
            raise ValueError(f"x 必须是 [B, N, F, T]，当前 shape={tuple(x.shape)}")
        if x.shape[1] != self.num_nodes or x.shape[2] != self.in_channels:
            raise ValueError(
                "x 的节点数或特征数与组件初始化参数不匹配："
                f"期望 N={self.num_nodes}, F={self.in_channels}，"
                f"实际 N={x.shape[1]}, F={x.shape[2]}"
            )
        if x.shape[3] != self.num_timesteps:
            raise ValueError(
                f"x 的时间长度必须为 {self.num_timesteps}，实际为 {x.shape[3]}"
            )

        for block in self.blocks:
            x = block(x)

        x = x.reshape(x.shape[0], x.shape[1], self.hidden_channels * self.num_timesteps)
        return self.projection(x)


__all__ = ["ASTGCNComponent"]
