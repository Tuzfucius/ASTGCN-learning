import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    ASTGCN 三组件融合层。

    输入:
        y_h: [B, N, pred_len]
        y_d: [B, N, pred_len]
        y_w: [B, N, pred_len]

    输出:
        y: [B, N, pred_len]

    其中 W_h、W_d、W_w 的形状均为 [N, pred_len]。
    """

    def __init__(self, num_nodes: int, pred_len: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len

        self.W_h = nn.Parameter(torch.ones(num_nodes, pred_len))
        self.W_d = nn.Parameter(torch.ones(num_nodes, pred_len))
        self.W_w = nn.Parameter(torch.ones(num_nodes, pred_len))

    def forward(
        self,
        y_h: torch.Tensor,
        y_d: torch.Tensor,
        y_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        融合 recent、daily、weekly 三个组件输出。

        输入:
            y_h: [B, N, pred_len]
            y_d: [B, N, pred_len]
            y_w: [B, N, pred_len]

        输出:
            y: [B, N, pred_len]
        """
        expected = (self.num_nodes, self.pred_len)
        for name, tensor in (("y_h", y_h), ("y_d", y_d), ("y_w", y_w)):
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected:
                raise ValueError(
                    f"{name} 必须是 [B, N, pred_len]，"
                    f"期望后两维 {expected}，实际 shape={tuple(tensor.shape)}"
                )

        return self.W_h * y_h + self.W_d * y_d + self.W_w * y_w


__all__ = ["FusionLayer"]
