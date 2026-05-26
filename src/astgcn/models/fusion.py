from typing import Dict, List

import torch
import torch.nn as nn

from astgcn.models.ablation import FusionMode


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

    def __init__(
        self,
        num_nodes: int,
        pred_len: int,
        branches: List[str] | None = None,
        mode: FusionMode = "matrix",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.branches = ["recent", "daily", "weekly"] if branches is None else list(branches)
        self.mode = mode
        if not self.branches:
            raise ValueError("FusionLayer 至少需要一个分支")

        if mode == "matrix":
            if self.branches == ["recent", "daily", "weekly"]:
                self.W_h = nn.Parameter(torch.ones(num_nodes, pred_len))
                self.W_d = nn.Parameter(torch.ones(num_nodes, pred_len))
                self.W_w = nn.Parameter(torch.ones(num_nodes, pred_len))
            else:
                self.branch_weights = nn.Parameter(torch.ones(len(self.branches), num_nodes, pred_len))
        elif mode == "scalar":
            self.branch_weights = nn.Parameter(torch.ones(len(self.branches)))
        elif mode == "concat_mlp":
            self.mlp = nn.Linear(len(self.branches) * pred_len, pred_len)
        elif mode != "average":
            raise ValueError(f"不支持的 fusion mode: {mode}")

    def forward(
        self,
        components: Dict[str, torch.Tensor],
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
        tensors = []
        expected = (self.num_nodes, self.pred_len)
        for name in self.branches:
            if name not in components:
                raise ValueError(f"缺少分支输出: {name}")
            tensor = components[name]
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected:
                raise ValueError(
                    f"{name} 必须是 [B, N, pred_len]，"
                    f"期望后两维 {expected}，实际 shape={tuple(tensor.shape)}"
                )
            tensors.append(tensor)

        stacked = torch.stack(tensors, dim=0)
        if self.mode == "average":
            return stacked.mean(dim=0)
        if self.mode == "scalar":
            weights = torch.softmax(self.branch_weights, dim=0).view(-1, 1, 1, 1)
            return (weights * stacked).sum(dim=0)
        if self.mode == "concat_mlp":
            return self.mlp(torch.cat(tensors, dim=-1))
        if hasattr(self, "branch_weights"):
            return (self.branch_weights.unsqueeze(1) * stacked).sum(dim=0)
        weights = {
            "recent": self.W_h,
            "daily": self.W_d,
            "weekly": self.W_w,
        }
        return sum(weights[name] * tensor for name, tensor in zip(self.branches, tensors))


__all__ = ["FusionLayer"]
