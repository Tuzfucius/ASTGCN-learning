from __future__ import annotations

import torch
import torch.nn as nn

from astgcn.models.ablation import AblationConfig
from astgcn.models.component import ASTGCNComponent
from astgcn.models.fusion import FusionLayer


class ASTGCN(nn.Module):
    """
    论文式三组件 ASTGCN 模型。

    输入:
        recent: [B, N, F, T_h]
        daily: [B, N, F, T_d]
        weekly: [B, N, F, T_w]

    输出:
        return_components=False 时返回 prediction: [B, N, pred_len]
        return_components=True 时返回字典，包含 prediction、recent、daily、weekly，
        每个组件预测形状均为 [B, N, pred_len]。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        pred_len: int,
        cheb_polynomials,
        recent_timesteps: int,
        daily_timesteps: int = None,
        weekly_timesteps: int = None,
        num_blocks: int = 2,
        hidden_channels: int = 64,
        temporal_kernel_size: int = 3,
        ablation_config: AblationConfig | dict | None = None,
    ) -> None:
        super().__init__()
        self.ablation_config = (
            AblationConfig.from_dict(ablation_config)
            if isinstance(ablation_config, dict) or ablation_config is None
            else ablation_config
        )
        self.ablation_config.validate()
        daily_timesteps = recent_timesteps if daily_timesteps is None else daily_timesteps
        weekly_timesteps = recent_timesteps if weekly_timesteps is None else weekly_timesteps

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.pred_len = pred_len
        self.recent_timesteps = recent_timesteps
        self.daily_timesteps = daily_timesteps
        self.weekly_timesteps = weekly_timesteps

        component_kwargs = {
            "num_blocks": num_blocks,
            "num_nodes": num_nodes,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "pred_len": pred_len,
            "cheb_polynomials": cheb_polynomials,
            "temporal_kernel_size": temporal_kernel_size,
            "ablation_config": self.ablation_config,
        }
        self.components = nn.ModuleDict()
        if self.ablation_config.use_recent:
            self.components["recent"] = ASTGCNComponent(
                num_timesteps=recent_timesteps,
                **component_kwargs,
            )
        if self.ablation_config.use_daily:
            self.components["daily"] = ASTGCNComponent(
                num_timesteps=daily_timesteps,
                **component_kwargs,
            )
        if self.ablation_config.use_weekly:
            self.components["weekly"] = ASTGCNComponent(
                num_timesteps=weekly_timesteps,
                **component_kwargs,
            )
        self.fusion = FusionLayer(
            num_nodes=num_nodes,
            pred_len=pred_len,
            branches=self.ablation_config.active_branches,
            mode=self.ablation_config.fusion_mode,
        )

    def forward(
        self,
        recent: torch.Tensor,
        daily: torch.Tensor,
        weekly: torch.Tensor,
        return_components: bool = False,
    ):
        """
        执行三组件 ASTGCN 前向计算。

        输入:
            recent: [B, N, F, T_h]
            daily: [B, N, F, T_d]
            weekly: [B, N, F, T_w]
            return_components: 是否返回三个组件的中间预测。

        输出:
            return_components=False: prediction [B, N, pred_len]
            return_components=True: 字典，包含:
                prediction: [B, N, pred_len]
                recent: [B, N, pred_len]
                daily: [B, N, pred_len]
                weekly: [B, N, pred_len]
        """
        inputs = {
            "recent": recent,
            "daily": daily,
            "weekly": weekly,
        }
        component_outputs = {
            name: component(inputs[name])
            for name, component in self.components.items()
        }
        prediction = self.fusion(component_outputs)

        if return_components:
            return {"prediction": prediction, **component_outputs}
        return prediction


__all__ = ["ASTGCN"]
