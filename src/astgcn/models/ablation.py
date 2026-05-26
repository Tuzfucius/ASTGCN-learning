from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal


GraphMode = Literal["cheb", "identity", "random", "none"]
FusionMode = Literal["matrix", "average", "scalar", "concat_mlp"]


@dataclass
class AblationConfig:
    """ASTGCN 消融实验配置。"""

    # 三个时间分支
    use_recent: bool = True
    use_daily: bool = True
    use_weekly: bool = True

    # ST-Block 内部结构
    use_temporal_attention: bool = True
    use_spatial_attention: bool = True
    use_graph_conv: bool = True
    use_temporal_conv: bool = True

    # 图结构实验
    graph_mode: GraphMode = "cheb"

    # 融合方式实验
    fusion_mode: FusionMode = "matrix"

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "AblationConfig":
        """从配置字典构造消融配置。"""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError("ablation 配置必须是字典")
        valid_fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"未知 ablation 配置项: {unknown}")
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """校验配置组合是否合法。"""
        if not self.active_branches:
            raise ValueError("至少需要启用一个时间分支")
        if self.graph_mode not in ("cheb", "identity", "random", "none"):
            raise ValueError(f"不支持的 graph_mode: {self.graph_mode}")
        if self.fusion_mode not in ("matrix", "average", "scalar", "concat_mlp"):
            raise ValueError(f"不支持的 fusion_mode: {self.fusion_mode}")

    @property
    def active_branches(self) -> List[str]:
        """返回当前启用的时间分支名称。"""
        branches = []
        if self.use_recent:
            branches.append("recent")
        if self.use_daily:
            branches.append("daily")
        if self.use_weekly:
            branches.append("weekly")
        return branches


__all__ = ["AblationConfig", "FusionMode", "GraphMode"]
