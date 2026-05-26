# models 目录说明

本目录实现 ASTGCN 网络结构。ASTGCN 同时建模交通路网的空间依赖和时间依赖，并融合 recent、daily、weekly 三类历史信息。

## 主要模块

- `attention.py`：Temporal Attention、Spatial Attention 和时间注意力应用函数。
- `cheb_conv.py`：Chebyshev 图卷积和带空间注意力的图卷积。
- `temporal_conv.py`：时间卷积。
- `st_block.py`：时空块，组合注意力、图卷积、时间卷积、残差和归一化。
- `component.py`：单个 ASTGCN 分支组件。
- `fusion.py`：recent/daily/weekly 三分支可学习融合。
- `astgcn.py`：顶层 ASTGCN 模型。

## 输入输出

```text
recent/daily/weekly: [B, N, F, T]
component output:    [B, N, T_p]
final output:        [B, N, T_p]
```

`ASTGCN.forward(..., return_components=True)` 会返回 `prediction/recent/daily/weekly`，用于 notebook 和对比脚本可视化。

## 消融配置

`ablation.py` 提供 `AblationConfig`，用于统一管理消融实验开关：

- 时间分支：`use_recent`、`use_daily`、`use_weekly`
- ST-Block：`use_temporal_attention`、`use_spatial_attention`、`use_graph_conv`、`use_temporal_conv`
- 图结构：`graph_mode` 支持 `cheb`、`identity`、`random`、`none`
- 融合方式：`fusion_mode` 支持 `matrix`、`average`、`scalar`、`concat_mlp`
