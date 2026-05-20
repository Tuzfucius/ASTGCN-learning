# PEMS04 数据说明

本目录保存 PEMS04 数据集相关文件。

## 数据集基本信息

| 项目 | 说明 |
| --- | --- |
| 数据集名称 | PEMS04 |
| 节点数量 | 307 |
| 采样间隔 | 5 分钟 |
| 每天采样点 | 288 |
| 原始特征数 | 3 |
| 默认预测目标 | 第 0 个特征未来 12 个时间步 |

## 当前任务定义

当前项目使用 ASTGCN 三类历史依赖：

```text
recent: 最近 12 个时间步
daily:  过去 1 天中相同时间段的 12 个时间步
weekly: 过去 1 周中相同星期和时间段的 12 个时间步
target: 未来 12 个时间步
```

DataLoader 批量输出：

```text
recent/daily/weekly: [B, 307, 3, 12]
target:              [B, 307, 12]
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `pems04.npz` | 原始时序数据，key 为 `data`，形状为 `[16992, 307, 3]`。 |
| `distance.csv` | 节点距离边表，用于构造邻接矩阵和 Chebyshev 多项式。 |
| `PEMS04_r1_d0_w0_astcgn.npz` | 旧的 recent-only 预处理样例，仅作为参考，不是当前主流程依赖。 |
| `README.md` | 本说明文件。 |

## 当前主流程

当前软件包不依赖预处理后的 `.npz` 文件，而是直接从 `pems04.npz` 和 `distance.csv` 构造 DataLoader、图结构和训练样本。这样更适合教学复现，也方便在 Kaggle notebook 中自动定位数据路径。
