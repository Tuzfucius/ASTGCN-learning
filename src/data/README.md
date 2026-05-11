# `src/data` 数据模块说明

本目录负责数据预处理和数据加载。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `preprocessing.py` | 从原始 `.npz` 生成监督学习数据。 |
| `dataset.py` | 从预处理 `.npz` 构造 PyTorch DataLoader。 |

## `preprocessing.py`

建议函数：

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `search_data` | 序列长度、依赖数量、预测起点、预测长度、单位跨度 | 历史窗口索引 | 查找历史片段。 |
| `get_sample_indices` | 原始序列、周/天/小时依赖、预测起点 | 单个样本 | 生成一个监督样本。 |
| `generate_dataset` | 原始数据路径、配置变量 | 数据集字典 | 生成 train/val/test。 |
| `standardize` | train/val/test 输入 | 标准化结果 | 用训练集统计量标准化。 |
| `save_dataset` | 数据集字典、保存路径 | 无 | 保存 `.npz`。 |

## `dataset.py`

建议函数：

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `load_processed_dataset` | 预处理 `.npz` 路径 | 数组字典 | 读取 train/val/test。 |
| `build_dataloader` | 输入数组、目标数组、batch size | DataLoader | 构造 PyTorch DataLoader。 |
| `build_all_dataloaders` | 配置对象 | train/val/test loaders | 一次性构造三类加载器。 |

## 张量约定

```text
train_x: (B_train, N, F, T)
train_target: (B_train, N, T_pred)
```

PEMS04 第一阶段：

```text
train_x: (B_train, 307, 1, 12)
train_target: (B_train, 307, 12)
```

当前阶段只使用 PEMS04、第 0 个特征和 ASTGCN recent component。`num_of_days`、`num_of_weeks` 保持为 0，`num_of_hours` 至少为 1。

标准化约定：

- 只标准化输入 `x`，使用训练集输入统计量计算 `mean` 和 `std`。
- `target` 保持原始数值尺度，训练 loss、验证 loss 和评估指标直接对模型输出与 `target` 计算。
- 预处理后的 `mean`、`std` 只用于记录输入标准化参数，不用于当前阶段的预测反标准化。

## 变量说明

| 变量 | 含义 |
| --- | --- |
| `num_of_hours` | recent component 数量。 |
| `num_of_days` | daily component 数量，第一阶段为 0。 |
| `num_of_weeks` | weekly component 数量，第一阶段为 0。 |
| `points_per_hour` | 每小时采样点数。 |
| `num_for_predict` | 预测未来步数。 |
| `target_channel` | 预测目标特征索引。 |

## 注意事项

- 预处理阶段可以读取原始 `.npz`。
- 加载阶段不应重新切片原始数据。
- 标准化统计量只能由训练集计算。
- 第一阶段只保留第 0 个特征作为输入和目标。
- 预处理和加载阶段都会校验 `N`、`F`、`T`、`T_pred`、样本数和 batch size，报错信息应能定位到具体字段。
