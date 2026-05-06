# 配置目录说明

本目录保存实验配置文件。配置文件只描述变量，不写训练逻辑。

当前建议使用：

- `PEMS04_astgcn.yaml`：PEMS04 数据集的 ASTGCN recent component 配置模板。

## 配置分组

### `data`

数据文件和数据集基础信息。

| 变量 | 含义 | 建议值 |
| --- | --- | --- |
| `dataset_name` | 数据集名称，用于实验目录命名。 | `PEMS04` |
| `graph_signal_matrix_filename` | 原始图信号数据文件。 | `data/PEMS04/pems04.npz` |
| `processed_dataset_filename` | 预处理后的训练文件。 | `data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz` |
| `adj_filename` | 距离或邻接关系文件。 | `data/PEMS04/distance.csv` |
| `num_of_vertices` | 交通传感器节点数量。 | `307` |
| `points_per_hour` | 每小时采样点数。PEMS04 为 5 分钟一采样，因此是 12。 | `12` |

### `task`

预测任务和时间窗口设置。

| 变量 | 含义 | 建议值 |
| --- | --- | --- |
| `len_input` | 输入历史时间步数量。 | `12` |
| `num_for_predict` | 预测未来时间步数量。 | `12` |
| `num_of_hours` | recent component 使用的小时依赖数量。 | `1` |
| `num_of_days` | daily component 数量，第一阶段不启用。 | `0` |
| `num_of_weeks` | weekly component 数量，第一阶段不启用。 | `0` |
| `in_channels` | 输入特征通道数，第一阶段只使用第 0 个特征。 | `1` |
| `target_channel` | 预测目标特征索引。 | `0` |

### `model`

ASTGCN 模型结构参数。

| 变量 | 含义 | 建议值 |
| --- | --- | --- |
| `model_name` | 模型名称，用于日志和实验目录。 | `astgcn_recent` |
| `nb_block` | ASTGCN block 堆叠数量。 | `2` |
| `K` | Chebyshev 图卷积阶数。 | `3` |
| `nb_chev_filter` | 图卷积输出通道数。 | `64` |
| `nb_time_filter` | 时间卷积输出通道数。 | `64` |
| `time_strides` | 时间卷积步幅。recent component 下通常等于 `num_of_hours`。 | `1` |

### `training`

训练参数。

| 变量 | 含义 | 建议值 |
| --- | --- | --- |
| `batch_size` | 批大小。 | `32` |
| `epochs` | 训练轮数。 | `80` |
| `start_epoch` | 从第几轮开始训练。 | `0` |
| `learning_rate` | Adam 学习率。 | `0.001` |
| `loss_function` | 训练损失函数。 | `mse` |
| `metric_method` | 测试指标是否使用 mask。 | `unmask` |
| `missing_value` | 缺失值标识。 | `0.0` |
| `device` | 训练设备。 | `cuda` 或 `cpu` |
| `seed` | 随机种子。 | `42` |

### `output`

实验输出路径。

| 变量 | 含义 | 建议值 |
| --- | --- | --- |
| `experiment_root` | 实验根目录。 | `experiments` |
| `save_best_only` | 是否只保存验证集最优模型。 | `true` |
| `save_predictions` | 是否保存预测结果。 | `true` |

## 使用原则

- 配置文件中的路径统一相对项目根目录书写。
- 后续新增数据集时，不修改代码主体，只新增配置文件。
- 如果变量含义不清，应先更新本文档，再写代码。
