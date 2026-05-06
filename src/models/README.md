# `src/models` 模型模块说明

本目录保存 ASTGCN 模型结构。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `layers.py` | 模型基础层和 ASTGCN block。 |
| `astgcn.py` | 完整 ASTGCN 模型和模型构造函数。 |

## 张量约定

模型输入：

```text
x: (B, N, F, T)
```

模型输出：

```text
y_hat: (B, N, T_pred)
```

PEMS04 第一阶段：

```text
x: (B, 307, 1, 12)
y_hat: (B, 307, 12)
```

## `layers.py`

建议类：

| 类 | 职责 |
| --- | --- |
| `TemporalAttention` | 计算时间注意力矩阵。 |
| `SpatialAttention` | 计算空间注意力矩阵。 |
| `ChebGraphConvWithAttention` | 带空间注意力的 Chebyshev 图卷积。 |
| `ASTGCNBlock` | ASTGCN 基础块。 |

## `astgcn.py`

建议类和函数：

| 名称 | 职责 |
| --- | --- |
| `ASTGCN` | 完整模型。 |
| `build_astgcn_model` | 根据配置和图结构构造模型。 |

## 模型参数

| 参数 | 含义 |
| --- | --- |
| `nb_block` | ASTGCN block 堆叠数量。 |
| `in_channels` | 输入特征数。 |
| `K` | Chebyshev 阶数。 |
| `nb_chev_filter` | 图卷积输出通道数。 |
| `nb_time_filter` | 时间卷积输出通道数。 |
| `time_strides` | 时间卷积步幅。 |
| `num_for_predict` | 预测步数。 |
| `len_input` | 输入步数。 |
| `num_of_vertices` | 节点数。 |

## 模型模块边界

模型层可以接收：

- 输入张量。
- Chebyshev 多项式。
- 模型结构参数。

模型层不应处理：

- 配置文件解析。
- 数据文件读取。
- DataLoader 构造。
- 训练循环。
- 实验目录创建。

## 实现建议

- 每个类在注释中写清楚输入输出形状。
- 先用随机张量测试前向传播。
- 不要一开始加入 daily 和 weekly 分支。
- 不要为了简化而改变 ASTGCN block 的核心组成。
