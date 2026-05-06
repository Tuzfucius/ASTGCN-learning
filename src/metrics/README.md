# `src/metrics` 指标模块说明

本目录负责训练损失和测试指标。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `losses.py` | PyTorch 训练损失函数。 |
| `evaluation.py` | 测试评估指标和汇总函数。 |

## 训练损失

`losses.py` 建议实现：

| 函数 | 说明 |
| --- | --- |
| `masked_mae` | 忽略缺失值的 MAE。 |
| `masked_mse` | 忽略缺失值的 MSE。 |
| `masked_rmse` | 忽略缺失值的 RMSE。 |
| `get_loss_function` | 根据配置返回损失函数。 |

输入形状：

```text
preds: (B, N, T_pred)
labels: (B, N, T_pred)
```

## 测试指标

`evaluation.py` 建议实现：

| 函数 | 说明 |
| --- | --- |
| `mae` | 平均绝对误差。 |
| `mse` | 均方误差。 |
| `rmse` | 均方根误差。 |
| `mape` | 平均绝对百分比误差。 |
| `evaluate_prediction` | 汇总多个指标。 |

## 变量说明

| 变量 | 含义 |
| --- | --- |
| `preds` | 模型预测值。 |
| `labels` | 真实目标值。 |
| `missing_value` | 缺失值标识。 |
| `mask` | 有效位置掩码。 |
| `metric_method` | 是否使用 mask 计算指标。 |

## masked 指标逻辑

masked 指标用于忽略缺失值。

一般逻辑：

```text
mask = labels != missing_value
loss = loss * mask
loss = mean(loss)
```

注意：

- mask 应转换为浮点数。
- 可以对 mask 做归一化，避免有效值数量影响尺度。
- 评估阶段和训练阶段应使用一致的缺失值定义。

## 输出建议

评估阶段至少输出：

```text
MAE
RMSE
MAPE
```

可以额外输出：

- 按预测步长统计的指标。
- 按样本统计的指标。
- 整体平均指标。
