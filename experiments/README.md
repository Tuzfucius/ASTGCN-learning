# 实验输出目录说明

本目录保存训练、验证和测试产生的实验结果。

实验输出通常不提交到 git，除非是很小的说明文件。

## 推荐结构

```text
experiments/
  PEMS04/
    astgcn_recent/
      config.yaml
      best.pt
      metrics.json
      predictions.npz
      train.log
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `config.yaml` | 本次实验使用的配置副本。 |
| `best.pt` | 验证集损失最优的模型权重。 |
| `metrics.json` | 测试集指标。 |
| `predictions.npz` | 测试集预测值和真实值。 |
| `train.log` | 训练过程日志。 |

## 实验命名建议

建议按数据集和模型命名：

```text
experiments/{dataset_name}/{model_name}/
```

示例：

```text
experiments/PEMS04/astgcn_recent/
```

## 需要记录的信息

每次实验至少记录：

- 数据集名称。
- 输入长度。
- 预测长度。
- 模型参数。
- 训练参数。
- 最优 epoch。
- 测试 MAE、RMSE、MAPE。

## 注意事项

- 不要把大型模型权重提交到 git。
- 如果要对比实验结果，建议单独写 Markdown 表格或 JSON 摘要。
