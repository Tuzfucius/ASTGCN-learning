# `src/engine` 训练评估模块说明

本目录负责训练流程和评估流程。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `trainer.py` | 训练、验证和模型保存。 |
| `evaluator.py` | 测试推理、指标计算和预测结果保存。 |

## `trainer.py`

建议类：

```python
Trainer
```

建议方法：

| 方法 | 职责 |
| --- | --- |
| `fit` | 完整训练流程。 |
| `train_one_epoch` | 单轮训练。 |
| `validate` | 验证集评估。 |
| `save_checkpoint` | 保存模型权重。 |

`Trainer` 输入：

- 模型。
- 优化器。
- 损失函数。
- train DataLoader。
- val DataLoader。
- 训练配置。
- 输出目录。

## `evaluator.py`

建议类：

```python
Evaluator
```

建议方法：

| 方法 | 职责 |
| --- | --- |
| `predict` | 对 DataLoader 执行推理。 |
| `evaluate` | 计算测试指标。 |
| `save_predictions` | 保存预测值和真实值。 |

## 训练数据形状

```text
batch_x: (B, N, F, T)
batch_y: (B, N, T_pred)
pred_y: (B, N, T_pred)
```

## 训练流程

```text
model.train()
  -> forward
  -> loss
  -> backward
  -> optimizer.step()
```

## 验证流程

```text
model.eval()
  -> no_grad
  -> forward
  -> val_loss
  -> update best checkpoint
```

## 评估流程

```text
load best checkpoint
  -> predict test set
  -> calculate metrics on original target scale
  -> save predictions
```

## 注意事项

- `Trainer` 不应读取原始数据。
- `Evaluator` 不应修改模型参数。
- 验证和评估阶段必须使用 `torch.no_grad()`。
- 保存权重时记录对应 epoch 和验证 loss。
- 当前阶段只标准化输入 `x`，`target` 保持原始数值；训练 loss、验证 loss 和评估指标不做预测反标准化。
