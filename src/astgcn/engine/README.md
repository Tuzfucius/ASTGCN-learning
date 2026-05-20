# engine 目录说明

本目录用于放置训练、验证、评估、预测和 checkpoint 管理代码。它连接 `data/` 和 `models/`，但不应承担数据窗口或网络层的具体实现。

## 主要文件

- `trainer.py`：预留训练循环，包括 train one epoch、validate、early stopping 和日志记录。
- `evaluator.py`：预留测试集评估逻辑。
- `predictor.py`：预留推理与预测结果保存逻辑。
- `checkpoint.py`：预留模型权重、优化器状态、配置和 scaler 的保存加载逻辑。

## 推荐训练流程

```text
读取配置
构建 DataLoader 和 scaler
构建图结构和模型
循环训练并在验证集评估
保存验证集最优 checkpoint
在测试集反标准化后计算指标
保存预测结果和图表
```

## 指标要求

最终 MAE、RMSE、MAPE 应在反标准化后的真实交通流量尺度上计算。标准化空间中的指标只能用于调试，不适合作为最终报告结果。
