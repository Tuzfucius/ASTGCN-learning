# astgcn 包说明

`astgcn` 是项目的核心包，目标是按模块复现 ASTGCN 在 PEMS04 交通预测任务中的主要流程。

## 子目录职责

- `data/`：读取 PEMS04 数据、构造 recent/daily/weekly 时间片段、标准化数据、构建 DataLoader 和图结构。
- `models/`：放置 ASTGCN 相关网络层、注意力模块、时空块、三分支组件和融合层。
- `engine/`：放置训练、验证、测试、预测和 checkpoint 管理逻辑。
- `baselines/`：放置 Historical Average、LSTM 等对比模型。

## 顶层模块

- `utils.py`：配置读取、目录创建、随机种子和设备选择等通用函数。
- `metrics.py`：预留指标函数，如 MAE、RMSE、MAPE。
- `losses.py`：预留损失函数封装。
- `logger.py`：预留日志工具。

## 维度约定

当前数据集样本返回：

```text
recent: [T, N, F]
daily:  [T, N, F]
weekly: [T, N, F]
target: [N, Tp]
```

加入 batch 后：

```text
recent: [B, T, N, F]
target: [B, N, Tp]
```

模型内部如果需要论文常用的 `[B, N, F, T]`，应在 `forward()` 中显式转换，并在注释或文档中写清楚。
