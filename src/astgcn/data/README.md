# data 目录说明

本目录负责 PEMS04 数据读取、时间窗口构造、标准化、DataLoader 和图结构处理。

## 主要模块

- `io.py`：读取 `pems04.npz`，返回 `[T, N, F]`。
- `window.py`：构造 recent、daily、weekly 和 target 时间片段。
- `dataset.py`：封装 `ASTGCNDataset`，单样本输出 `[N, F, T]` 输入和 `[N, T_p]` 目标。
- `dataloader.py`：构建 train/val/test DataLoader，并只用训练时间段 fit scaler。
- `scaler.py`：标准化和目标通道反标准化。
- `graph.py`：根据 `distance.csv` 构造邻接矩阵、拉普拉斯矩阵和 Chebyshev 多项式。

## 当前 shape

```text
raw data: [T, N, F]
single sample recent/daily/weekly: [N, F, T]
batch recent/daily/weekly: [B, N, F, T]
target: [B, N, T_p]
```

最终评估必须先反标准化目标通道，再计算 MAE、RMSE、MAPE。
