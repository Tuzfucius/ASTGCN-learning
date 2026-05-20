# data 目录说明

本目录负责 PEMS04 数据读取、时间窗口构造、标准化、DataLoader 和图结构处理，是 ASTGCN 实验最容易出错的部分。

## 主要文件

- `io.py`：读取 `.npz` 时序数据，期望输出形状为 `[T, N, F]`。
- `window.py`：根据时间点 `t0` 构造 recent、daily、weekly 和 target 片段。
- `dataset.py`：把窗口逻辑封装为 PyTorch `Dataset`。
- `dataloader.py`：按时间顺序切分 train/val/test，并返回 DataLoader 与 scaler。
- `scaler.py`：标准化与目标维度反标准化。
- `graph.py`：读取 `distance.csv`，构造邻接矩阵、归一化拉普拉斯矩阵和 Chebyshev 多项式。

## 数据文件要求

PEMS04 原始数据默认放在：

```text
data/raw/PEMS04/pems04.npz
data/raw/PEMS04/distance.csv
```

其中 `pems04.npz` 应包含三维数组 `[时间步, 节点数, 特征数]`，`distance.csv` 应包含 `from,to,cost` 三列。

## 常用检查命令

```bash
python tests/check4_io.py
python tests/check5_real_dataset.py
python tests/check6_dataloader.py
```

标准化只能使用训练时间段拟合，不能用全量数据拟合，否则会产生数据泄漏。
