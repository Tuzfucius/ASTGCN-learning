# configs 目录说明

本目录保存实验配置文件。配置文件用于集中管理数据路径、时间窗口、图结构、模型超参数、训练参数和输出目录，便于复现实验和对比不同设置。

## 主要文件

- `pems04.yaml`：PEMS04 数据集的默认配置。

## 配置结构

- `dataset`：数据集名称、时序数据路径、距离文件路径、节点数、特征维度、预测目标维度和一天采样点数。
- `time_window`：预测长度，以及 recent、daily、weekly 三类历史片段的截取长度。
- `split`：训练集、验证集和测试集的时间顺序切分比例。
- `graph`：图是否有向、是否使用边权、Chebyshev 多项式阶数。
- `model`：ASTGCN 或实验模型的隐藏维度、块数、dropout 等参数。
- `train`：batch size、epoch、学习率、权重衰减、设备、随机种子和早停设置。
- `log`：输出、checkpoint、日志和预测结果保存目录。

## 常用命令

```bash
python -m pip install -e .
python tests/check6_dataloader.py
```

在 Kaggle notebook 中会读取 `configs/pems04.yaml`，然后根据 Kaggle 文件系统自动修正 `data_path` 和 `distance_path`。
