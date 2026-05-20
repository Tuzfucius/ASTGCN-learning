# configs 目录说明

本目录保存实验配置文件。配置文件集中管理数据路径、时间窗口、图结构、模型超参数、训练参数和输出目录。

## 主要文件

- `pems04.yaml`：PEMS04 默认配置。

## 配置结构

- `dataset`：数据路径、距离文件、节点数、特征数、目标通道和每天采样点数。
- `time_window`：预测长度，以及 recent、daily、weekly 三类历史片段设置。
- `split`：训练集、验证集和测试集时间顺序切分比例。
- `graph`：图是否有向、是否使用边权、Chebyshev 阶数。
- `model`：ASTGCN 隐藏维度、块数和时间卷积核等参数。
- `train`：batch size、epoch、学习率、设备、随机种子和早停设置。
- `log`：checkpoint、日志、预测结果保存目录。

## 使用场景

训练、推理、baseline 对比和 Kaggle notebook 都读取 `pems04.yaml`。在 Kaggle 中，notebook 会基于该配置生成临时配置，并自动修正数据路径。
