# baselines 目录说明

本目录用于放置对比方法。baseline 的作用是判断 ASTGCN 是否确实优于简单统计方法、传统机器学习方法或常见深度学习序列模型。

## 当前文件

- `historical_average.py`：预留 Historical Average 方法。
- `lstm.py`：预留 LSTM baseline。

## 建议补充的 baseline

- Historical Average：使用历史相同时段平均值作为预测。
- Last Value：直接复制最近一个观测值，作为很强的简单基线。
- LSTM/GRU：按节点或全局序列建模。
- SVR/RandomForest：可在抽样节点或聚合特征上做传统机器学习对比。

## 实验原则

baseline 与 ASTGCN 应使用相同的数据切分、预测长度和评估指标。评估时同样要先反标准化，再计算 MAE、RMSE 和 MAPE。
