# baselines 目录说明

本目录保存对比模型。baseline 用于判断 ASTGCN 是否优于简单统计方法、传统机器学习方法和常见深度学习序列模型。

## 当前实现

- `historical_average.py`：Historical Average，使用 recent 输入中目标特征的历史均值预测未来。
- `svr.py`：SVR，按节点级样本展开 recent 特征，并使用多输出 SVR 预测未来窗口。
- `lstm.py`：LSTM，逐节点共享参数的序列模型。
- `gru.py`：GRU，逐节点共享参数的序列模型。

所有可直接前向的 baseline 均接收 `recent [B, N, F, T]`，输出 `[B, N, T_p]`。`SVRBaseline` 不是 `nn.Module`，需要先调用 `fit_loader()`，再调用 `predict_batch()`。

## 当前对比范围

本轮实现的性能对比范围为：

```text
HA / SVR / LSTM / GRU / ASTGCN
```

论文中还出现 ARIMA、VAR、STGCN、GLU-STGCN、GeoMAN 等模型。这些模型暂未实现，后续可作为扩展工作加入。

## 评估原则

- 所有模型使用相同数据切分、预测长度和指标。
- 最终 MAE、RMSE、MAPE 均在反标准化后的真实交通流量尺度上计算。
- SVR 默认使用抽样训练，避免 PEMS04 全节点全样本展开导致训练过慢。
