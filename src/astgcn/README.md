# astgcn 包说明

`astgcn` 是项目核心包，按模块复现 PEMS04 交通流预测流程。

## 子目录职责

- `data/`：读取 PEMS04 数据，构造 recent/daily/weekly 时间片段，标准化数据，构建 DataLoader 和图结构。
- `models/`：实现 ASTGCN 注意力层、Chebyshev 图卷积、STBlock、三分支组件和融合层。
- `engine/`：实现训练、验证、测试、预测保存和 checkpoint 管理。
- `baselines/`：实现 HA、SVR、LSTM、GRU 对比模型。

## 顶层模块

- `utils.py`：配置读取、目录创建、随机种子和设备选择。
- `metrics.py`：MAE、RMSE、MAPE 及 mask 版本。
- `losses.py`：训练损失选择器。
- `logger.py`：控制台和文件日志工具。

## 统一接口

深度模型和 HA baseline 的前向接口统一为：

```python
model(recent, daily, weekly, return_components=False)
```

其中 `recent/daily/weekly` 的形状为 `[B, N, F, T]`，输出为 `[B, N, T_p]`。

`SVRBaseline` 是 scikit-learn 适配器，使用：

```python
model.fit_loader(train_loader)
prediction = model.predict_batch(batch)
```
