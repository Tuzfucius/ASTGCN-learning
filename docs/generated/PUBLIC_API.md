# Public API

> This file is generated automatically. Do not edit manually.

## `src/astgcn/baselines/gru.py`

### class `GRUBaseline`

逐节点共享参数的 GRU 预测基线。

## `src/astgcn/baselines/historical_average.py`

### class `HistoricalAverage`

使用历史输入窗口的目标特征均值作为未来预测。

## `src/astgcn/baselines/lstm.py`

### class `LSTMBaseline`

逐节点共享参数的 LSTM 预测基线。

## `src/astgcn/baselines/rnn.py`

### class `RNNBaseline`

逐节点共享参数的 RNN 预测基线。

## `src/astgcn/baselines/svr.py`

### class `SVRBaseline`

基于节点级样本展开的 SVR 基线。

## `src/astgcn/data/dataloader.py`

### function `build_dataloaders(data_path: str, num_recent: int, num_days: int, num_weeks: int, pred_len: int, batch_size: int, points_per_day: int = ..., target_dim: int = ..., train_ratio: float = ..., val_ratio: float = ..., num_workers: int = ...)`

构建训练、验证和测试 DataLoader。

## `src/astgcn/data/dataset.py`

### class `ASTGCNDataset`

ASTGCN 三组件数据集。

## `src/astgcn/data/graph.py`

### function `get_distance_matrix(file_path: str | Path, num_nodes: int | None = ..., directed: bool = ...)`

从 ``distance.csv`` 构造距离矩阵。

### function `get_adjacency_matrix(distance_matrix: np.ndarray, weighted: bool = ...)`

根据距离矩阵构造邻接矩阵。

### function `get_normalized_laplacian(adj_matrix: np.ndarray)`

计算归一化拉普拉斯矩阵 ``L = I - D^{-1/2} A D^{-1/2}``。

### function `get_scaled_laplacian(laplacian: np.ndarray)`

计算缩放拉普拉斯矩阵 ``L_tilde = 2L / lambda_max - I``。

### function `get_chebyshev_polynomials(scaled_laplacian: np.ndarray, k_order: int)`

生成 Chebyshev 多项式矩阵。

### function `build_graph_data(file_path: str | Path, k_order: int, num_nodes: int | None = ..., directed: bool = ..., weighted: bool = ...)`

一次性构造 ASTGCN 所需图数据。

## `src/astgcn/data/io.py`

### function `load_pems_npz(file_path: str | Path)`

读取 PEMS 时序数据。

## `src/astgcn/data/scaler.py`

### class `StandardScaler`

按特征维度做标准化。

## `src/astgcn/data/window.py`

### function `get_segment_data(data: np.ndarray, start: int, end: int)`

获取连续时间片段。

### function `get_recent_data(data: np.ndarray, t0: int, num_recent: int)`

获取当前时间点之前的近期片段。

### function `get_days_data(data: np.ndarray, t0: int, num_days: int, pred_len: int, points_per_day: int = ...)`

获取过去若干天相同时间段的数据。

### function `get_weeks_data(data: np.ndarray, t0: int, num_weeks: int, pred_len: int, points_per_day: int = ...)`

获取过去若干周相同星期和时间段的数据。

### function `get_target_data(data: np.ndarray, t0: int, pred_len: int, target_dim: int = ...)`

获取未来预测目标。

### function `build_t0_list(num_samples: int, num_recent: int, num_days: int, num_weeks: int, pred_len: int, points_per_day: int = ...)`

构造所有合法的当前时间步 ``t0``。

### function `split_t0_list(t0_list: np.ndarray, train_ratio: float = ..., val_ratio: float = ...)`

按时间顺序划分训练、验证和测试 ``t0``。

## `src/astgcn/engine/checkpoint.py`

### function `save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = ..., epoch: int = ..., best_metric: float | None = ..., config: Dict[str, Any] | None = ..., scaler: Any | None = ..., **extra: Any)`

保存训练 checkpoint。

### function `load_checkpoint(path: str | Path, model: torch.nn.Module | None = ..., optimizer: torch.optim.Optimizer | None = ..., map_location: str | torch.device | None = ..., strict: bool = ...)`

加载训练 checkpoint。

## `src/astgcn/engine/evaluator.py`

### class `Evaluator`

在反标准化尺度上评估模型。

## `src/astgcn/engine/predictor.py`

### class `Predictor`

生成预测结果并保存为 npz 文件。

## `src/astgcn/engine/trainer.py`

### class `Trainer`

ASTGCN 训练器。

## `src/astgcn/logger.py`

### function `get_logger(name: str = ..., log_file: str | Path | None = ..., level: int = ...)`

创建同时输出到控制台和文件的 logger。

## `src/astgcn/losses.py`

### class `MaskedLoss`

将 masked 指标包装为 PyTorch 损失模块。

### function `get_loss(name: str, mask_value: float = ...)`

按名称创建损失函数。

### class `RMSELoss`

均方根误差损失。

## `src/astgcn/metrics.py`

### function `mae(pred: ArrayLike, target: ArrayLike)`

计算平均绝对误差。

### function `rmse(pred: ArrayLike, target: ArrayLike)`

计算均方根误差。

### function `mape(pred: ArrayLike, target: ArrayLike, eps: float = ...)`

计算平均绝对百分比误差。

### function `masked_mae(pred: ArrayLike, target: ArrayLike, mask_value: float = ...)`

按掩码计算平均绝对误差。

### function `masked_rmse(pred: ArrayLike, target: ArrayLike, mask_value: float = ...)`

按掩码计算均方根误差。

### function `masked_mape(pred: ArrayLike, target: ArrayLike, mask_value: float = ..., eps: float = ...)`

按掩码计算平均绝对百分比误差。

### function `get_metric(name: str)`

按名称获取指标函数。

### function `compute_metrics(pred: ArrayLike, target: ArrayLike, mask_value: float | None = ...)`

一次性计算 MAE、RMSE、MAPE。

## `src/astgcn/models/ablation.py`

### class `AblationConfig`

ASTGCN 消融实验配置。

## `src/astgcn/models/astgcn.py`

### class `ASTGCN`

论文式三组件 ASTGCN 模型。

## `src/astgcn/models/attention.py`

### class `TemporalAttention`

时间注意力层。

### class `SpatialAttention`

空间注意力层。

### function `apply_temporal_attention(x: torch.Tensor, temporal_attention: torch.Tensor)`

将时间注意力作用到输入张量。

## `src/astgcn/models/cheb_conv.py`

### class `ChebGraphConv`

Chebyshev 图卷积层。

### class `ChebGraphConvWithSAtt`

带空间注意力的 Chebyshev 图卷积层。

## `src/astgcn/models/component.py`

### class `ASTGCNComponent`

ASTGCN 单个时间依赖组件。

## `src/astgcn/models/fusion.py`

### class `FusionLayer`

ASTGCN 三组件融合层。

## `src/astgcn/models/st_block.py`

### class `STBlock`

ASTGCN 时空块。

## `src/astgcn/models/temporal_conv.py`

### class `TemporalConv`

沿时间维度执行二维卷积的时间卷积层。

## `src/astgcn/utils.py`

### function `load_config(path: str | Path)`

读取 YAML 配置文件。

### function `ensure_dir(path: str | Path)`

确保目录存在并返回 Path 对象。

### function `set_random_seed(seed: int)`

固定 Python、NumPy 和 PyTorch 的随机种子。

### function `select_device(device: str = ...)`

选择训练设备。

