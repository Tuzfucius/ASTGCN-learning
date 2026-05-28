# Context Pack

> This file is generated automatically for ChatGPT / Codex context.

This pack is intended to be a single text blob that summarizes the repository structure and the public API surface, then embeds the most relevant project docs and config files.

## File: `README.md`

```markdown
# ASTGCN PEMS04 教学复现

本项目是对论文 **Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting** 中 ASTGCN 模型的教学型 PyTorch 实现。当前代码已经形成可运行的软件包结构，覆盖数据处理、图结构构建、三分支 ASTGCN、训练评估、常用 baseline 对比、Kaggle notebook 和可视化输出。

## 项目结构

```text
configs/              PEMS04 训练与对比配置
data/raw/PEMS04/      PEMS04 原始数据和距离文件
docs/                 论文 PDF 等资料
scripts/              训练、推理、baseline 对比和 Kaggle notebook
src/astgcn/data/      数据读取、窗口构造、标准化、DataLoader、图结构
src/astgcn/models/    ASTGCN 注意力、图卷积、STBlock、三分支融合模型
src/astgcn/engine/    训练、评估、预测和 checkpoint
src/astgcn/baselines/ HA、SVR、LSTM、GRU baseline
tests/                shape、baseline、模型前向和数据链路测试
outputs/              运行输出目录，生成产物默认不进入 git
```

## 当前功能状态

- 已实现 ASTGCN recent / daily / weekly 三组件预测与可学习融合。
- 已实现 HA、SVR、LSTM、GRU baseline。
- 已提供统一性能对比脚本，输出 MAE、RMSE、MAPE 表格和可视化图。
- 已提供 Kaggle notebook，支持云端运行、性能对比和单节点预测曲线展示。
- 已补充目录级中文 README，说明各目录职责和常用命令。

论文中的 ARIMA、VAR、STGCN、GLU-STGCN、GeoMAN 暂未在本轮实现，后续可作为扩展 baseline。

## 核心张量约定

原始 PEMS04 数据形状为 `[T, N, F]`，当前数据文件为：

- `T = 16992`
- `N = 307`
- `F = 3`

`Dataset` 输出单样本：

- `recent`: `[N, F, T_h]`
- `daily`: `[N, F, T_d]`
- `weekly`: `[N, F, T_w]`
- `target`: `[N, T_p]`

`DataLoader` 批量后：

- `recent/daily/weekly`: `[B, N, F, T]`
- `target`: `[B, N, T_p]`

模型输出固定为 `[B, N, T_p]`。

## 环境安装

建议使用 conda 环境：

```powershell
conda create -n astgcn python=3.9
conda activate astgcn
pip install -e .
pip install -r requirements.txt
```

## 训练、推理与对比

最小 smoke 训练：

```powershell
python scripts\train.py --config configs\pems04.yaml --epochs 1 --max-batches 1
```

推理评估：

```powershell
python scripts\infer.py --config configs\pems04.yaml --checkpoint outputs\checkpoints\best.pt --save-components
```

baseline 性能对比：

```powershell
python scripts\compare_baselines.py --config configs\pems04.yaml --epochs 1 --max-batches 1 --svr-samples 64 --device cpu
```

对比结果输出到：

- `outputs/comparison/baseline_metrics.csv`
- `outputs/comparison/baseline_metrics.json`
- `outputs/comparison/metrics_bar.png`
- `outputs/comparison/sample_prediction.png`
- `outputs/comparison/baseline_predictions.npz`

## Kaggle Notebook

云端入口：

```text
scripts/kaggle_astgcn_pems04_training.ipynb
```

notebook 会自动定位项目根目录和 PEMS04 数据文件，运行同一套 baseline 对比逻辑，并展示：

- 指标表格
- MAE / RMSE / MAPE 柱状图
- `target / HA / SVR / LSTM / GRU / ASTGCN` 单节点预测曲线

## 验证命令

```powershell
python -m compileall -q src scripts tests
python -m pytest tests -q
python scripts\compare_baselines.py --config configs\pems04.yaml --epochs 1 --max-batches 1 --svr-samples 64 --device cpu
```

当前实现已通过数据窗口、图结构、baseline、三组件 ASTGCN 前向、训练 smoke、推理 smoke、baseline 对比脚本和 notebook 语法校验。

```

## File: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "astgcn"
version = "0.1.0"
description = "ASTGCN research project"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
  "numpy",
  "PyYAML",
  "torch",
  "pandas",
  "matplotlib",
  "tqdm",
  "scikit-learn",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

```

## File: `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
cache_dir = C:/tmp/astgcn-pytest-cache
markers =
    slow: 依赖大体积数据文件或运行时间较长的测试。
    integration: 覆盖跨模块数据流的测试。

```

## File: `configs/pems04.yaml`

```yaml
dataset:
  name: PEMS04
  data_path: data/raw/PEMS04/pems04.npz
  distance_path: data/raw/PEMS04/distance.csv
  num_nodes: 307
  input_dim: 3
  target_dim: 0
  points_per_day: 288

time_window:
  pred_len: 12
  recent_len: 12
  daily_days: 1
  weekly_weeks: 1

split:
  train_ratio: 0.6
  val_ratio: 0.2

graph:
  directed: false
  weighted: false
  cheb_order: 3

model:
  num_blocks: 2
  in_channels: 3
  hidden_channels: 64
  cheb_channels: 64
  time_channels: 64
  time_kernel_size: 3
  dropout: 0.0

ablation:
  use_recent: true
  use_daily: true
  use_weekly: true
  use_temporal_attention: true
  use_spatial_attention: true
  use_graph_conv: true
  use_temporal_conv: true
  graph_mode: cheb
  fusion_mode: matrix

train:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: Adam
  loss: mae
  device: auto
  num_workers: 0
  seed: 42
  early_stop_patience: 10
  grad_clip: 5.0

log:
  save_dir: outputs
  checkpoint_dir: outputs/checkpoints
  log_dir: outputs/logs
  prediction_dir: outputs/predictions

```

## File: `src/README.md`

```markdown
# src 目录说明

本目录是 Python 包源码目录。项目使用 `src` 布局，安装后可通过 `import astgcn` 引入包内模块。

## 职责

- 隔离可复用代码和一次性运行脚本。
- 为数据处理、图构建、模型、训练评估和 baseline 提供统一包结构。
- 让本地脚本、Kaggle notebook 和测试脚本使用同一套实现。

## 安装当前包

```powershell
python -m pip install -e .
```

安装后检查：

```powershell
python -c "import astgcn; print(astgcn.__version__)"
```

## 当前状态

`src/astgcn` 已包含可运行的 ASTGCN 主模型、HA/SVR/LSTM/GRU baseline、训练评估工具、指标函数和数据加载逻辑。脚本层不再维护重复模型实现。

```

## File: `src/astgcn/README.md`

```markdown
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

```

## File: `src/astgcn/baselines/README.md`

```markdown
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

```

## File: `src/astgcn/data/README.md`

```markdown
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

```

## File: `src/astgcn/engine/README.md`

```markdown
# engine 目录说明

本目录保存训练、评估、预测和 checkpoint 管理逻辑。

## 主要模块

- `trainer.py`：训练循环、验证循环、早停和最优 checkpoint 保存。
- `evaluator.py`：在测试集上评估模型，并在反标准化后计算指标。
- `predictor.py`：保存预测值、真实值、时间索引和可选分支预测。
- `checkpoint.py`：保存和加载模型参数、优化器状态、配置和 scaler 参数。

## 使用原则

- 训练阶段可在标准化空间计算 loss。
- 最终 MAE、RMSE、MAPE 必须在反标准化后的真实交通流量尺度上计算。
- 运行产物写入 `outputs/` 下的子目录，不直接提交到版本库。

baseline 对比脚本复用本目录中的训练和评估思想，但为了统一 HA、SVR 和深度模型，也包含了少量脚本级适配逻辑。

```

## File: `src/astgcn/models/README.md`

```markdown
# models 目录说明

本目录实现 ASTGCN 网络结构。ASTGCN 同时建模交通路网的空间依赖和时间依赖，并融合 recent、daily、weekly 三类历史信息。

## 主要模块

- `attention.py`：Temporal Attention、Spatial Attention 和时间注意力应用函数。
- `cheb_conv.py`：Chebyshev 图卷积和带空间注意力的图卷积。
- `temporal_conv.py`：时间卷积。
- `st_block.py`：时空块，组合注意力、图卷积、时间卷积、残差和归一化。
- `component.py`：单个 ASTGCN 分支组件。
- `fusion.py`：recent/daily/weekly 三分支可学习融合。
- `astgcn.py`：顶层 ASTGCN 模型。

## 输入输出

```text
recent/daily/weekly: [B, N, F, T]
component output:    [B, N, T_p]
final output:        [B, N, T_p]
```

`ASTGCN.forward(..., return_components=True)` 会返回 `prediction/recent/daily/weekly`，用于 notebook 和对比脚本可视化。

## 消融配置

`ablation.py` 提供 `AblationConfig`，用于统一管理消融实验开关：

- 时间分支：`use_recent`、`use_daily`、`use_weekly`
- ST-Block：`use_temporal_attention`、`use_spatial_attention`、`use_graph_conv`、`use_temporal_conv`
- 图结构：`graph_mode` 支持 `cheb`、`identity`、`random`、`none`
- 融合方式：`fusion_mode` 支持 `matrix`、`average`、`scalar`、`concat_mlp`

```

## File: `scripts/README.md`

```markdown
# scripts 目录说明

本目录保存项目级运行入口。脚本只负责编排配置、数据、模型、训练、评估和可视化，核心实现放在 `src/astgcn/` 包内。

## 当前脚本

- `train.py`：训练 ASTGCN，并保存最优 checkpoint。
- `infer.py`：加载 checkpoint，在测试集上评估并保存预测结果。
- `compare_baselines.py`：统一比较 HA、SVR、LSTM、GRU、ASTGCN，输出指标表和图像。
- `kaggle_astgcn_pems04_training.ipynb`：面向 Kaggle/Ubuntu 的训练、性能对比与报告分析 notebook。

## 常用命令

```powershell
python scripts\train.py --config configs\pems04.yaml --epochs 1 --max-batches 1
python scripts\infer.py --config configs\pems04.yaml --checkpoint outputs\checkpoints\best.pt --save-components
python scripts\compare_baselines.py --config configs\pems04.yaml --epochs 1 --max-batches 1 --svr-samples 64 --device cpu
```

`compare_baselines.py` 默认用于教学和流程验证。正式实验可增加 `--epochs`、`--max-batches` 和 `--svr-samples`。

## Kaggle 使用方式

1. 在 Kaggle 新建 Notebook，并把本项目代码作为 Dataset 或上传到 `/kaggle/working/ASTGCN`。
2. 挂载包含 `pems04.npz` 和 `distance.csv` 的 PEMS04 数据集。
3. 打开 `kaggle_astgcn_pems04_training.ipynb`。
4. 调试时保持 `RUN_MODE = "quick"`，正式实验时改为 `RUN_MODE = "full"`。

notebook 会运行统一对比脚本，并展示指标表、柱状图、单节点预测曲线、稳定指标、相对提升率、分预测时长误差、节点级误差、高峰/非高峰误差、典型案例和推理流程 shape 表。

# 消融实验 notebook

- `ablation_cloud_training.ipynb`：面向 Kaggle、Colab 或其他云端环境的 ASTGCN 消融实验 notebook。它会读取 `configs/pems04.yaml`，批量覆盖 `ablation` 配置，完成训练、测试集评估、预测结果保存、指标柱状图和单节点预测曲线展示。
- 调试时保持 `RUN_MODE = "quick"`；正式训练时改为 `RUN_MODE = "full"`。
- 如果云端数据目录和项目默认路径不同，可设置环境变量 `DATA_ROOT`，其中应包含 `pems04.npz` 和 `distance.csv`。

```

## File: `scripts/context/README.md`

```markdown
# scripts/context 目录说明

这个目录保存提交前自动生成项目上下文包的脚本。

## 脚本职责

- `generate_file_index.py`：扫描项目目录，生成可靠的文件索引，不解析代码语义。
- `generate_public_api.py`：解析 `src/` 下的 Python 文件，提取公开类和函数签名。
- `build_context_pack.py`：把文件索引、公开 API 和关键文档拼接成单一上下文文本。
- `update_context_pack.py`：给 `pre-commit` 调用的入口，负责生成并把产物重新加入暂存区。

## 产物

- `docs/generated/FILE_INDEX.md`
- `docs/generated/PUBLIC_API.md`
- `docs/generated/CONTEXT_PACK.md`

## 约定

- 这些文件由脚本自动维护，不建议手工编辑。
- 提交前 hook 会自动更新并暂存它们，保证每次 commit 都携带最新上下文。

```

## File: `docs/generated/FILE_INDEX.md`

```markdown
# File Index

> This file is generated automatically. Do not edit manually.

## src

- `src/astgcn/__init__.py`
- `src/astgcn/baselines/__init__.py`
- `src/astgcn/baselines/gru.py`
- `src/astgcn/baselines/historical_average.py`
- `src/astgcn/baselines/lstm.py`
- `src/astgcn/baselines/README.md`
- `src/astgcn/baselines/rnn.py`
- `src/astgcn/baselines/svr.py`
- `src/astgcn/data/__init__.py`
- `src/astgcn/data/dataloader.py`
- `src/astgcn/data/dataset.py`
- `src/astgcn/data/graph.py`
- `src/astgcn/data/io.py`
- `src/astgcn/data/README.md`
- `src/astgcn/data/scaler.py`
- `src/astgcn/data/window.py`
- `src/astgcn/engine/__init__.py`
- `src/astgcn/engine/checkpoint.py`
- `src/astgcn/engine/evaluator.py`
- `src/astgcn/engine/predictor.py`
- `src/astgcn/engine/README.md`
- `src/astgcn/engine/trainer.py`
- `src/astgcn/logger.py`
- `src/astgcn/losses.py`
- `src/astgcn/metrics.py`
- `src/astgcn/models/__init__.py`
- `src/astgcn/models/ablation.py`
- `src/astgcn/models/astgcn.py`
- `src/astgcn/models/attention.py`
- `src/astgcn/models/cheb_conv.py`
- `src/astgcn/models/component.py`
- `src/astgcn/models/fusion.py`
- `src/astgcn/models/README.md`
- `src/astgcn/models/st_block.py`
- `src/astgcn/models/temporal_conv.py`
- `src/astgcn/README.md`
- `src/astgcn/utils.py`
- `src/README.md`

## tests

- `tests/check_rnn_baseline.py`
- `tests/conftest.py`
- `tests/README.md`
- `tests/test_attention.py`
- `tests/test_baselines.py`
- `tests/test_dataloader.py`
- `tests/test_dataset.py`
- `tests/test_io.py`
- `tests/test_model_forward.py`
- `tests/test_real_dataset.py`
- `tests/test_window_graph.py`

## configs

- `configs/pems04.yaml`
- `configs/README.md`

## scripts

- `scripts/compare_baselines.py`
- `scripts/context/build_context_pack.py`
- `scripts/context/generate_file_index.py`
- `scripts/context/generate_public_api.py`
- `scripts/context/README.md`
- `scripts/context/update_context_pack.py`
- `scripts/infer.py`
- `scripts/README.md`
- `scripts/train.py`


```

## File: `docs/generated/PUBLIC_API.md`

```markdown
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


```

