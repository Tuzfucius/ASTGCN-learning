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
