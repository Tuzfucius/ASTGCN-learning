# ASTGCN PEMS04 教学复现

本项目是对论文 **Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting** 中 ASTGCN 模型的教学型 PyTorch 实现。代码按软件包方式组织，重点是把数据窗口、图结构、模型组件、训练评估和云端 notebook 分开，便于阅读、调试和课程报告撰写。

## 项目结构

```text
configs/              训练配置
data/raw/PEMS04/      PEMS04 原始数据和距离文件
docs/                 论文 PDF 等资料
scripts/              本地训练、推理和 Kaggle notebook
src/astgcn/data/      数据读取、窗口构造、标准化、图结构
src/astgcn/models/    ASTGCN 注意力、图卷积、STBlock、三分支融合模型
src/astgcn/engine/    训练、评估、预测和 checkpoint
src/astgcn/baselines/ HA 与 LSTM baseline
tests/                shape、数据链路和模型前向测试
outputs/              运行输出目录，生成产物不进入 git
```

每个主要目录下都有 `README.md`，说明该目录的职责和常用入口。

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

## 本地环境

建议使用 conda 环境：

```powershell
conda create -n astgcn python=3.9
conda activate astgcn
pip install -e .
pip install -r requirements.txt
```

## 训练与推理

最小 smoke 训练：

```powershell
python scripts\train.py --config configs\pems04.yaml --epochs 1 --max-batches 1
```

完整训练：

```powershell
python scripts\train.py --config configs\pems04.yaml
```

推理评估：

```powershell
python scripts\infer.py --config configs\pems04.yaml --checkpoint outputs\checkpoints\best.pt --save-components
```

生成的 checkpoint、日志和预测结果分别写入：

- `outputs/checkpoints/`
- `outputs/logs/`
- `outputs/predictions/`

这些运行产物已在 `.gitignore` 中忽略。

## Kaggle 训练

云端训练入口为：

```text
scripts/kaggle_astgcn_pems04_training.ipynb
```

notebook 面向 Kaggle/Ubuntu 环境，会自动定位项目根目录和 PEMS04 数据目录，并复用当前 `src/astgcn` 软件包完成训练、评估和分支预测可视化。

## 验证

常用验证命令：

```powershell
python -m compileall -q src scripts tests
python -m pytest tests -q
python tests\check1_window_graph.py
python tests\check2_graph.py
python tests\check3_dataset.py
python tests\check4_io.py
python tests\check5_real_dataset.py
python tests\check6_dataloader.py
python tests\check7_attention.py
```

当前实现已覆盖数据窗口、图结构、三组件 ASTGCN 前向、baseline shape、训练 smoke、推理 smoke 和 notebook JSON/代码单元语法校验。
