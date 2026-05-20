# scripts 目录说明

本目录用于放置项目级运行脚本和实验 notebook。脚本应只负责串联配置、数据、模型、训练和评估流程，具体实现优先放在 `src/astgcn/` 包内，避免把核心逻辑堆在单个脚本中。

## 当前文件

- `train.py`：预留的训练入口。
- `infer.py`：预留的推理入口。
- `kaggle_astgcn_pems04_training.ipynb`：面向 Kaggle/Ubuntu 的 PEMS04 训练 notebook，包含项目根目录和数据目录自动定位、包安装、短训练/正式训练切换、评估、预测可视化与融合权重可视化。

## Kaggle notebook 使用方式

1. 在 Kaggle 新建 Notebook，并把本项目代码作为 Dataset 或直接上传到 `/kaggle/working/ASTGCN`。
2. 把 PEMS04 数据作为 Kaggle Dataset 挂载，确保其中包含 `pems04.npz` 和 `distance.csv`。
3. 打开 `kaggle_astgcn_pems04_training.ipynb`，按顺序执行全部单元。
4. 调试时保持 `RUN_MODE = "quick"`；正式训练时改为 `RUN_MODE = "full"`。

## 本地常用命令

```bash
python -m pip install -e .
python tests/check1_window_graph.py
python tests/check6_dataloader.py
```

在 Windows 本地使用 conda 时，建议先进入项目根目录：

```powershell
conda create -n astgcn python=3.10 -y
conda activate astgcn
python -m pip install -e .
```
