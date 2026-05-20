# scripts 目录说明

本目录保存项目级运行入口。脚本只负责编排配置、数据、模型、训练、评估和可视化，核心实现放在 `src/astgcn/` 包内。

## 当前脚本

- `train.py`：训练 ASTGCN，并保存最优 checkpoint。
- `infer.py`：加载 checkpoint，在测试集上评估并保存预测结果。
- `compare_baselines.py`：统一比较 HA、SVR、LSTM、GRU、ASTGCN，输出指标表和图像。
- `kaggle_astgcn_pems04_training.ipynb`：面向 Kaggle/Ubuntu 的训练与性能对比 notebook。

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

notebook 会运行统一对比脚本，并展示指标表、柱状图和单节点预测曲线。
