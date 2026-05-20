# outputs 目录说明

本目录保存实验输出。大体积模型权重、预测数组、日志和图像默认不提交到版本库。

## 子目录

- `checkpoints/`：训练 checkpoint。
- `logs/`：训练、推理和对比脚本日志。
- `predictions/`：推理预测结果。
- `figures/`：普通可视化图像。
- `comparison/`：baseline 性能对比结果。

## baseline 对比输出

运行：

```powershell
python scripts\compare_baselines.py --config configs\pems04.yaml --epochs 1 --max-batches 1 --svr-samples 64 --device cpu
```

会生成：

```text
outputs/comparison/baseline_metrics.csv
outputs/comparison/baseline_metrics.json
outputs/comparison/metrics_bar.png
outputs/comparison/sample_prediction.png
outputs/comparison/baseline_predictions.npz
```

其中 `baseline_metrics.csv` 适合写入实验报告，`sample_prediction.png` 用于直观比较不同模型在同一节点上的预测曲线。
