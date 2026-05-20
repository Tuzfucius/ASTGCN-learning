# outputs 目录说明

本目录用于保存实验输出，不建议提交大体积模型权重、预测数组或图片结果到版本库，除非它们是报告必须引用的小文件。

## 建议子目录

- `checkpoints/`：保存训练 checkpoint。
- `logs/`：保存训练日志。
- `predictions/`：保存预测值、真实值和指标文件。
- `figures/`：保存可视化图片。

## Kaggle notebook 输出

`scripts/kaggle_astgcn_pems04_training.ipynb` 默认会写入：

```text
outputs/checkpoints/kaggle_astgcn_best.pt
outputs/predictions/kaggle_astgcn_predictions.npz
outputs/figures/
```

这些文件用于复盘训练过程、绘制预测曲线和分析 recent/daily/weekly 融合权重。
