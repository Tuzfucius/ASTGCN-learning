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
