# tests 目录说明

本目录保存检查脚本和 pytest 测试。测试重点是验证数据 shape、模型前向、baseline 接口和脚本依赖是否稳定。

## 检查脚本

- `check1_window_graph.py`：检查时间窗口和图构建基础逻辑。
- `check2_graph.py`：检查图结构矩阵形状。
- `check3_dataset.py`：检查 Dataset 输出。
- `check4_io.py`：检查 `.npz` 数据读取。
- `check5_real_dataset.py`：检查真实 PEMS04 数据集。
- `check6_dataloader.py`：检查 DataLoader、切分和 scaler。
- `check7_attention.py`：检查注意力模块输出形状。

## pytest 测试

- `test_model_forward.py`：验证 ASTGCN 三组件前向输出 `[B, N, T_p]`。
- `test_baselines.py`：验证 HA、SVR、LSTM、GRU baseline shape 和最小训练预测。

## 常用命令

```powershell
python -m compileall -q src scripts tests
python -m pytest tests -q
python tests\check6_dataloader.py
```

ASTGCN 项目最常见的问题是维度顺序错误。先通过 shape 测试，再进入训练和性能对比。
