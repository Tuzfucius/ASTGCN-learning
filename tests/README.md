# tests 目录说明

本目录保存项目的检查脚本。当前测试以教学和逐步验证为主，适合在实现 ASTGCN 各模块时逐项排查数据形状和核心逻辑。

## 当前检查脚本

- `check1_window_graph.py`：检查时间窗口和图构建基础逻辑。
- `check2_graph.py`：检查图结构相关函数。
- `check3_dataset.py`：检查 Dataset 输出。
- `check4_io.py`：检查 `.npz` 数据读取。
- `check5_real_dataset.py`：检查真实 PEMS04 数据集。
- `check6_dataloader.py`：检查 DataLoader、切分和 scaler。
- `check7_attention.py`：检查时间注意力模块输出形状。

## 常用命令

```bash
python tests/check1_window_graph.py
python tests/check4_io.py
python tests/check6_dataloader.py
python tests/check7_attention.py
```

建议每完成一个模块就运行对应检查脚本。ASTGCN 项目中最常见的问题是维度顺序错误，先通过 shape 检查再进入训练。
