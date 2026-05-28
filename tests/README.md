# tests 目录说明

本目录保存 `pytest` 测试用例和测试辅助配置，目标是把原先一次性执行的检查脚本改成可自动发现、可批量运行的测试套件。

## 测试组织

- `conftest.py`：提供公共 fixture，并把 `src/` 加入 `sys.path`，保证在仓库根目录直接运行 `pytest` 时可以正常导入 `astgcn`。
- `test_window_graph.py`：覆盖时间窗口切片、`t0` 构造以及图结构构造。
- `test_dataset.py`：覆盖 `ASTGCNDataset` 和 `DataLoader` 的基础批次形状。
- `test_io.py`：覆盖 `pems04.npz` 读取。
- `test_real_dataset.py`：覆盖真实 PEMS04 数据、`StandardScaler` 和时间切分逻辑。
- `test_dataloader.py`：覆盖完整数据加载流水线。
- `test_attention.py`：覆盖时间注意力和空间注意力模块。
- `test_baselines.py`：覆盖基线模型输出形状和最小训练/预测流程。
- `test_model_forward.py`：覆盖 ASTGCN 主模型前向输出。

## 运行方式

```powershell
python -m pytest tests -q
python -m pytest tests -q -m "not slow"
```

## 说明

- 带 `slow` 标记的测试依赖较大的真实数据文件，适合本地或 CI 的完整回归。
- 带 `integration` 标记的测试覆盖跨模块数据流，主要用于验证数据加载链路是否保持一致。
