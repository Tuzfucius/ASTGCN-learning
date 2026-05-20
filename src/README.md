# src 目录说明

本目录是 Python 包源码目录。项目使用 `src` 布局，安装后可通过 `import astgcn` 引入包内模块。

## 职责

- 隔离可复用代码和一次性运行脚本。
- 为数据处理、图构建、模型、训练评估和 baseline 提供统一包结构。
- 让本地脚本、Kaggle notebook 和测试脚本使用同一套实现。

## 安装当前包

```powershell
python -m pip install -e .
```

安装后检查：

```powershell
python -c "import astgcn; print(astgcn.__version__)"
```

## 当前状态

`src/astgcn` 已包含可运行的 ASTGCN 主模型、HA/SVR/LSTM/GRU baseline、训练评估工具、指标函数和数据加载逻辑。脚本层不再维护重复模型实现。
