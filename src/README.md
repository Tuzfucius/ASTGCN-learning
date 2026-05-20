# src 目录说明

本目录是 Python 包源码目录。项目使用 `pyproject.toml` 中的 `src` 布局，安装后可通过 `import astgcn` 引入包内模块。

## 职责

- 隔离可复用代码和一次性运行脚本。
- 为数据处理、图构建、模型、训练评估和 baseline 提供统一包结构。
- 便于在本地、Kaggle 和测试脚本中用相同方式导入代码。

## 安装当前包

```bash
python -m pip install -e .
```

安装后可以检查导入：

```bash
python -c "import astgcn; print(astgcn.__version__)"
```

如果只在 notebook 中临时运行，也可以把项目根目录加入 `sys.path`，但正式实验建议使用可编辑安装。
