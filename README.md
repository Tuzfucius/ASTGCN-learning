# ASTGCN 手工复现与重构项目

本项目用于手工复现 ASTGCN 模型，并在保持模型核心结构不变的前提下，对原始代码进行更清晰的工程化重构。

官方参考仓库：

```text
https://github.com/guoshnBJTU/ASTGCN-2019-pytorch
```

## 项目目标

本项目当前优先复现 ASTGCN 的 recent component，用于交通数据集预测任务。

第一阶段目标是基于 PEMS04 数据集跑通以下流程：

```text
数据预处理
  -> 图结构构建
  -> ASTGCN 模型前向
  -> 训练
  -> 测试评估
  -> 保存实验结果
```

当前建议固定任务为：

```text
所有节点过去 12 个时间步的第 0 个特征
预测
所有节点未来 12 个时间步的第 0 个特征
```

输入形状：

```text
(B, 307, 1, 12)
```

输出形状：

```text
(B, 307, 12)
```

## 当前目录说明

```text
docs/
  手动复现与重构指南.md

data/
  PEMS04/

PEMS04/
  原始数据文件暂存目录
```

说明：

- `docs/` 保存中文复现与重构说明。
- `data/PEMS04/` 是规划中的 PEMS04 数据目录。
- `PEMS04/` 是当前已有数据集的原始暂存目录，后续应迁移到 `data/PEMS04/`。
- 数据集文件已在 `.gitignore` 中排除，不应提交到 git。

## 建议重构方向

后续代码建议按以下结构手工撰写：

```text
configurations/
src/
  data/
  graph/
  metrics/
  models/
  engine/
  utils/
scripts/
experiments/
```

核心原则：

- 数据处理、图计算、模型结构、训练流程分层编写。
- 训练脚本只作为入口，不堆积核心逻辑。
- 模型结构保持 ASTGCN 原算法思路不变。
- 每个模块优先写清楚输入输出形状。

文档总入口见：

```text
docs/README.md
```

详细手工复现计划见：

```text
docs/手动复现与重构指南.md
```

## 运行依赖

最小 Python 依赖见：

```text
requirements.txt
```

其中 PyTorch 的安装方式可能和 CUDA 版本有关。如果目标机器需要 GPU 训练，建议先按 PyTorch 官方说明安装匹配 CUDA 的 `torch`，再安装其余依赖。

## 数据集与 git 规则

数据集文件通常体积较大，因此不进入 git。

当前 `.gitignore` 已排除：

```text
/PEMS04/
/data/PEMS04/*.npz
/data/PEMS04/*.csv
```

如果后续移动数据集，推荐目标位置为：

```text
data/PEMS04/
```

## 后续手工撰写顺序

建议按以下顺序推进：

1. 完成目录骨架和空文件。
2. 编写配置读取。
3. 编写数据预处理。
4. 编写图结构工具。
5. 编写指标函数。
6. 编写 ASTGCN 模型层。
7. 编写训练流程。
8. 编写评估流程。

每完成一个小任务，建议使用 git 单独提交。
