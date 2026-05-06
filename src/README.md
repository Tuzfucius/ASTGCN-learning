# 源码目录说明

本目录保存可复用源码。

整体分层：

```text
src/
  data/
  graph/
  metrics/
  models/
  engine/
  utils/
```

## 模块职责

| 目录 | 职责 |
| --- | --- |
| `data/` | 数据预处理和 DataLoader 构造。 |
| `graph/` | 邻接矩阵、拉普拉斯矩阵和 Chebyshev 多项式。 |
| `metrics/` | 损失函数和测试指标。 |
| `models/` | ASTGCN 模型结构。 |
| `engine/` | 训练和评估流程。 |
| `utils/` | 配置、日志、随机种子等通用工具。 |

## 依赖方向

推荐依赖方向：

```text
scripts -> engine -> models
scripts -> data
scripts -> graph
engine -> metrics
engine -> utils
models -> graph 生成的矩阵
```

不建议出现：

```text
models -> scripts
models -> engine
data -> models
metrics -> engine
```

## 张量约定

本项目统一使用：

```text
输入 x: (B, N, F, T)
输出 y_hat: (B, N, T_pred)
目标 y: (B, N, T_pred)
```

对于 PEMS04 第一阶段：

```text
x: (B, 307, 1, 12)
y_hat: (B, 307, 12)
y: (B, 307, 12)
```

## 实现原则

- 每个文件只承担一类职责。
- 每个核心函数应能独立测试。
- 函数参数尽量显式传入，少用全局变量。
- 模型层不读取文件。
- 训练层不实现模型数学细节。
