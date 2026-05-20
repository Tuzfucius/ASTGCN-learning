# models 目录说明

本目录用于实现 ASTGCN 网络结构。ASTGCN 的核心思想是同时建模交通路网的空间依赖和时间依赖，并融合 recent、daily、weekly 三类历史信息。

## 主要文件

- `attention.py`：时间注意力模块，后续可扩展空间注意力模块。
- `cheb_conv.py`：预留 Chebyshev 图卷积实现。
- `temporal_conv.py`：预留时间卷积实现。
- `st_block.py`：预留时空块，通常组合时间注意力、空间注意力、图卷积、时间卷积、残差和归一化。
- `component.py`：预留单个 ASTGCN 分支组件。
- `fusion.py`：预留三分支融合层。
- `astgcn.py`：预留顶层 ASTGCN 模型。

## 推荐实现顺序

1. 先实现并测试输入输出维度。
2. 实现时间注意力和空间注意力。
3. 实现 Chebyshev 图卷积。
4. 组合为 STBlock。
5. 组合为 recent、daily、weekly 三个 Component。
6. 加入 FusionLayer 输出最终预测。

## 维度提醒

数据层输出通常是 `[B, T, N, F]`，论文和注意力模块更常使用 `[B, N, F, T]`。模型内部必须统一转换，不能在不同模块中混用。

最小 forward 检查应满足：

```text
recent/daily/weekly -> y_hat
[B, T, N, F]        -> [B, N, Tp]
```
