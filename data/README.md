# 数据目录说明

本目录保存交通数据集和预处理结果。

数据文件体积较大，不提交到 git。当前 `.gitignore` 已排除：

```text
/data/PEMS04/*.npz
/data/PEMS04/*.csv
```

## 推荐结构

```text
data/
  PEMS04/
    pems04.npz
    distance.csv
    PEMS04_r1_d0_w0_astcgn.npz
    README.md
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `pems04.npz` | 原始交通时序数据。 |
| `distance.csv` | 节点之间的距离或连接关系，用于构造图结构。 |
| `PEMS04_r1_d0_w0_astcgn.npz` | 预处理后的训练、验证、测试数据。 |

## 数据使用链路

```text
pems04.npz
  -> scripts/prepare_data.py
  -> src/data/preprocessing.py
  -> PEMS04_r1_d0_w0_astcgn.npz
  -> src/data/dataset.py
  -> DataLoader
```

## 注意事项

- 原始数据只作为输入，不在训练阶段反复切片。
- 预处理结果中应保存训练集、验证集、测试集和标准化统计量。
- 标准化统计量只能由训练集计算，验证集和测试集复用训练集均值、标准差。
