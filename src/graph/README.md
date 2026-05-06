# `src/graph` 图结构模块说明

本目录负责 ASTGCN 所需的图结构计算。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `adjacency.py` | 从 `distance.csv` 构造邻接矩阵。 |
| `laplacian.py` | 计算缩放拉普拉斯矩阵和 Chebyshev 多项式。 |

## 变量说明

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `adj_mx` | `(N, N)` | 邻接矩阵。 |
| `distance_mx` | `(N, N)` | 距离矩阵，可选。 |
| `degree_mx` | `(N, N)` | 度矩阵。 |
| `laplacian` | `(N, N)` | 图拉普拉斯矩阵。 |
| `scaled_laplacian` | `(N, N)` | 缩放后的拉普拉斯矩阵。 |
| `cheb_polynomials` | `K * (N, N)` | Chebyshev 多项式。 |

PEMS04 中：

```text
N = 307
```

## `adjacency.py`

建议函数：

```python
load_adjacency_matrix(distance_filename, num_of_vertices, id_filename=None)
```

职责：

- 读取 `distance.csv`。
- 根据节点数量构造邻接矩阵。
- 如果存在节点 ID 映射，则按映射写入矩阵。
- 返回 `(adj_mx, distance_mx)`。

## `laplacian.py`

建议函数：

```python
scaled_laplacian(adj_mx)
chebyshev_polynomials(l_tilde, k)
```

职责：

- 计算图拉普拉斯矩阵。
- 对拉普拉斯矩阵做缩放。
- 递推生成 Chebyshev 多项式。

## Chebyshev 递推

```text
T_0 = I
T_1 = L_tilde
T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
```

其中：

- `I` 是单位矩阵。
- `L_tilde` 是缩放拉普拉斯矩阵。
- `K` 是多项式阶数。

## 注意事项

- 邻接矩阵节点数必须和数据节点数一致。
- 图结构模块不应依赖模型类。
- 图结构计算结果可传入模型构造函数。
