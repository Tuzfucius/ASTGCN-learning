"""交通图结构构造工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def get_distance_matrix(
    file_path: str | Path,
    num_nodes: int | None = None,
    directed: bool = False,
) -> np.ndarray:
    """从 ``distance.csv`` 构造距离矩阵。

    :param file_path: CSV 路径，要求包含 ``from``、``to``、``cost`` 三列。
    :param num_nodes: 节点数；为空时根据边表最大节点编号推断。
    :param directed: 是否按有向图处理。默认 ``False``，符合论文无向图设定。
    :return: 距离矩阵，形状为 ``[N, N]``。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到距离文件: {path}")

    edges = pd.read_csv(path)
    required = {"from", "to", "cost"}
    if not required.issubset(edges.columns):
        raise ValueError(f"distance.csv 必须包含列: {sorted(required)}")

    max_node_id = int(max(edges["from"].max(), edges["to"].max()))
    node_count = num_nodes if num_nodes is not None else max_node_id + 1
    if node_count <= max_node_id:
        raise ValueError("num_nodes 小于边表中的最大节点编号")

    distance = np.zeros((node_count, node_count), dtype=np.float32)
    for _, row in edges.iterrows():
        src = int(row["from"])
        dst = int(row["to"])
        cost = float(row["cost"])
        distance[src, dst] = cost
        if not directed:
            distance[dst, src] = cost
    return distance


def get_adjacency_matrix(distance_matrix: np.ndarray, weighted: bool = False) -> np.ndarray:
    """根据距离矩阵构造邻接矩阵。

    :param distance_matrix: 距离矩阵，形状为 ``[N, N]``。
    :param weighted: 是否保留距离权重。默认 ``False`` 表示只保留连接关系。
    :return: 邻接矩阵，形状为 ``[N, N]``。
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix 必须是方阵")
    if weighted:
        return distance_matrix.astype(np.float32)
    return (distance_matrix > 0).astype(np.float32)


def get_normalized_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """计算归一化拉普拉斯矩阵 ``L = I - D^{-1/2} A D^{-1/2}``。

    :param adj_matrix: 邻接矩阵，形状为 ``[N, N]``。
    :return: 归一化拉普拉斯矩阵，形状为 ``[N, N]``。
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix 必须是方阵")

    degree = adj_matrix.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    d_mat = np.diag(d_inv_sqrt)
    identity = np.eye(adj_matrix.shape[0], dtype=np.float32)
    return (identity - d_mat @ adj_matrix @ d_mat).astype(np.float32)


def get_scaled_laplacian(laplacian: np.ndarray) -> np.ndarray:
    """计算缩放拉普拉斯矩阵 ``L_tilde = 2L / lambda_max - I``。

    :param laplacian: 归一化拉普拉斯矩阵，形状为 ``[N, N]``。
    :return: 缩放拉普拉斯矩阵，形状为 ``[N, N]``。
    """
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian 必须是方阵")

    lambda_max = float(np.linalg.eigvals(laplacian).real.max())
    if abs(lambda_max) < 1e-8:
        lambda_max = 1.0
    identity = np.eye(laplacian.shape[0], dtype=np.float32)
    return (2.0 * laplacian / lambda_max - identity).astype(np.float32)


def get_chebyshev_polynomials(scaled_laplacian: np.ndarray, k_order: int) -> np.ndarray:
    """生成 Chebyshev 多项式矩阵。

    :param scaled_laplacian: 缩放拉普拉斯矩阵，形状为 ``[N, N]``。
    :param k_order: Chebyshev 阶数 ``K``。
    :return: 多项式矩阵，形状为 ``[K, N, N]``。
    """
    if scaled_laplacian.ndim != 2 or scaled_laplacian.shape[0] != scaled_laplacian.shape[1]:
        raise ValueError("scaled_laplacian 必须是方阵")
    if k_order <= 0:
        raise ValueError("k_order 必须是正整数")

    num_nodes = scaled_laplacian.shape[0]
    polynomials = [np.eye(num_nodes, dtype=np.float32)]
    if k_order > 1:
        polynomials.append(scaled_laplacian.astype(np.float32))
    for order in range(2, k_order):
        polynomials.append((2 * scaled_laplacian @ polynomials[-1] - polynomials[-2]).astype(np.float32))
    return np.stack(polynomials, axis=0).astype(np.float32)


def build_graph_data(
    file_path: str | Path,
    k_order: int,
    num_nodes: int | None = None,
    directed: bool = False,
    weighted: bool = False,
) -> dict[str, np.ndarray]:
    """一次性构造 ASTGCN 所需图数据。

    :param file_path: ``distance.csv`` 路径。
    :param k_order: Chebyshev 阶数。
    :param num_nodes: 节点数。
    :param directed: 是否有向图。
    :param weighted: 邻接矩阵是否保留距离权重。
    :return: 包含距离矩阵、邻接矩阵、拉普拉斯矩阵和 Chebyshev 多项式的字典。
    """
    distance = get_distance_matrix(file_path, num_nodes=num_nodes, directed=directed)
    adjacency = get_adjacency_matrix(distance, weighted=weighted)
    laplacian = get_normalized_laplacian(adjacency)
    scaled = get_scaled_laplacian(laplacian)
    cheb = get_chebyshev_polynomials(scaled, k_order)
    return {
        "distance_matrix": distance,
        "adjacency_matrix": adjacency,
        "normalized_laplacian": laplacian,
        "scaled_laplacian": scaled,
        "chebyshev_polynomials": cheb,
    }
