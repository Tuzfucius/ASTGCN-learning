"""拉普拉斯矩阵和 Chebyshev 多项式工具。"""

from __future__ import annotations

from typing import Any

import numpy as np


def scaled_laplacian(adj_mx: Any) -> Any:
    """计算缩放拉普拉斯矩阵。

    返回形状为 `(N, N)` 的 `L_tilde = 2L / lambda_max - I`。
    """
    adj_mx = np.asarray(adj_mx, dtype=np.float32)
    if adj_mx.ndim != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，实际形状: {adj_mx.shape}")

    degree_mx = np.diag(np.sum(adj_mx, axis=1))
    laplacian = degree_mx - adj_mx
    lambda_max = np.max(np.real(np.linalg.eigvals(laplacian)))
    if np.isclose(lambda_max, 0.0):
        return -np.eye(adj_mx.shape[0], dtype=np.float32)

    scaled = (2.0 * laplacian) / lambda_max - np.eye(adj_mx.shape[0], dtype=np.float32)
    return scaled.astype(np.float32)


def chebyshev_polynomials(l_tilde: Any, k: int) -> list[Any]:
    """生成 Chebyshev 多项式列表。

    返回 `[T_0, T_1, ..., T_{k-1}]`。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0。")

    l_tilde = np.asarray(l_tilde, dtype=np.float32)
    if l_tilde.ndim != 2 or l_tilde.shape[0] != l_tilde.shape[1]:
        raise ValueError(f"缩放拉普拉斯矩阵必须为方阵，实际形状: {l_tilde.shape}")

    num_of_vertices = l_tilde.shape[0]
    polynomials = [np.eye(num_of_vertices, dtype=np.float32)]
    if k == 1:
        return polynomials

    polynomials.append(l_tilde.copy())
    for i in range(2, k):
        polynomials.append((2.0 * l_tilde @ polynomials[i - 1] - polynomials[i - 2]).astype(np.float32))

    return polynomials
