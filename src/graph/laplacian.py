"""拉普拉斯矩阵和 Chebyshev 多项式工具。"""

from __future__ import annotations

from typing import Any


def scaled_laplacian(adj_mx: Any) -> Any:
    """计算缩放拉普拉斯矩阵。

    TODO:
    - 根据邻接矩阵计算度矩阵。
    - 计算图拉普拉斯矩阵。
    - 计算最大特征值。
    - 将拉普拉斯矩阵缩放到 Chebyshev 适用范围。
    """
    raise NotImplementedError("TODO: 实现缩放拉普拉斯矩阵计算。")


def chebyshev_polynomials(l_tilde: Any, k: int) -> list[Any]:
    """生成 Chebyshev 多项式列表。

    TODO:
    - T_0 = I。
    - T_1 = L_tilde。
    - T_i = 2 * L_tilde * T_{i-1} - T_{i-2}。
    - 返回长度为 k 的列表。
    """
    raise NotImplementedError("TODO: 实现 Chebyshev 多项式递推。")
