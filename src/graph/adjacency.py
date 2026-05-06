"""邻接矩阵构造模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_adjacency_matrix(
    distance_filename: str | Path,
    num_of_vertices: int,
    id_filename: str | Path | None = None,
) -> tuple[Any, Any]:
    """从距离文件构造邻接矩阵。

    TODO:
    - 读取 `distance.csv`。
    - 构造 `adj_mx: (N, N)`。
    - 构造 `distance_mx: (N, N)`。
    - 如果存在 `id_filename`，按节点 ID 映射写入矩阵。
    """
    raise NotImplementedError("TODO: 实现 distance.csv 到邻接矩阵的转换。")
