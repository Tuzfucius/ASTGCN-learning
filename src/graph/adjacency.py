"""邻接矩阵构造模块。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def load_adjacency_matrix(
    distance_filename: str | Path,
    num_of_vertices: int,
    id_filename: str | Path | None = None,
) -> tuple[Any, Any]:
    """从距离文件构造邻接矩阵。

    `distance.csv` 应包含三列: 起点、终点、距离。
    """
    distance_path = Path(distance_filename)
    if not distance_path.exists():
        raise FileNotFoundError(f"距离文件不存在: {distance_path}")

    if num_of_vertices <= 0:
        raise ValueError("num_of_vertices 必须大于 0。")

    if distance_path.suffix == ".npy":
        adj_mx = np.load(distance_path).astype(np.float32)
        if adj_mx.shape != (num_of_vertices, num_of_vertices):
            raise ValueError(f"邻接矩阵形状应为 {(num_of_vertices, num_of_vertices)}，实际为 {adj_mx.shape}")
        return adj_mx, None

    id_map = _load_id_map(id_filename) if id_filename is not None else None
    adj_mx = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    distance_mx = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    with distance_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            src_id, dst_id, distance = int(row[0]), int(row[1]), float(row[2])
            src_idx = id_map[src_id] if id_map is not None else src_id
            dst_idx = id_map[dst_id] if id_map is not None else dst_id
            _check_vertex_index(src_idx, num_of_vertices)
            _check_vertex_index(dst_idx, num_of_vertices)
            adj_mx[src_idx, dst_idx] = 1.0
            distance_mx[src_idx, dst_idx] = distance

    return adj_mx, distance_mx


def _load_id_map(id_filename: str | Path) -> dict[int, int]:
    """读取节点 ID 到矩阵下标的映射。"""
    id_path = Path(id_filename)
    if not id_path.exists():
        raise FileNotFoundError(f"节点 ID 文件不存在: {id_path}")

    with id_path.open("r", encoding="utf-8") as file:
        node_ids = [int(line.strip()) for line in file if line.strip()]

    return {node_id: idx for idx, node_id in enumerate(node_ids)}


def _check_vertex_index(index: int, num_of_vertices: int) -> None:
    """检查节点下标是否落在邻接矩阵范围内。"""
    if index < 0 or index >= num_of_vertices:
        raise ValueError(f"节点下标越界: {index}，节点数量: {num_of_vertices}")
