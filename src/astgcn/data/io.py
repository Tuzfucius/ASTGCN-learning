"""PEMS 数据文件读取工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pems_npz(file_path: str | Path) -> np.ndarray:
    """读取 PEMS 时序数据。

    :param file_path: ``.npz`` 文件路径，文件中优先读取 ``data`` 数组。
    :return: 原始时序数据，形状为 ``[T, N, F]``，其中 ``T`` 是时间步数，
        ``N`` 是节点数，``F`` 是特征数。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 PEMS 数据文件: {path}")

    raw = np.load(path)
    key = "data" if "data" in raw.files else raw.files[0]
    data = raw[key]

    if data.ndim != 3:
        raise ValueError(f"PEMS 数据必须是三维数组 [T, N, F]，当前 shape={data.shape}")

    return data.astype(np.float32)
