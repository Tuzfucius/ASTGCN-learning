import numpy as np


def load_pems_npz(file_path: str) -> np.ndarray:
    """
    读取 PEMS 数据集的 npz 文件。

    期望输出形状:
        [T, N, F]

    T: 时间步数量
    N: 节点数量
    F: 特征数量
    """
    raw = np.load(file_path)

    print("npz keys:", raw.files)

    if "data" in raw.files:
        data = raw["data"]
    else:
        # 如果没有 data 这个 key，就默认取第一个数组
        data = raw[raw.files[0]]

    if data.ndim != 3:
        raise ValueError(f"PEMS 数据必须是三维数组，当前 shape={data.shape}")

    data = data.astype(np.float32)

    return data