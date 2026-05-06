"""数据加载模块。

本文件负责把预处理后的 `.npz` 文件转换为 PyTorch DataLoader。
不要在这里重新切片原始时间序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_processed_dataset(processed_path: str | Path) -> dict[str, Any]:
    """读取预处理后的 `.npz` 文件。

    TODO:
    - 使用 `np.load` 读取预处理文件。
    - 返回 train/val/test 输入、目标和 mean/std。
    - 检查 `x` 是否为 `(B, N, F, T)`。
    """
    raise NotImplementedError("TODO: 实现预处理数据读取逻辑。")


def build_dataloader(x: Any, y: Any, batch_size: int, shuffle: bool) -> Any:
    """构造单个 DataLoader。

    TODO:
    - 将 numpy 数组转成 torch.FloatTensor。
    - 用 `TensorDataset(x_tensor, y_tensor)` 封装。
    - 返回 `DataLoader`。
    """
    raise NotImplementedError("TODO: 实现 DataLoader 构造逻辑。")


def build_all_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置构造 train/val/test DataLoader。

    TODO:
    - 从 `processed_dataset_filename` 读取数据。
    - 构造 train_loader、val_loader、test_loader。
    - 同时返回 target 张量和 mean/std，供评估使用。
    """
    raise NotImplementedError("TODO: 实现三类 DataLoader 构造逻辑。")
