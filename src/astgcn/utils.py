"""项目通用工具函数。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    :param path: YAML 配置文件路径。
    :return: 配置字典。
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须解析为字典: {config_path}")
    return config


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。

    :param path: 目录路径。
    :return: 已创建或已存在的目录路径。
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。

    :param seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str = "auto") -> torch.device:
    """选择训练设备。

    :param device: ``auto``、``cpu``、``cuda`` 或具体 CUDA 设备名。
    :return: PyTorch 设备对象。
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)
