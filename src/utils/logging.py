"""日志与实验目录工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def create_experiment_dir(config: dict[str, Any]) -> Path:
    """根据配置创建实验目录。"""
    root = Path(config["output"]["experiment_root"])
    dataset_name = config["data"]["dataset_name"]
    model_name = config["model"]["model_name"]
    experiment_dir = root / dataset_name / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> None:
    """保存当前实验配置副本。"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("保存配置快照需要先安装 PyYAML。") from exc

    output_path = Path(output_dir) / "config.yaml"
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def setup_logger(output_dir: str | Path) -> logging.Logger:
    """创建基础 logger。"""
    logger = logging.getLogger("astgcn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_path = Path(output_dir) / "train.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
