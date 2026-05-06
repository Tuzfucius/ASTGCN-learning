"""日志与实验目录工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def create_experiment_dir(config: dict[str, Any]) -> Path:
    """根据配置创建实验目录。

    TODO:
    - 目录格式建议为 `experiments/{dataset_name}/{model_name}`。
    - 如果同名目录已存在，决定是覆盖、追加时间戳，还是继续训练。
    """
    root = Path(config["output"]["experiment_root"])
    dataset_name = config["data"]["dataset_name"]
    model_name = config["model"]["model_name"]
    experiment_dir = root / dataset_name / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> None:
    """保存当前实验配置副本。

    TODO:
    - 如果后续使用 yaml，可以保存为 `config.yaml`。
    - 当前先用 JSON 便于无额外依赖写入。
    """
    output_path = Path(output_dir) / "config.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def setup_logger(output_dir: str | Path) -> logging.Logger:
    """创建基础 logger。

    TODO:
    - 后续可增加文件日志 `train.log`。
    - 后续可增加格式化输出和日志等级配置。
    """
    logger = logging.getLogger("astgcn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger
