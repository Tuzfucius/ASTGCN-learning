"""配置读取与校验工具。

本文件只负责把配置文件读成 Python 对象，并检查必要字段是否存在。
不要在这里写数据加载、模型构建或训练逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ("data", "task", "model", "training", "output")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 yaml 配置文件。"""
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = resolve_project_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("请先安装 PyYAML，或改用标准库 configparser。") from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    validate_config(config)
    resolve_paths(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """检查配置文件是否包含基础字段。"""
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(f"配置缺少必要分组: {missing_sections}")

    required_keys = {
        "data": (
            "dataset_name",
            "graph_signal_matrix_filename",
            "processed_dataset_filename",
            "adj_filename",
            "num_of_vertices",
            "points_per_hour",
        ),
        "task": (
            "len_input",
            "num_for_predict",
            "num_of_hours",
            "num_of_days",
            "num_of_weeks",
            "in_channels",
            "target_channel",
        ),
        "model": ("model_name", "nb_block", "K", "nb_chev_filter", "nb_time_filter", "time_strides"),
        "training": (
            "batch_size",
            "epochs",
            "start_epoch",
            "learning_rate",
            "loss_function",
            "metric_method",
            "missing_value",
            "device",
            "seed",
        ),
        "output": ("experiment_root", "save_best_only", "save_predictions"),
    }
    for section, keys in required_keys.items():
        missing_keys = [key for key in keys if key not in config[section]]
        if missing_keys:
            raise ValueError(f"配置分组 {section} 缺少字段: {missing_keys}")

    positive_fields = (
        ("data", "num_of_vertices"),
        ("data", "points_per_hour"),
        ("task", "len_input"),
        ("task", "num_for_predict"),
        ("task", "in_channels"),
        ("model", "nb_block"),
        ("model", "K"),
        ("model", "nb_chev_filter"),
        ("model", "nb_time_filter"),
        ("model", "time_strides"),
        ("training", "batch_size"),
        ("training", "epochs"),
    )
    for section, key in positive_fields:
        if int(config[section][key]) <= 0:
            raise ValueError(f"配置 {section}.{key} 必须大于 0。")

    non_negative_fields = (
        ("task", "num_of_hours"),
        ("task", "num_of_days"),
        ("task", "num_of_weeks"),
        ("task", "target_channel"),
        ("training", "start_epoch"),
    )
    for section, key in non_negative_fields:
        if int(config[section][key]) < 0:
            raise ValueError(f"配置 {section}.{key} 不能小于 0。")

    if config["training"]["metric_method"] not in ("mask", "unmask"):
        raise ValueError("training.metric_method 只支持 mask 或 unmask。")


def get_project_root() -> Path:
    """返回项目根目录。

    当前文件位于 `src/utils/config.py`，向上两级是 `src/`，再向上一级是项目根目录。
    """
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """将配置中的相对路径转换为项目根目录下路径。

    返回 Path 对象，调用方可按需转换为字符串。
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def resolve_paths(config: dict[str, Any]) -> None:
    """将配置中的项目相对路径转换为绝对路径字符串。"""
    data_path_keys = ("graph_signal_matrix_filename", "processed_dataset_filename", "adj_filename")
    for key in data_path_keys:
        config["data"][key] = str(resolve_project_path(config["data"][key]))

    config["output"]["experiment_root"] = str(resolve_project_path(config["output"]["experiment_root"]))
