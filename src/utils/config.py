"""配置读取与校验工具。

本文件只负责把配置文件读成 Python 对象，并检查必要字段是否存在。
不要在这里写数据加载、模型构建或训练逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ("data", "task", "model", "training", "output")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 yaml 配置文件。

    TODO:
    - 使用 `yaml.safe_load` 读取配置。
    - 调用 `validate_config(config)` 检查必要字段。
    - 调用 `resolve_paths(config, project_root)` 将相对路径统一处理。
    - 返回配置字典。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("请先安装 PyYAML，或改用标准库 configparser。") from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """检查配置文件是否包含基础分组。

    TODO:
    - 继续检查每个分组下的必要字段。
    - 检查整数参数是否为正数。
    - 检查 `num_of_vertices` 是否与数据集一致。
    """
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(f"配置缺少必要分组: {missing_sections}")


def get_project_root() -> Path:
    """返回项目根目录。

    当前文件位于 `src/utils/config.py`，向上两级是 `src/`，再向上一级是项目根目录。
    """
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """将配置中的相对路径转换为项目根目录下路径。

    TODO:
    - 后续可以支持环境变量展开。
    - 后续可以支持 Windows 和 Linux 路径兼容。
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path
