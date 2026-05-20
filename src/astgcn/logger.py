"""日志工具。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str = "astgcn",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """创建同时输出到控制台和文件的 logger。

    参数:
        name: logger 名称。
        log_file: 日志文件路径；为 ``None`` 时只输出到控制台。
        level: 日志级别，默认 ``logging.INFO``。

    返回:
        配置好的 ``logging.Logger`` 对象。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(getattr(handler, "_astgcn_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler._astgcn_console = True
        logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        has_file = any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
