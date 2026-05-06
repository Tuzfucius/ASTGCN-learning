"""数据预处理入口脚本。"""

from __future__ import annotations

import argparse

from src.data.preprocessing import generate_dataset, save_dataset
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="准备 ASTGCN 训练数据。")
    parser.add_argument("--config", default="configurations/PEMS04_astgcn.yaml", help="配置文件路径。")
    return parser.parse_args()


def main() -> None:
    """执行数据预处理。

    TODO:
    - 读取配置。
    - 调用 generate_dataset。
    - 调用 save_dataset。
    - 打印生成数据的关键形状。
    """
    args = parse_args()
    config = load_config(args.config)
    dataset = generate_dataset(config)
    save_dataset(dataset, config["data"]["processed_dataset_filename"])


if __name__ == "__main__":
    main()
