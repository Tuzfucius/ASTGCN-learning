"""数据预处理入口脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import generate_dataset, save_dataset
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="准备 ASTGCN 训练数据。")
    parser.add_argument("--config", default="configurations/PEMS04_astgcn.yaml", help="配置文件路径。")
    return parser.parse_args()


def main() -> None:
    """执行数据预处理。"""
    args = parse_args()
    config = load_config(args.config)
    dataset = generate_dataset(config)
    save_dataset(dataset, config["data"]["processed_dataset_filename"])

    print("数据预处理完成:")
    print(f"train_x: {dataset['train_x'].shape}")
    print(f"train_target: {dataset['train_target'].shape}")
    print(f"val_x: {dataset['val_x'].shape}")
    print(f"val_target: {dataset['val_target'].shape}")
    print(f"test_x: {dataset['test_x'].shape}")
    print(f"test_target: {dataset['test_target'].shape}")
    print(f"mean: {dataset['mean'].shape}")
    print(f"std: {dataset['std'].shape}")


if __name__ == "__main__":
    main()
