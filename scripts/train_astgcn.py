"""ASTGCN 训练入口脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_all_dataloaders
from src.engine.trainer import Trainer
from src.graph.adjacency import load_adjacency_matrix
from src.graph.laplacian import chebyshev_polynomials, scaled_laplacian
from src.metrics.losses import get_loss_function
from src.models.astgcn import build_astgcn_model
from src.utils.config import load_config
from src.utils.logging import create_experiment_dir, save_config_snapshot
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练 ASTGCN 模型。")
    parser.add_argument("--config", default="configurations/PEMS04_astgcn.yaml", help="配置文件路径。")
    parser.add_argument("--device", default=None, help="覆盖配置中的训练设备，例如 cpu 或 cuda。")
    return parser.parse_args()


def main() -> None:
    """执行训练流程。"""
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device

    set_seed(config["training"]["seed"])
    output_dir = create_experiment_dir(config)
    save_config_snapshot(config, output_dir)

    dataloaders = build_all_dataloaders(config)
    adj_mx, _ = load_adjacency_matrix(
        config["data"]["adj_filename"],
        config["data"]["num_of_vertices"],
    )
    l_tilde = scaled_laplacian(adj_mx)
    cheb_polys = chebyshev_polynomials(l_tilde, config["model"]["K"])
    model = build_astgcn_model(config, cheb_polys)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = get_loss_function(config["training"]["loss_function"], config["training"]["missing_value"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        output_dir=output_dir,
        device=config["training"]["device"],
    )
    history = trainer.fit(config["training"]["epochs"], config["training"]["start_epoch"])
    if history["best_epoch"] is not None:
        print(f"best_epoch: {history['best_epoch'] + 1}, best_val_loss: {history['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
