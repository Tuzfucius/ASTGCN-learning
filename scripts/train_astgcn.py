"""ASTGCN 训练入口脚本。"""

from __future__ import annotations

import argparse

from src.data.dataset import build_all_dataloaders
from src.engine.trainer import Trainer
from src.graph.adjacency import load_adjacency_matrix
from src.graph.laplacian import chebyshev_polynomials, scaled_laplacian
from src.models.astgcn import build_astgcn_model
from src.utils.config import load_config
from src.utils.logging import create_experiment_dir, save_config_snapshot
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练 ASTGCN 模型。")
    parser.add_argument("--config", default="configurations/PEMS04_astgcn.yaml", help="配置文件路径。")
    return parser.parse_args()


def main() -> None:
    """执行训练流程。

    TODO:
    - 读取配置。
    - 固定随机种子。
    - 创建实验目录。
    - 加载 DataLoader。
    - 构造图结构和 Chebyshev 多项式。
    - 构造 ASTGCN 模型。
    - 构造 optimizer 和 loss_fn。
    - 创建 Trainer 并调用 fit。
    """
    args = parse_args()
    config = load_config(args.config)
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

    # TODO: 在这里创建 optimizer 和 loss_fn。
    optimizer = None
    loss_fn = None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        output_dir=output_dir,
        device=config["training"]["device"],
    )
    trainer.fit(config["training"]["epochs"], config["training"]["start_epoch"])


if __name__ == "__main__":
    main()
