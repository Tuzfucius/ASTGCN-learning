"""ASTGCN 训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from astgcn.data.dataloader import build_dataloaders
from astgcn.data.graph import build_graph_data
from astgcn.engine.trainer import Trainer
from astgcn.logger import get_logger
from astgcn.models.astgcn import ASTGCN
from astgcn.utils import ensure_dir, load_config, select_device, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 ASTGCN 模型")
    parser.add_argument("--config", default="configs/pems04.yaml", help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--max-batches", type=int, default=None, help="每轮最多训练多少个 batch，用于 smoke test")
    return parser.parse_args()


def build_model(config: dict, cheb_polynomials, device: torch.device) -> ASTGCN:
    """根据配置构建 ASTGCN 模型。

    :param config: 配置字典。
    :param cheb_polynomials: Chebyshev 多项式，形状为 ``[K, N, N]``。
    :param device: 训练设备。
    :return: ASTGCN 模型。
    """
    dataset_cfg = config["dataset"]
    window_cfg = config["time_window"]
    model_cfg = config["model"]
    model = ASTGCN(
        num_nodes=dataset_cfg["num_nodes"],
        in_channels=model_cfg["in_channels"],
        pred_len=window_cfg["pred_len"],
        cheb_polynomials=cheb_polynomials,
        recent_timesteps=window_cfg["recent_len"],
        daily_timesteps=window_cfg["daily_days"] * window_cfg["pred_len"],
        weekly_timesteps=window_cfg["weekly_weeks"] * window_cfg["pred_len"],
        num_blocks=model_cfg["num_blocks"],
        hidden_channels=model_cfg["hidden_channels"],
        temporal_kernel_size=model_cfg["time_kernel_size"],
    )
    return model.to(device)


def main() -> None:
    """执行训练流程。"""
    args = parse_args()
    config = load_config(args.config)
    set_random_seed(int(config["train"].get("seed", 42)))

    log_dir = ensure_dir(config["log"]["log_dir"])
    checkpoint_dir = ensure_dir(config["log"]["checkpoint_dir"])
    logger = get_logger("astgcn.train", log_dir / "train.log")
    device = select_device(config["train"].get("device", "auto"))
    logger.info("使用设备: %s", device)

    dataset_cfg = config["dataset"]
    window_cfg = config["time_window"]
    train_cfg = config["train"]
    graph_cfg = config["graph"]

    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        data_path=dataset_cfg["data_path"],
        num_recent=window_cfg["recent_len"],
        num_days=window_cfg["daily_days"],
        num_weeks=window_cfg["weekly_weeks"],
        pred_len=window_cfg["pred_len"],
        batch_size=train_cfg["batch_size"],
        points_per_day=dataset_cfg["points_per_day"],
        target_dim=dataset_cfg["target_dim"],
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        num_workers=train_cfg.get("num_workers", 0),
    )
    logger.info("DataLoader: train=%s val=%s test=%s", len(train_loader), len(val_loader), len(test_loader))

    graph_data = build_graph_data(
        file_path=dataset_cfg["distance_path"],
        k_order=graph_cfg["cheb_order"],
        num_nodes=dataset_cfg["num_nodes"],
        directed=graph_cfg.get("directed", False),
        weighted=graph_cfg.get("weighted", False),
    )
    model = build_model(config, graph_data["chebyshev_polynomials"], device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_name=train_cfg.get("loss", "mae"),
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        config=config,
        scaler=scaler,
    )
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    history = trainer.fit(
        epochs=epochs,
        patience=train_cfg.get("early_stop_patience"),
        max_batches=args.max_batches,
    )
    logger.info("训练结束: %s", history)
    logger.info("最优 checkpoint: %s", Path(checkpoint_dir) / "best.pt")


if __name__ == "__main__":
    main()
