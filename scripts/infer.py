"""ASTGCN 推理与评估入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from astgcn.data.dataloader import build_dataloaders
from astgcn.data.graph import build_graph_data
from astgcn.engine.checkpoint import load_checkpoint
from astgcn.engine.evaluator import Evaluator
from astgcn.engine.predictor import Predictor
from astgcn.logger import get_logger
from astgcn.models.astgcn import ASTGCN
from astgcn.utils import ensure_dir, load_config, select_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 ASTGCN 模型")
    parser.add_argument("--config", default="configs/pems04.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径，默认读取配置输出目录下的 best.pt")
    parser.add_argument("--max-batches", type=int, default=None, help="最多评估多少个 batch，用于 smoke test")
    parser.add_argument("--save-components", action="store_true", help="保存 recent/daily/weekly 分支预测")
    return parser.parse_args()


def build_model(config: dict, cheb_polynomials, device: torch.device) -> ASTGCN:
    """根据配置构建模型。"""
    dataset_cfg = config["dataset"]
    window_cfg = config["time_window"]
    model_cfg = config["model"]
    return ASTGCN(
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
    ).to(device)


def main() -> None:
    """执行评估与预测保存。"""
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config["train"].get("device", "auto"))

    log_dir = ensure_dir(config["log"]["log_dir"])
    prediction_dir = ensure_dir(config["log"]["prediction_dir"])
    logger = get_logger("astgcn.infer", log_dir / "infer.log")

    dataset_cfg = config["dataset"]
    window_cfg = config["time_window"]
    train_cfg = config["train"]
    graph_cfg = config["graph"]
    _, _, test_loader, scaler = build_dataloaders(
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
    graph_data = build_graph_data(
        file_path=dataset_cfg["distance_path"],
        k_order=graph_cfg["cheb_order"],
        num_nodes=dataset_cfg["num_nodes"],
        directed=graph_cfg.get("directed", False),
        weighted=graph_cfg.get("weighted", False),
    )
    model = build_model(config, graph_data["chebyshev_polynomials"], device)

    checkpoint = args.checkpoint or Path(config["log"]["checkpoint_dir"]) / "best.pt"
    load_checkpoint(checkpoint, model=model, map_location=device, strict=True)
    logger.info("已加载 checkpoint: %s", checkpoint)

    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        scaler=scaler,
        device=device,
        target_dim=dataset_cfg["target_dim"],
    )
    metrics = evaluator.evaluate(max_batches=args.max_batches)
    logger.info("测试集指标: %s", metrics)

    output_path = prediction_dir / "test_predictions.npz"
    predictor = Predictor(model=model, data_loader=test_loader, device=device)
    predictor.predict(output_path, save_components=args.save_components, max_batches=args.max_batches)
    logger.info("预测结果已保存: %s", output_path)


if __name__ == "__main__":
    main()
