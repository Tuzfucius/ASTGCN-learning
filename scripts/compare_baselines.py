"""统一运行 ASTGCN 与常用 baseline 的性能对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from astgcn.baselines.gru import GRUBaseline
from astgcn.baselines.historical_average import HistoricalAverage
from astgcn.baselines.lstm import LSTMBaseline
from astgcn.baselines.svr import SVRBaseline
from astgcn.data.dataloader import build_dataloaders
from astgcn.data.graph import build_graph_data
from astgcn.losses import get_loss
from astgcn.metrics import compute_metrics
from astgcn.models.astgcn import ASTGCN
from astgcn.utils import ensure_dir, load_config, select_device, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 HA、SVR、LSTM、GRU、ASTGCN 的预测性能")
    parser.add_argument("--config", default="configs/pems04.yaml", help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=1, help="可训练模型的训练轮数")
    parser.add_argument("--max-batches", type=int, default=1, help="训练和评估最多使用多少个 batch")
    parser.add_argument("--svr-samples", type=int, default=256, help="SVR 最多使用多少个节点级样本训练")
    parser.add_argument("--device", default="cpu", help="运行设备，建议 smoke test 使用 cpu")
    parser.add_argument("--output-dir", default="outputs/comparison", help="对比结果输出目录")
    return parser.parse_args()


def build_astgcn(config: dict[str, Any], cheb_polynomials: np.ndarray, device: torch.device) -> ASTGCN:
    """根据配置构建 ASTGCN。

    :param config: 配置字典。
    :param cheb_polynomials: Chebyshev 多项式，形状为 ``[K, N, N]``。
    :param device: 运行设备。
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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 中的张量移动到目标设备。"""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """用统一接口执行模型前向。"""
    output = model(batch["recent"], batch["daily"], batch["weekly"], return_components=False)
    if isinstance(output, dict):
        return output["prediction"]
    return output


def train_torch_model(
    model: torch.nn.Module,
    train_loader: Any,
    device: torch.device,
    epochs: int,
    max_batches: int,
    learning_rate: float,
) -> None:
    """训练 PyTorch baseline 或 ASTGCN。

    :param model: 待训练模型。
    :param train_loader: 训练 DataLoader。
    :param device: 运行设备。
    :param epochs: 训练轮数。
    :param max_batches: 每轮最多训练多少个 batch。
    :param learning_rate: 学习率。
    """
    if not any(param.requires_grad for param in model.parameters()):
        return
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = get_loss("mae")
    for _ in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            pred = forward_model(model, batch).float()
            target = batch["target"].float()
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def collect_torch_predictions(
    model: torch.nn.Module,
    data_loader: Any,
    scaler: Any,
    device: torch.device,
    target_dim: int,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """收集 PyTorch 模型预测并反标准化。

    :return: ``prediction, target``，形状均为 ``[B, N, T_p]``。
    """
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        pred = forward_model(model, batch)
        preds.append(pred.detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
    pred_all = scaler.inverse_transform_target(np.concatenate(preds, axis=0), target_dim=target_dim)
    target_all = scaler.inverse_transform_target(np.concatenate(targets, axis=0), target_dim=target_dim)
    return pred_all, target_all


def collect_svr_predictions(
    model: SVRBaseline,
    data_loader: Any,
    scaler: Any,
    target_dim: int,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """收集 SVR 预测并反标准化。"""
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        preds.append(model.predict_batch(batch))
        targets.append(batch["target"].detach().cpu().numpy())
    pred_all = scaler.inverse_transform_target(np.concatenate(preds, axis=0), target_dim=target_dim)
    target_all = scaler.inverse_transform_target(np.concatenate(targets, axis=0), target_dim=target_dim)
    return pred_all, target_all


def save_metric_outputs(rows: list[dict[str, float]], output_dir: Path) -> pd.DataFrame:
    """保存指标 CSV、JSON 和柱状图。"""
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        axis.bar(frame["model"], frame[metric], color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "metrics_bar.png", dpi=160)
    plt.close(fig)
    return frame


def save_sample_prediction_plot(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    output_dir: Path,
    sample_id: int = 0,
    node_id: int = 0,
) -> None:
    """保存一个样本、一个节点的预测曲线对比图。"""
    steps = np.arange(1, target.shape[-1] + 1)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(steps, target[sample_id, node_id], marker="o", linewidth=2.2, label="Target", color="#111111")
    for name, pred in predictions.items():
        axis.plot(steps, pred[sample_id, node_id], marker="o", linewidth=1.4, label=name)
    axis.set_xlabel("Prediction step")
    axis.set_ylabel("Traffic flow")
    axis.set_title(f"Baseline prediction comparison: sample={sample_id}, node={node_id}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output_dir / "sample_prediction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """运行完整性能对比流程。"""
    args = parse_args()
    config = load_config(args.config)
    set_random_seed(int(config["train"].get("seed", 42)))
    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    dataset_cfg = config["dataset"]
    window_cfg = config["time_window"]
    train_cfg = config["train"]
    graph_cfg = config["graph"]
    train_loader, _, test_loader, scaler = build_dataloaders(
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

    torch_models: dict[str, torch.nn.Module] = {
        "HA": HistoricalAverage(pred_len=window_cfg["pred_len"], target_dim=dataset_cfg["target_dim"]).to(device),
        "LSTM": LSTMBaseline(
            in_channels=dataset_cfg["input_dim"],
            hidden_size=config["model"]["hidden_channels"],
            pred_len=window_cfg["pred_len"],
        ).to(device),
        "GRU": GRUBaseline(
            in_channels=dataset_cfg["input_dim"],
            hidden_size=config["model"]["hidden_channels"],
            pred_len=window_cfg["pred_len"],
        ).to(device),
        "ASTGCN": build_astgcn(config, graph_data["chebyshev_polynomials"], device),
    }

    rows: list[dict[str, float]] = []
    predictions: dict[str, np.ndarray] = {}
    target_ref: np.ndarray | None = None

    for name, model in torch_models.items():
        train_torch_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            max_batches=args.max_batches,
            learning_rate=float(train_cfg["learning_rate"]),
        )
        pred, target = collect_torch_predictions(
            model=model,
            data_loader=test_loader,
            scaler=scaler,
            device=device,
            target_dim=dataset_cfg["target_dim"],
            max_batches=args.max_batches,
        )
        metrics = compute_metrics(pred, target)
        rows.append({"model": name, "MAE": metrics["mae"], "RMSE": metrics["rmse"], "MAPE": metrics["mape"]})
        predictions[name] = pred
        target_ref = target

    svr = SVRBaseline(pred_len=window_cfg["pred_len"], max_samples=args.svr_samples)
    svr.fit_loader(train_loader, max_batches=args.max_batches)
    svr_pred, svr_target = collect_svr_predictions(
        model=svr,
        data_loader=test_loader,
        scaler=scaler,
        target_dim=dataset_cfg["target_dim"],
        max_batches=args.max_batches,
    )
    svr_metrics = compute_metrics(svr_pred, svr_target)
    rows.append({"model": "SVR", "MAE": svr_metrics["mae"], "RMSE": svr_metrics["rmse"], "MAPE": svr_metrics["mape"]})
    predictions["SVR"] = svr_pred
    target_ref = svr_target if target_ref is None else target_ref

    frame = save_metric_outputs(rows, output_dir)
    save_sample_prediction_plot(predictions, target_ref, output_dir)
    np.savez_compressed(output_dir / "baseline_predictions.npz", target=target_ref, **predictions)

    print(frame.to_string(index=False))
    print(f"对比结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
