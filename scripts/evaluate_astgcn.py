"""ASTGCN 测试评估入口脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_all_dataloaders
from src.engine.evaluator import Evaluator
from src.graph.adjacency import load_adjacency_matrix
from src.graph.laplacian import chebyshev_polynomials, scaled_laplacian
from src.models.astgcn import build_astgcn_model
from src.utils.config import load_config
from src.utils.logging import create_experiment_dir


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="评估 ASTGCN 模型。")
    parser.add_argument("--config", default="configurations/PEMS04_astgcn.yaml", help="配置文件路径。")
    parser.add_argument("--model", required=True, help="模型权重路径。")
    parser.add_argument("--device", default=None, help="覆盖配置中的评估设备，例如 cpu 或 cuda。")
    return parser.parse_args()


def main() -> None:
    """执行测试评估。"""
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device

    output_dir = create_experiment_dir(config)
    dataloaders = build_all_dataloaders(config)

    adj_mx, _ = load_adjacency_matrix(
        config["data"]["adj_filename"],
        config["data"]["num_of_vertices"],
    )
    l_tilde = scaled_laplacian(adj_mx)
    cheb_polys = chebyshev_polynomials(l_tilde, config["model"]["K"])
    model = build_astgcn_model(config, cheb_polys)

    evaluator = Evaluator(
        model=model,
        test_loader=dataloaders["test"],
        output_dir=output_dir,
        device=config["training"]["device"],
    )
    evaluator.load_checkpoint(args.model)
    y_true, y_pred = evaluator.predict()
    metrics = evaluator.evaluate(
        y_true,
        y_pred,
        config["training"]["metric_method"],
        config["training"]["missing_value"],
    )
    metrics_path = Path(output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    if config["output"].get("save_predictions", True):
        evaluator.save_predictions(y_true, y_pred)

    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.6f}")


if __name__ == "__main__":
    main()
