"""ASTGCN 测试评估入口脚本。"""

from __future__ import annotations

import argparse

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
    return parser.parse_args()


def main() -> None:
    """执行测试评估。

    TODO:
    - 读取配置。
    - 加载测试 DataLoader。
    - 构造模型。
    - 加载权重。
    - 执行预测。
    - 计算并保存指标。
    """
    args = parse_args()
    config = load_config(args.config)
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
    evaluator.evaluate(
        y_true,
        y_pred,
        config["training"]["metric_method"],
        config["training"]["missing_value"],
    )
    evaluator.save_predictions(y_true, y_pred)


if __name__ == "__main__":
    main()
