"""评估流程封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Evaluator:
    """ASTGCN 测试评估器。"""

    def __init__(self, model: Any, test_loader: Any, output_dir: str | Path, device: str) -> None:
        self.model = model
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.device = device

    def load_checkpoint(self, model_path: str | Path) -> None:
        """加载模型权重。

        TODO:
        - 使用 torch.load 读取权重。
        - 调用 model.load_state_dict。
        """
        raise NotImplementedError("TODO: 实现模型权重加载。")

    def predict(self) -> tuple[Any, Any]:
        """对测试集执行推理。

        TODO:
        - model.eval()。
        - 使用 no_grad。
        - 拼接所有 batch 的预测值和真实值。
        - 返回 (y_true, y_pred)。
        """
        raise NotImplementedError("TODO: 实现测试集预测。")

    def evaluate(self, y_true: Any, y_pred: Any, metric_method: str, missing_value: float) -> dict[str, float]:
        """计算评估指标。

        TODO:
        - 调用 `src.metrics.evaluation.evaluate_prediction`。
        - 返回 MAE/RMSE/MAPE 等指标。
        """
        raise NotImplementedError("TODO: 实现指标计算。")

    def save_predictions(self, y_true: Any, y_pred: Any) -> Path:
        """保存预测结果。

        TODO:
        - 使用 np.savez_compressed 保存 y_true 和 y_pred。
        - 文件名建议为 `predictions.npz`。
        """
        raise NotImplementedError("TODO: 实现预测结果保存。")
