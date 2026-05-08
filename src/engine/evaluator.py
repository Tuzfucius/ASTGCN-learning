"""评估流程封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.metrics.evaluation import evaluate_prediction


class Evaluator:
    """ASTGCN 测试评估器。"""

    def __init__(self, model: Any, test_loader: Any, output_dir: str | Path, device: str) -> None:
        self.model = model
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(device)
        self.model.to(self.device)

    def load_checkpoint(self, model_path: str | Path) -> None:
        """加载模型权重。"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

    def predict(self) -> tuple[Any, Any]:
        """对测试集执行推理。"""
        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                y_pred_list.append(predictions.cpu().numpy())
                y_true_list.append(batch_y.cpu().numpy())

        if not y_true_list:
            raise ValueError("测试集 DataLoader 为空。")

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        return y_true, y_pred

    def evaluate(self, y_true: Any, y_pred: Any, metric_method: str, missing_value: float) -> dict[str, float]:
        """计算评估指标。"""
        return evaluate_prediction(y_true, y_pred, metric_method, missing_value)

    def save_predictions(self, y_true: Any, y_pred: Any) -> Path:
        """保存预测结果。"""
        output_path = self.output_dir / "predictions.npz"
        np.savez_compressed(output_path, y_true=y_true, y_pred=y_pred)
        return output_path

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """解析评估设备，CUDA 不可用时回退到 CPU。"""
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA 不可用，自动使用 CPU。")
            return torch.device("cpu")
        return torch.device(device)
