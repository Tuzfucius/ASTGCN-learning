"""模型预测与结果保存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _split_output(output: Any) -> tuple[torch.Tensor, Any | None]:
    if isinstance(output, dict):
        pred = output.get("prediction", output.get("pred"))
        components = output.get("components")
        if components is None:
            components = {
                key: value
                for key, value in output.items()
                if key not in {"prediction", "pred"}
            }
        return pred, components
    if isinstance(output, (tuple, list)):
        pred = output[0]
        components = output[1] if len(output) > 1 else None
        return pred, components
    return output, None


def _to_numpy_component(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_component(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy_component(item) for item in value]
    return value


def _flatten_components(components: list[Any]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, list[np.ndarray]] = {}
    for item in components:
        item = _to_numpy_component(item)
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    arrays.setdefault(f"component_{key}", []).append(value)
        elif isinstance(item, (list, tuple)):
            for idx, value in enumerate(item):
                if isinstance(value, np.ndarray):
                    arrays.setdefault(f"component_{idx}", []).append(value)
        elif isinstance(item, np.ndarray):
            arrays.setdefault("components", []).append(item)
    return {key: np.concatenate(value, axis=0) for key, value in arrays.items() if value}


class Predictor:
    """生成预测结果并保存为 npz 文件。

    参数:
        model: 待预测模型，前向接口为 ``forward(recent, daily, weekly, return_components=False)``。
        data_loader: 预测 DataLoader，batch 需包含 ``recent``、``daily``、``weekly``、``target``、``t0``。
        device: 推理设备。
    """

    def __init__(self, model: nn.Module, data_loader: Any, device: str | torch.device = "cpu"):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        output_path: str | Path,
        save_components: bool = False,
        max_batches: int | None = None,
    ) -> Path:
        """执行预测并保存结果。

        参数:
            output_path: ``.npz`` 输出路径。
            save_components: 是否请求并保存模型分支输出或中间组件。
            max_batches: smoke test 使用的最大 batch 数；为 ``None`` 时遍历完整数据集。

        返回:
            保存完成的 ``.npz`` 文件路径。
        """
        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        t0_values: list[np.ndarray] = []
        components: list[Any] = []

        for batch_idx, batch in enumerate(self.data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch(batch, self.device)
            output = self.model(
                batch["recent"],
                batch["daily"],
                batch["weekly"],
                return_components=save_components,
            )
            pred, component = _split_output(output)
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            t0_values.append(batch["t0"].detach().cpu().numpy())
            if save_components and component is not None:
                components.append(component)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_data: Dict[str, np.ndarray] = {
            "prediction": np.concatenate(predictions, axis=0) if predictions else np.empty((0,)),
            "target": np.concatenate(targets, axis=0) if targets else np.empty((0,)),
            "t0": np.concatenate(t0_values, axis=0) if t0_values else np.empty((0,), dtype=np.int64),
        }
        if save_components and components:
            save_data.update(_flatten_components(components))
        np.savez_compressed(output, **save_data)
        return output
