"""训练流程封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Trainer:
    """ASTGCN 训练器。

    本类只组织训练流程，不实现模型数学细节。
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        train_loader: Any,
        val_loader: Any,
        output_dir: str | Path,
        device: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.device = device

    def fit(self, epochs: int, start_epoch: int = 0) -> dict[str, Any]:
        """执行完整训练流程。

        TODO:
        - 循环 epoch。
        - 每轮调用 `train_one_epoch`。
        - 每轮调用 `validate`。
        - 如果验证 loss 更优，调用 `save_checkpoint`。
        """
        raise NotImplementedError("TODO: 实现 Trainer.fit。")

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个 epoch。

        TODO:
        - model.train()。
        - 遍历 train_loader。
        - 前向传播、计算 loss、反向传播、optimizer.step()。
        - 返回平均训练 loss。
        """
        raise NotImplementedError("TODO: 实现单轮训练。")

    def validate(self, epoch: int) -> float:
        """在验证集上计算 loss。

        TODO:
        - model.eval()。
        - 使用 no_grad。
        - 遍历 val_loader。
        - 返回平均验证 loss。
        """
        raise NotImplementedError("TODO: 实现验证流程。")

    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """保存模型权重。

        TODO:
        - 保存 `best.pt`。
        - 同时记录 epoch 和 val_loss。
        """
        raise NotImplementedError("TODO: 实现权重保存。")
