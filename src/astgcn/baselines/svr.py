"""SVR 基线适配器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SVRBaseline:
    """基于节点级样本展开的 SVR 基线。

    该模型把 ``recent [B, N, F, T]`` 展开为 ``[B*N, F*T]``，把目标
    ``target [B, N, T_p]`` 展开为 ``[B*N, T_p]``，用一个多输出 SVR 学习
    所有节点共享的非线性回归关系。为了避免 PEMS04 全量样本过大，训练时支持
    ``max_samples`` 抽样。
    """

    def __init__(
        self,
        pred_len: int,
        max_samples: int = 1024,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """初始化 SVR 基线。

        :param pred_len: 预测长度。
        :param max_samples: 最多使用多少个节点级样本训练。
        :param kernel: SVR kernel 名称。
        :param C: SVR 正则参数。
        :param epsilon: SVR epsilon 参数。
        :param random_state: 抽样随机种子。
        """
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR

        self.pred_len = pred_len
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = MultiOutputRegressor(SVR(kernel=kernel, C=C, epsilon=epsilon))
        self.is_fitted = False

    def fit_loader(self, data_loader: Any, max_batches: int | None = None) -> None:
        """从 DataLoader 中抽取节点级样本并训练 SVR。

        :param data_loader: 训练 DataLoader，batch 需包含 ``recent`` 与 ``target``。
        :param max_batches: 最多读取多少个 batch。
        """
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = self._batch_to_numpy(batch)
            features.append(x)
            targets.append(y)
            if sum(item.shape[0] for item in features) >= self.max_samples:
                break

        if not features:
            raise ValueError("SVR 没有收到可训练样本")

        x_all = np.concatenate(features, axis=0)
        y_all = np.concatenate(targets, axis=0)
        if x_all.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(x_all.shape[0], size=self.max_samples, replace=False)
            x_all = x_all[indices]
            y_all = y_all[indices]

        self.model.fit(x_all, y_all)
        self.is_fitted = True

    def predict_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """预测一个 batch。

        :param batch: DataLoader batch，需包含 ``recent``。
        :return: 预测结果，形状为 ``[B, N, pred_len]``。
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用 fit_loader() 训练 SVRBaseline")
        recent = batch["recent"].detach().cpu().numpy().astype(np.float32)
        batch_size, num_nodes, in_channels, num_timesteps = recent.shape
        x = recent.transpose(0, 1, 3, 2).reshape(batch_size * num_nodes, num_timesteps * in_channels)
        pred = self.model.predict(x).astype(np.float32)
        return pred.reshape(batch_size, num_nodes, self.pred_len)

    def _batch_to_numpy(self, batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        recent = batch["recent"].detach().cpu().numpy().astype(np.float32)
        target = batch["target"].detach().cpu().numpy().astype(np.float32)
        batch_size, num_nodes, in_channels, num_timesteps = recent.shape
        x = recent.transpose(0, 1, 3, 2).reshape(batch_size * num_nodes, num_timesteps * in_channels)
        y = target.reshape(batch_size * num_nodes, self.pred_len)
        return x, y


__all__ = ["SVRBaseline"]
