"""数据标准化工具。"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """按特征维度做标准化。

    标准化公式为 ``x_norm = (x - mean) / std``。均值和标准差只应使用训练集
    时间范围计算，避免验证集和测试集信息泄漏。
    """

    def __init__(self) -> None:
        """初始化空的标准化器。"""
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """根据训练数据计算均值和标准差。

        :param data: 训练数据，形状为 ``[T, N, F]``。
        """
        if data.ndim != 3:
            raise ValueError(f"data 必须是 [T, N, F]，当前 shape={data.shape}")
        self.mean = data.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = data.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std[self.std == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """标准化数据。

        :param data: 待标准化数据，形状为 ``[T, N, F]``。
        :return: 标准化后的数据，形状仍为 ``[T, N, F]``。
        """
        self._check_fitted()
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform_all(self, data: np.ndarray) -> np.ndarray:
        """反标准化完整特征数据。

        :param data: 标准化后的完整特征数据，最后一维必须能广播到 ``F``。
        :return: 原始尺度数据。
        """
        self._check_fitted()
        return (data * self.std + self.mean).astype(np.float32)

    def inverse_transform_target(
        self,
        data: np.ndarray,
        target_dim: int = 0,
    ) -> np.ndarray:
        """反标准化目标通道预测结果。

        :param data: 预测或标签数组，常见形状为 ``[B, N, T_p]`` 或 ``[N, T_p]``。
        :param target_dim: 目标特征编号。
        :return: 原始尺度下的预测或标签。
        """
        self._check_fitted()
        mean = float(self.mean[0, 0, target_dim])
        std = float(self.std[0, 0, target_dim])
        return (data * std + mean).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        """导出标准化器参数。

        :return: 包含 ``mean`` 和 ``std`` 的字典。
        """
        self._check_fitted()
        return {"mean": self.mean.copy(), "std": self.std.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """加载标准化器参数。

        :param state: 包含 ``mean`` 和 ``std`` 的字典。
        """
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("必须先调用 fit() 计算 mean 和 std")
