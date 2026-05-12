import numpy as np

class StandardScaler:
    """
    标准化器。

    x_norm = (x - mean) / std
    x = x_norm * std + mean
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray):
        """
        根据训练集计算均值和标准差。

        :param data: 训练数据，形状为 [T, N, F]
        """
        self.mean = data.mean(axis=(0, 1), keepdims=True) 
        self.std = data.std(axis=(0, 1), keepdims=True)
        # 压缩时间维度和节点维度，保留特征维度，得到形状为 [1, 1, F] 的均值
        # 使用 keepdims=True 保持维度不变，为了后续广播操作
        
        self.std[self.std == 0] = 1.0  # 找到全部为0的位置，并将其设置为1.0，避免除以0导致的错误
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        使用计算得到的均值和标准差对数据进行标准化。

        :param data: 待标准化的数据，形状为 [T, N, F]
        :return: 标准化后的数据，形状为 [T, N, F]
        """
        if self.mean is None or self.std is None:
            raise ValueError("必须先调用 fit 方法计算均值和标准差")
        return (data - self.mean) / self.std
    
    def inverse_transform_all(self, data: np.ndarray) -> np.ndarray:
        """
        将标准化后的数据还原为原始数据。

        :param data: 标准化后的数据，形状为 [T, N, F]
        :return: 还原后的原始数据，形状为 [T, N, F]
        """
        if self.mean is None or self.std is None:
            raise ValueError("必须先调用 fit 方法计算均值和标准差")
        return data * self.std + self.mean
    
    def inverse_transform_target(
        self,
        data: np.ndarray,
        target_dim: int = 0,
    ) -> np.ndarray:
        """
        反标准化预测结果。

        :param data: 预测结果，常见形状为 [B, N, Tp] 或 [N, Tp]
        :param target_dim: 目标特征维度，通常 flow 是 0
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("必须先调用 fit()")

        mean = float(self.mean[0, 0, target_dim])
        std = float(self.std[0, 0, target_dim])

        return data * std + mean