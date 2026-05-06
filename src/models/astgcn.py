"""ASTGCN 完整模型。"""

from __future__ import annotations

from typing import Any

try:
    from torch import nn
except ImportError:
    class _FallbackModule:
        """未安装 PyTorch 时的占位基类。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()

from src.models.layers import ASTGCNBlock


class ASTGCN(nn.Module):
    """ASTGCN recent component 模型。

    输入:
        x: (B, N, F, T)
    输出:
        y_hat: (B, N, T_pred)
    """

    def __init__(
        self,
        nb_block: int,
        in_channels: int,
        k: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        cheb_polynomials: list[Any],
        num_for_predict: int,
        len_input: int,
        num_of_vertices: int,
    ) -> None:
        super().__init__()
        # TODO: 根据 nb_block 堆叠 ASTGCNBlock。
        # TODO: 定义 final convolution，将中间特征映射为未来 num_for_predict 步。
        self.nb_block = nb_block
        self.in_channels = in_channels
        self.k = k
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.cheb_polynomials = cheb_polynomials
        self.num_for_predict = num_for_predict
        self.len_input = len_input
        self.num_of_vertices = num_of_vertices

    def forward(self, x: Any) -> Any:
        """执行完整模型前向传播。"""
        # TODO: 将 x 依次送入每个 ASTGCNBlock。
        # TODO: 使用 final convolution 输出 (B, N, T_pred)。
        raise NotImplementedError("TODO: 实现 ASTGCN.forward。")


def build_astgcn_model(config: dict[str, Any], cheb_polynomials: list[Any]) -> ASTGCN:
    """根据配置构造 ASTGCN 模型。

    TODO:
    - 从 config 中取 model/task/data 参数。
    - 创建 ASTGCN 实例。
    - 后续可以在这里做参数初始化。
    """
    model_config = config["model"]
    task_config = config["task"]
    data_config = config["data"]
    return ASTGCN(
        nb_block=model_config["nb_block"],
        in_channels=task_config["in_channels"],
        k=model_config["K"],
        nb_chev_filter=model_config["nb_chev_filter"],
        nb_time_filter=model_config["nb_time_filter"],
        time_strides=model_config["time_strides"],
        cheb_polynomials=cheb_polynomials,
        num_for_predict=task_config["num_for_predict"],
        len_input=task_config["len_input"],
        num_of_vertices=data_config["num_of_vertices"],
    )
