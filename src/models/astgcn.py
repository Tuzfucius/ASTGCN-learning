"""ASTGCN 完整模型。"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

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
        if nb_block <= 0:
            raise ValueError("nb_block 必须大于 0。")
        if time_strides <= 0:
            raise ValueError("time_strides 必须大于 0。")

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
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ASTGCNBlock(
                in_channels=in_channels,
                k=k,
                nb_chev_filter=nb_chev_filter,
                nb_time_filter=nb_time_filter,
                time_strides=time_strides,
                cheb_polynomials=cheb_polynomials,
                num_of_vertices=num_of_vertices,
                num_of_timesteps=len_input,
            )
        )

        reduced_len_input = _conv_time_output_length(len_input, time_strides)
        for _ in range(nb_block - 1):
            self.blocks.append(
                ASTGCNBlock(
                    in_channels=nb_time_filter,
                    k=k,
                    nb_chev_filter=nb_chev_filter,
                    nb_time_filter=nb_time_filter,
                    time_strides=1,
                    cheb_polynomials=cheb_polynomials,
                    num_of_vertices=num_of_vertices,
                    num_of_timesteps=reduced_len_input,
                )
            )

        self.final_conv = nn.Conv2d(reduced_len_input, num_for_predict, kernel_size=(1, nb_time_filter))
        self.reset_parameters()

    def forward(self, x: Any) -> Any:
        """执行完整模型前向传播。"""
        for block in self.blocks:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return output.permute(0, 2, 1)

    def reset_parameters(self) -> None:
        """初始化模型参数。"""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)


def build_astgcn_model(config: dict[str, Any], cheb_polynomials: list[Any]) -> ASTGCN:
    """根据配置构造 ASTGCN 模型。"""
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


def _conv_time_output_length(len_input: int, time_strides: int) -> int:
    """计算 block 中时间卷积后的时间步数。"""
    return (len_input - 1) // time_strides + 1
