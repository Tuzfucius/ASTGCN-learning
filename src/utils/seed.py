"""随机种子工具。"""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 的随机种子。

    TODO:
    - 引入 NumPy 后固定 `np.random.seed(seed)`。
    - 引入 PyTorch 后固定 `torch.manual_seed(seed)`。
    - 如果使用 CUDA，固定 `torch.cuda.manual_seed_all(seed)`。
    - 根据需要设置 cudnn deterministic。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
