from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from astgcn.data.dataset import ASTGCNDataset
from astgcn.data.window import build_t0_list, split_t0_list


def test_dataset_and_dataloader_batch_shapes(synthetic_pems_like_data: np.ndarray) -> None:
    data = synthetic_pems_like_data
    t0_list = build_t0_list(
        num_samples=data.shape[0],
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        points_per_day=288,
    )
    train_t0, _, _ = split_t0_list(
        t0_list=t0_list,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    train_dataset = ASTGCNDataset(
        data=data,
        t0_list=train_t0,
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        points_per_day=288,
        target_dim=0,
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))

    assert len(train_dataset) == len(train_t0)
    assert batch["recent"].shape == (4, 307, 3, 12)
    assert batch["daily"].shape == (4, 307, 3, 12)
    assert batch["weekly"].shape == (4, 307, 3, 12)
    assert batch["target"].shape == (4, 307, 12)
    assert batch["t0"].shape == (4,)
    assert batch["t0"].dtype == torch.long
