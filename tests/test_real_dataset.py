from __future__ import annotations

import numpy as np
import pytest

from astgcn.data.io import load_pems_npz
from astgcn.data.scaler import StandardScaler
from astgcn.data.window import build_t0_list, split_t0_list


@pytest.mark.slow
@pytest.mark.integration
def test_real_dataset_split_and_scaler(pems04_npz_path) -> None:
    data = load_pems_npz(pems04_npz_path)

    t0_list = build_t0_list(
        num_samples=data.shape[0],
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        points_per_day=288,
    )
    train_t0, val_t0, test_t0 = split_t0_list(
        t0_list=t0_list,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    train_end_time = int(train_t0[-1]) + 12 + 1
    scaler = StandardScaler()
    scaler.fit(data[:train_end_time])
    data_norm = scaler.transform(data)

    assert data_norm.shape == data.shape
    assert data_norm.dtype == np.float32
    assert scaler.mean.shape == (1, 1, 3)
    assert scaler.std.shape == (1, 1, 3)
    assert len(train_t0) > 0
    assert len(val_t0) > 0
    assert len(test_t0) > 0
    assert train_t0[-1] < val_t0[0] < test_t0[0]
