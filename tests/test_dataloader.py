from __future__ import annotations

import pytest

from astgcn.data.dataloader import build_dataloaders


@pytest.mark.slow
@pytest.mark.integration
def test_build_dataloaders_returns_expected_batch_shapes(pems04_npz_path) -> None:
    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        data_path=str(pems04_npz_path),
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        batch_size=8,
        points_per_day=288,
        target_dim=0,
    )

    batch = next(iter(train_loader))

    assert batch["recent"].shape == (8, 307, 3, 12)
    assert batch["daily"].shape == (8, 307, 3, 12)
    assert batch["weekly"].shape == (8, 307, 3, 12)
    assert batch["target"].shape == (8, 307, 12)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert scaler.mean.shape == (1, 1, 3)
    assert scaler.std.shape == (1, 1, 3)
