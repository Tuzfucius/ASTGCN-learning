import numpy as np
from torch.utils.data import DataLoader

from astgcn.data.window import build_t0_list, split_t0_list
from astgcn.data.dataset import ASTGCNDataset


def main():
    data = np.random.randn(5000, 307, 3).astype(np.float32)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("recent:", batch["recent"].shape) # [B, T, N, F]
    print("daily:", batch["daily"].shape)
    print("weekly:", batch["weekly"].shape)
    print("target:", batch["target"].shape)
    print("t0:", batch["t0"].shape)


if __name__ == "__main__":
    main()