from torch.utils.data import DataLoader

from astgcn.data.io import load_pems_npz
from astgcn.data.window import build_t0_list, split_t0_list
from astgcn.data.scaler import StandardScaler
from astgcn.data.dataset import ASTGCNDataset


def build_dataloaders(
    data_path: str,
    num_recent: int,
    num_days: int,
    num_weeks: int,
    pred_len: int,
    batch_size: int,
    points_per_day: int = 288,
    target_dim: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
):
    """
    构建 train / val / test 三个 DataLoader。

    返回:
        train_loader
        val_loader
        test_loader
        scaler
    """

    data = load_pems_npz(data_path)

    t0_list = build_t0_list(
        num_samples=data.shape[0],
        num_recent=num_recent,
        num_days=num_days,
        num_weeks=num_weeks,
        pred_len=pred_len,
        points_per_day=points_per_day,
    )

    train_t0, val_t0, test_t0 = split_t0_list(
        t0_list=t0_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # 只用训练集时间范围 fit scaler
    train_end_time = train_t0[-1] + pred_len + 1

    scaler = StandardScaler()
    scaler.fit(data[:train_end_time])

    data_norm = scaler.transform(data)

    train_dataset = ASTGCNDataset(
        data=data_norm,
        t0_list=train_t0,
        num_recent=num_recent,
        num_days=num_days,
        num_weeks=num_weeks,
        pred_len=pred_len,
        points_per_day=points_per_day,
        target_dim=target_dim,
    )

    val_dataset = ASTGCNDataset(
        data=data_norm,
        t0_list=val_t0,
        num_recent=num_recent,
        num_days=num_days,
        num_weeks=num_weeks,
        pred_len=pred_len,
        points_per_day=points_per_day,
        target_dim=target_dim,
    )

    test_dataset = ASTGCNDataset(
        data=data_norm,
        t0_list=test_t0,
        num_recent=num_recent,
        num_days=num_days,
        num_weeks=num_weeks,
        pred_len=pred_len,
        points_per_day=points_per_day,
        target_dim=target_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, scaler