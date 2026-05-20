"""DataLoader 构造函数。"""

from __future__ import annotations

from torch.utils.data import DataLoader

from astgcn.data.dataset import ASTGCNDataset
from astgcn.data.io import load_pems_npz
from astgcn.data.scaler import StandardScaler
from astgcn.data.window import build_t0_list, split_t0_list


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
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """构建训练、验证和测试 DataLoader。

    :param data_path: PEMS ``.npz`` 数据路径。
    :param num_recent: recent 分支输入长度。
    :param num_days: daily 分支使用天数。
    :param num_weeks: weekly 分支使用周数。
    :param pred_len: 预测长度。
    :param batch_size: 批大小。
    :param points_per_day: 每天时间步数。
    :param target_dim: 目标特征编号。
    :param train_ratio: 训练集比例。
    :param val_ratio: 验证集比例。
    :param num_workers: DataLoader 子进程数量。
    :return: ``train_loader, val_loader, test_loader, scaler``。
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
    train_t0, val_t0, test_t0 = split_t0_list(t0_list, train_ratio, val_ratio)

    train_end_time = int(train_t0[-1]) + pred_len + 1
    scaler = StandardScaler()
    scaler.fit(data[:train_end_time])
    data_norm = scaler.transform(data)

    common = {
        "data": data_norm,
        "num_recent": num_recent,
        "num_days": num_days,
        "num_weeks": num_weeks,
        "pred_len": pred_len,
        "points_per_day": points_per_day,
        "target_dim": target_dim,
    }
    train_dataset = ASTGCNDataset(t0_list=train_t0, **common)
    val_dataset = ASTGCNDataset(t0_list=val_t0, **common)
    test_dataset = ASTGCNDataset(t0_list=test_t0, **common)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, scaler
