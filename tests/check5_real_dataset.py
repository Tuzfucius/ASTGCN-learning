from torch.utils.data import DataLoader

from astgcn.data.io import load_pems_npz
from astgcn.data.window import build_t0_list, split_t0_list
from astgcn.data.scaler import StandardScaler
from astgcn.data.dataset import ASTGCNDataset


def main():
    data = load_pems_npz("data/raw/PEMS04/pems04.npz")

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

    # 只用训练集对应的时间范围 fit scaler
    # 标准化器只看训练集覆盖到的时间范围，避免数据泄漏
    train_end_time = train_t0[-1] + 12 + 1

    scaler = StandardScaler()
    scaler.fit(data[:train_end_time])

    data_norm = scaler.transform(data)

    train_dataset = ASTGCNDataset(
        data=data_norm,
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
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("recent:", batch["recent"].shape)
    print("daily:", batch["daily"].shape)
    print("weekly:", batch["weekly"].shape)
    print("target:", batch["target"].shape)
    print("t0:", batch["t0"].shape)

    print("train samples:", len(train_dataset))
    print("val samples:", len(val_t0))
    print("test samples:", len(test_t0))


if __name__ == "__main__":
    main()