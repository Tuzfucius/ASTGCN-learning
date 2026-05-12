from astgcn.data.dataloader import build_dataloaders


def main():
    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        data_path="data/raw/PEMS04/pems04.npz",
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        batch_size=8,
        points_per_day=288,
        target_dim=0,
    )

    batch = next(iter(train_loader))

    print("recent:", batch["recent"].shape)
    print("daily:", batch["daily"].shape)
    print("weekly:", batch["weekly"].shape)
    print("target:", batch["target"].shape)

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))


if __name__ == "__main__":
    main()