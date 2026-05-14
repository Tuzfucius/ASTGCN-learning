import numpy as np

from astgcn.data.window import (
    get_recent_data,
    get_days_data,
    get_weeks_data,
    get_target_data,
)

from astgcn.data.graph import get_distance_matrix


def main():
    # 假数据：[时间步, 节点数, 特征数]
    data = np.random.randn(5000, 307, 3).astype(np.float32)

    t0 = 3000
    pred_len = 12

    recent = get_recent_data(
        data=data,
        t0=t0,
        num_recent=12,
    )

    daily = get_days_data(
        data=data,
        t0=t0,
        num_days=1,
        pred_len=pred_len,
        points_per_day=288,
    )

    weekly = get_weeks_data(
        data=data,
        t0=t0,
        num_weeks=1,
        pred_len=pred_len,
        points_per_day=288,
    )

    target = get_target_data(
        data=data,
        t0=t0,
        pred_len=pred_len,
        target_dim=0,
    )

    print("recent:", recent.shape)
    print("daily:", daily.shape)
    print("weekly:", weekly.shape)
    print("target:", target.shape)

    assert recent.shape == (12, 307, 3)
    assert daily.shape == (12, 307, 3)
    assert weekly.shape == (12, 307, 3)
    assert target.shape == (307, 12)

    print("window.py 测试通过")


if __name__ == "__main__":
    main()