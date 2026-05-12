import numpy as np
import torch
from torch.utils.data import Dataset
from astgcn.data.window import (
    get_recent_data,
    get_days_data,
    get_weeks_data,
    get_target_data,
    build_t0_list
)

class ASTGCNDataset(Dataset):
    """
    ASTGCN 数据集。

    原始 data 形状:
        [T, N, F]

    每个样本输出:
        recent: [T_h, N, F]
        daily:  [T_d, N, F]
        weekly: [T_w, N, F]
        target: [N, T_p]
    """
    def __init__(
        self,
        data: np.ndarray,
        t0_list: np.ndarray, # 全部合法的当前时间列表
        num_recent: int,
        num_days: int,
        num_weeks: int,
        pred_len: int,
        points_per_day: int = 288,
        target_dim: int = 0, # 指定要预测的编号特征维度
    ):
        self.data = data.astype(np.float32)
        self.t0_list = t0_list.astype(np.int64)

        self.num_recent = num_recent
        self.num_days = num_days
        self.num_weeks = num_weeks
        self.pred_len = pred_len
        self.points_per_day = points_per_day
        self.target_dim = target_dim
        
    def __len__(self): # 返回数据集的样本数量
        return len(self.t0_list)
    
    def __getitem__(self, idx): 
        """
        根据样本编号 idx，返回一个 ASTGCN 训练样本。
        """
        t0 = int(self.t0_list[idx])

        recent = get_recent_data(
            data=self.data,
            t0=t0,
            num_recent=self.num_recent,
        )

        daily = get_days_data(
            data=self.data,
            t0=t0,
            num_days=self.num_days,
            pred_len=self.pred_len,
            points_per_day=self.points_per_day,
        )

        weekly = get_weeks_data(
            data=self.data,
            t0=t0,
            num_weeks=self.num_weeks,
            pred_len=self.pred_len,
            points_per_day=self.points_per_day,
        )

        target = get_target_data(
            data=self.data,
            t0=t0,
            pred_len=self.pred_len,
            target_dim=self.target_dim,
        )

        return {
            "recent": torch.from_numpy(recent),
            "daily": torch.from_numpy(daily),
            "weekly": torch.from_numpy(weekly),
            "target": torch.from_numpy(target),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
        
if __name__ == "__main__":
    data = np.random.randn(5000, 307, 3).astype(np.float32)

    t0_list = np.arange(3000, 3100)

    dataset = ASTGCNDataset(
        data=data,
        t0_list=t0_list,
        num_recent=12,
        num_days=1,
        num_weeks=1,
        pred_len=12,
        points_per_day=288,
        target_dim=0,
    )

    sample = dataset[0]

    print(sample["recent"].shape) # 期望输出 [12, 307, 3]
    print(sample["daily"].shape)
    print(sample["weekly"].shape)
    print(sample["target"].shape)
    print(sample["t0"])