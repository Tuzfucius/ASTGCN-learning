import numpy as np
import torch

from astgcn.models.astgcn import ASTGCN
from astgcn.models.ablation import AblationConfig


def test_astgcn_forward_shapes():
    batch_size = 2
    num_nodes = 8
    in_channels = 3
    num_timesteps = 12
    pred_len = 12
    cheb = np.stack(
        [np.eye(num_nodes, dtype=np.float32) for _ in range(3)],
        axis=0,
    )
    model = ASTGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        pred_len=pred_len,
        cheb_polynomials=cheb,
        recent_timesteps=num_timesteps,
        daily_timesteps=num_timesteps,
        weekly_timesteps=num_timesteps,
        num_blocks=1,
        hidden_channels=4,
    )
    x = torch.randn(batch_size, num_nodes, in_channels, num_timesteps)

    output = model(x, x, x)
    components = model(x, x, x, return_components=True)

    assert output.shape == (batch_size, num_nodes, pred_len)
    assert components["prediction"].shape == (batch_size, num_nodes, pred_len)
    assert components["recent"].shape == (batch_size, num_nodes, pred_len)
    assert components["daily"].shape == (batch_size, num_nodes, pred_len)
    assert components["weekly"].shape == (batch_size, num_nodes, pred_len)
    assert not any(key.endswith("cheb_polynomials") for key in model.state_dict())


def test_astgcn_ablation_single_branch_forward_shapes():
    batch_size = 2
    num_nodes = 8
    in_channels = 3
    num_timesteps = 12
    pred_len = 12
    cheb = np.stack(
        [np.eye(num_nodes, dtype=np.float32) for _ in range(3)],
        axis=0,
    )
    model = ASTGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        pred_len=pred_len,
        cheb_polynomials=cheb,
        recent_timesteps=num_timesteps,
        daily_timesteps=num_timesteps,
        weekly_timesteps=num_timesteps,
        num_blocks=1,
        hidden_channels=4,
        ablation_config=AblationConfig(
            use_daily=False,
            use_weekly=False,
            use_temporal_attention=False,
            use_spatial_attention=False,
            use_graph_conv=False,
            use_temporal_conv=False,
            fusion_mode="average",
        ),
    )
    x = torch.randn(batch_size, num_nodes, in_channels, num_timesteps)

    output = model(x, x, x)
    components = model(x, x, x, return_components=True)

    assert output.shape == (batch_size, num_nodes, pred_len)
    assert components["prediction"].shape == (batch_size, num_nodes, pred_len)
    assert components["recent"].shape == (batch_size, num_nodes, pred_len)
    assert "daily" not in components
    assert "weekly" not in components
