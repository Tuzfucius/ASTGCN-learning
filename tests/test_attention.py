from __future__ import annotations

import torch

from astgcn.models.attention import (
    SpatialAttention,
    TemporalAttention,
    apply_temporal_attention,
)


def test_temporal_attention_shapes_and_normalization() -> None:
    torch.manual_seed(0)
    batch_size = 4
    num_nodes = 307
    in_channels = 3
    num_timesteps = 12

    x = torch.randn(batch_size, num_nodes, in_channels, num_timesteps)
    temporal_attention = TemporalAttention(
        num_nodes=num_nodes,
        in_channels=in_channels,
        num_timesteps=num_timesteps,
    )

    e = temporal_attention(x)
    out = apply_temporal_attention(x, e)

    assert e.shape == (batch_size, num_timesteps, num_timesteps)
    assert out.shape == x.shape
    assert torch.allclose(e.sum(dim=-1), torch.ones(batch_size, num_timesteps), atol=1e-5)


def test_spatial_attention_shapes_and_normalization() -> None:
    torch.manual_seed(0)
    batch_size = 4
    num_nodes = 307
    in_channels = 3
    num_timesteps = 12

    x = torch.randn(batch_size, num_nodes, in_channels, num_timesteps)
    spatial_attention = SpatialAttention(
        num_nodes=num_nodes,
        in_channels=in_channels,
        num_timesteps=num_timesteps,
    )

    s = spatial_attention(x)

    assert s.shape == (batch_size, num_nodes, num_nodes)
    assert torch.allclose(s.sum(dim=-1), torch.ones(batch_size, num_nodes), atol=1e-5)
