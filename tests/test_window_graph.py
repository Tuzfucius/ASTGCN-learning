from __future__ import annotations

import numpy as np

from astgcn.data.graph import build_graph_data
from astgcn.data.window import (
    build_t0_list,
    get_days_data,
    get_recent_data,
    get_target_data,
    get_weeks_data,
    split_t0_list,
)


def test_window_segments_have_expected_shapes(synthetic_pems_like_data: np.ndarray) -> None:
    data = synthetic_pems_like_data
    t0 = 3000
    pred_len = 12

    recent = get_recent_data(data=data, t0=t0, num_recent=12)
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

    assert recent.shape == (12, 307, 3)
    assert daily.shape == (12, 307, 3)
    assert weekly.shape == (12, 307, 3)
    assert target.shape == (307, 12)


def test_build_t0_list_and_split_order(synthetic_pems_like_data: np.ndarray) -> None:
    data = synthetic_pems_like_data
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

    assert t0_list.ndim == 1
    assert len(t0_list) == len(train_t0) + len(val_t0) + len(test_t0)
    assert t0_list[0] == 7 * 288 - 1
    assert t0_list[-1] == data.shape[0] - 12 - 1
    assert train_t0[-1] < val_t0[0] < test_t0[0]


def test_build_graph_data_shapes(distance_csv_path) -> None:
    graph_data = build_graph_data(distance_csv_path, k_order=3)

    assert graph_data["distance_matrix"].shape == (307, 307)
    assert graph_data["adjacency_matrix"].shape == (307, 307)
    assert graph_data["normalized_laplacian"].shape == (307, 307)
    assert graph_data["scaled_laplacian"].shape == (307, 307)
    assert graph_data["chebyshev_polynomials"].shape == (3, 307, 307)
    assert np.allclose(graph_data["adjacency_matrix"], graph_data["adjacency_matrix"].T)
