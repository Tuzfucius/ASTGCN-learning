import pytest
import torch

from astgcn.baselines.rnn import RNNBaseline


def test_rnn_baseline_output_shape():
    batch_size = 2
    num_nodes = 307
    in_channels = 3
    num_timesteps = 12
    pred_len = 12

    recent = torch.randn(batch_size, num_nodes, in_channels, num_timesteps)

    model = RNNBaseline(
        in_channels=in_channels,
        hidden_size=64,
        pred_len=pred_len,
        num_layers=1,
        dropout=0.0,
    )

    output = model(recent)

    assert output.shape == (batch_size, num_nodes, pred_len)


def test_rnn_baseline_rejects_wrong_ndim():
    model = RNNBaseline(
        in_channels=3,
        hidden_size=64,
        pred_len=12,
    )

    wrong_recent = torch.randn(2, 307, 3)

    with pytest.raises(ValueError):
        model(wrong_recent)


def test_rnn_baseline_rejects_wrong_feature_dim():
    model = RNNBaseline(
        in_channels=3,
        hidden_size=64,
        pred_len=12,
    )

    wrong_recent = torch.randn(2, 307, 4, 12)

    with pytest.raises(ValueError):
        model(wrong_recent)
        
if __name__ == "__main__":
    test_rnn_baseline_output_shape()
    test_rnn_baseline_rejects_wrong_ndim()
    test_rnn_baseline_rejects_wrong_feature_dim()