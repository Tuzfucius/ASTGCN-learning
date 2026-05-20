import torch

from astgcn.baselines.historical_average import HistoricalAverage
from astgcn.baselines.lstm import LSTMBaseline


def test_historical_average_shape():
    x = torch.randn(2, 8, 3, 12)
    model = HistoricalAverage(pred_len=6, target_dim=0)
    assert model(x).shape == (2, 8, 6)


def test_lstm_baseline_shape():
    x = torch.randn(2, 8, 3, 12)
    model = LSTMBaseline(in_channels=3, hidden_size=4, pred_len=6)
    assert model(x).shape == (2, 8, 6)
