import torch

from astgcn.baselines.historical_average import HistoricalAverage
from astgcn.baselines.gru import GRUBaseline
from astgcn.baselines.lstm import LSTMBaseline
from astgcn.baselines.svr import SVRBaseline


def test_historical_average_shape():
    x = torch.randn(2, 8, 3, 12)
    model = HistoricalAverage(pred_len=6, target_dim=0)
    assert model(x).shape == (2, 8, 6)


def test_lstm_baseline_shape():
    x = torch.randn(2, 8, 3, 12)
    model = LSTMBaseline(in_channels=3, hidden_size=4, pred_len=6)
    assert model(x).shape == (2, 8, 6)


def test_gru_baseline_shape():
    x = torch.randn(2, 8, 3, 12)
    model = GRUBaseline(in_channels=3, hidden_size=4, pred_len=6)
    assert model(x).shape == (2, 8, 6)


def test_svr_baseline_small_fit_predict():
    batch = {
        "recent": torch.randn(2, 4, 3, 6),
        "target": torch.randn(2, 4, 2),
    }
    loader = [batch]
    model = SVRBaseline(pred_len=2, max_samples=8)
    model.fit_loader(loader)
    assert model.predict_batch(batch).shape == (2, 4, 2)
