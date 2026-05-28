from __future__ import annotations

import numpy as np

from astgcn.data.io import load_pems_npz


def test_load_pems_npz_shape_and_dtype(pems04_npz_path) -> None:
    data = load_pems_npz(pems04_npz_path)

    assert data.shape == (16992, 307, 3)
    assert data.dtype == np.float32
