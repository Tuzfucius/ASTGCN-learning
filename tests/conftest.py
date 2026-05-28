from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pems04_npz_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "PEMS04" / "pems04.npz"


@pytest.fixture(scope="session")
def distance_csv_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "PEMS04" / "distance.csv"


@pytest.fixture
def synthetic_pems_like_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((5000, 307, 3)).astype(np.float32)
