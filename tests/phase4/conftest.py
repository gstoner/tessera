"""
Phase 4 test fixtures — distributed training helpers.
"""
import pytest
import numpy as np
from tessera.testing.mock_collective import MockRankGroup


@pytest.fixture
def ranks_4():
    """4-rank group: dp=4."""
    return MockRankGroup(n=4, mesh_axes={"dp": 4})


@pytest.fixture
def ranks_8():
    """8-rank group: dp=4, tp=2."""
    return MockRankGroup(n=8, mesh_axes={"dp": 4, "tp": 2})


@pytest.fixture
def ranks_8_pp():
    """8-rank group: dp=2, tp=2, pp=2."""
    return MockRankGroup(n=8, mesh_axes={"dp": 2, "tp": 2, "pp": 2})


@pytest.fixture
def small_grad():
    """Small gradient tensor for collective tests."""
    return np.ones((64, 32), dtype=np.float32)
