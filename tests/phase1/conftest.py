"""
tests/phase1/conftest.py — shared fixtures for Phase 1 tests.

Provides:
    mesh4        — MeshSpec with dp=4
    mesh2x2      — MeshSpec with dp=2, tp=2
    group4       — MockRankGroup(n=4, mesh_axes={"dp": 4})
    group2x2     — MockRankGroup(n=4, mesh_axes={"dp": 2, "tp": 2})
    rect_3d      — Rect((4, 128, 256))
    block_dp_tp  — Block(mesh_axes=("dp", "tp"))
"""

import pytest
import numpy as np

import tessera
from tessera.distributed.shard import MeshSpec
from tessera.distributed.domain import Rect, Block, Replicated
from tessera.distributed.array import DistributedArray
from tessera.testing import MockRankGroup


# ─────────────────────────────────────────────────────────────────────────────
# Mesh fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mesh4() -> MeshSpec:
    """Single-axis mesh: dp=4."""
    return MeshSpec({"dp": 4})


@pytest.fixture
def mesh2x2() -> MeshSpec:
    """Two-axis mesh: dp=2, tp=2."""
    return MeshSpec({"dp": 2, "tp": 2})


@pytest.fixture
def mesh_tp8() -> MeshSpec:
    """Single-axis tensor-parallel mesh: tp=8."""
    return MeshSpec({"tp": 8})


# ─────────────────────────────────────────────────────────────────────────────
# MockRankGroup fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def group4() -> MockRankGroup:
    """4-rank group with single dp axis."""
    return MockRankGroup(n=4, mesh_axes={"dp": 4})


@pytest.fixture
def group2x2() -> MockRankGroup:
    """4-rank group with dp=2, tp=2."""
    return MockRankGroup(n=4, mesh_axes={"dp": 2, "tp": 2})


@pytest.fixture
def group1() -> MockRankGroup:
    """Single-rank group for sanity checks."""
    return MockRankGroup(n=1, mesh_axes={"dp": 1})


# ─────────────────────────────────────────────────────────────────────────────
# Domain / Distribution fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rect_3d() -> Rect:
    return Rect((4, 128, 256))


@pytest.fixture
def rect_2d() -> Rect:
    return Rect((8, 256))


@pytest.fixture
def block_dp_tp() -> Block:
    return Block(mesh_axes=("dp", "tp"))


@pytest.fixture
def block_dp() -> Block:
    return Block(mesh_axes=("dp",))


@pytest.fixture
def replicated() -> Replicated:
    return Replicated()


# ─────────────────────────────────────────────────────────────────────────────
# DistributedArray fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bf16_array_3d(rect_3d, block_dp) -> DistributedArray:
    """4×128×256 bf16 array partitioned over dp."""
    return DistributedArray.from_domain(rect_3d, dtype="bf16", distribution=block_dp)


@pytest.fixture
def fp32_array_2d(rect_2d, block_dp) -> DistributedArray:
    """8×256 fp32 array partitioned over dp."""
    return DistributedArray.from_domain(rect_2d, dtype="fp32", distribution=block_dp)
