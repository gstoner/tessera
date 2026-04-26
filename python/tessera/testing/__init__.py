"""
tessera.testing — test utilities for Tessera Phase 1.

Exports:
    MockRankGroup — thread-based fake multi-rank group for testing
                    index_launch and collective ops without NCCL/MPI.
"""

from .mock_collective import MockRankGroup, MockRank, MockCollectiveError

__all__ = [
    "MockRankGroup",
    "MockRank",
    "MockCollectiveError",
]
