"""Host-free exact-M graph-pool lifetime and capacity tests."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tessera.compiler import rocm_mxfp4_graph_pipeline as module


class _Session:
    def __init__(self, package: object, payload: object, m: int) -> None:
        self.package = package
        self.payload = payload
        self.m = m
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_exact_m_pool_reuses_shapes_and_invalidates_released_session() -> None:
    payload = object()
    compiled: list[int] = []

    def package_for_m(m: int, _: object) -> object:
        compiled.append(m)
        return object()

    with patch.object(module, "PackedFoldedGraphPipeline", _Session):
        with module.PackedFoldedGraphPipelinePool(
            payload, package_for_m, max_shapes=2,
        ) as pool:
            first = pool.get(65)
            second = pool.get(129)
            assert pool.get(65) is first
            assert compiled == [65, 129]
            assert pool.receipt()["active_m"] == [65, 129]
            with pytest.raises(RuntimeError, match="shape budget exhausted"):
                pool.get(257)
            pool.release(65)
            assert first.closed
            third = pool.get(257)
            assert third is not first
            assert pool.receipt()["active_m"] == [129, 257]
        assert second.closed and third.closed
        with pytest.raises(RuntimeError, match="pool is closed"):
            pool.get(65)
        pool.close()


def test_exact_m_pool_refuses_zero_capacity() -> None:
    with pytest.raises(ValueError, match="max_shapes"):
        module.PackedFoldedGraphPipelinePool(object(), lambda *_: object(), max_shapes=0)
