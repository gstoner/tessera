"""
conftest.py — Phase 6 shared fixtures.
"""
from __future__ import annotations

import pytest

# ---- runtime fixtures -------------------------------------------------------
from tessera.runtime import TesseraRuntime, DeviceKind, TsrStatus


@pytest.fixture
def mock_runtime():
    """A mock TesseraRuntime (no real library needed)."""
    rt = TesseraRuntime(mock=True)
    rt.init()
    yield rt
    rt.shutdown()


@pytest.fixture
def fresh_runtime():
    """Un-initialised mock runtime (caller must call init() themselves)."""
    return TesseraRuntime(mock=True)


# ---- diagnostics fixtures ---------------------------------------------------
from tessera.diagnostics import (
    ErrorReporter,
    ShapeInferenceEngine,
    DiagnosticLevel,
)


@pytest.fixture
def reporter():
    return ErrorReporter()


@pytest.fixture
def engine(reporter):
    return ShapeInferenceEngine(reporter)


# ---- benchmark fixtures -----------------------------------------------------
from benchmarks.benchmark_gemm import GEMMBenchmark
from benchmarks.benchmark_attention import FlashAttnBenchmark
from benchmarks.benchmark_collective import CollectiveBenchmark, CollectiveOp


@pytest.fixture
def gemm_bench():
    return GEMMBenchmark(dtype="bf16", peak_tflops=312.0, peak_membw_gbps=2000.0)


@pytest.fixture
def attn_bench():
    return FlashAttnBenchmark(causal=True, peak_tflops=312.0, peak_membw_gbps=2000.0)


@pytest.fixture
def coll_bench():
    return CollectiveBenchmark(peak_bw_gbps=600.0, latency_us=5.0)
