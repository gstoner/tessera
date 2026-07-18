"""Portable contracts for the APPLE-EPILOGUE-1 measurement producer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "apple_gpu" / "benchmark_epilogue_routes.py"


def _module():
    spec = importlib.util.spec_from_file_location("apple_epilogue_benchmark", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shape_parser_requires_positive_matmul_shape():
    benchmark = _module()
    assert benchmark._shape("65x63x67") == (65, 63, 67)
    for invalid in ("64x64", "64x0x64", "64xbadx64"):
        with pytest.raises(ValueError):
            benchmark._shape(invalid)


def test_dispatch_combiner_sums_only_complete_native_device_intervals():
    benchmark = _module()
    output = np.ones((2, 2), dtype=np.float32)
    complete = benchmark._combine_dispatches(output, [
        {"device_time_ns": 10, "timing_source": "metal_command_buffer_interval",
         "resources": None},
        {"device_time_ns": 20, "timing_source": "metal_kernel_interval",
         "resources": {"threadgroup": [32, 1, 1]}},
    ], route="unfused", native_flags=[True, True])
    assert complete.native_dispatched
    assert complete.device_time_ns == 30
    assert complete.resources["dispatch_count"] == 2
    assert len(complete.resources["dispatches"]) == 2

    partial = benchmark._combine_dispatches(output, [
        {"device_time_ns": 10, "timing_source": "metal_command_buffer_interval",
         "resources": None},
        {"device_time_ns": None, "timing_source": None, "resources": None},
    ], route="unfused", native_flags=[True, True])
    assert partial.native_dispatched
    assert partial.device_time_ns is None
