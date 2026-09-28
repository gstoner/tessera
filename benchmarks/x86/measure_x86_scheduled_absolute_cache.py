"""Time first and exact-repeat native x86 trunc packaging on the owning CPU."""

from __future__ import annotations

import json
import statistics
import time

from tessera.compiler import scheduled_absolute as scheduled, x86_native
from tests.unit.test_scheduled_absolute import absolute_module


def package(shape):
    module = absolute_module(shape)
    module.functions[0].body[0].op_name = "tessera.trunc"
    return x86_native.package_elementwise(module, pipeline_name="tessera-lower-to-x86")


def measure(shape):
    begin = time.perf_counter_ns()
    first = package(shape)
    first_ms = (time.perf_counter_ns() - begin) / 1e6
    repeats = []
    for _ in range(7):
        begin = time.perf_counter_ns()
        repeated = package(shape)
        repeats.append((time.perf_counter_ns() - begin) / 1e6)
        assert repeated.image.image_digest == first.image.image_digest
        assert repeated.descriptor == first.descriptor
    return {"shape": list(shape), "first_ms": first_ms,
            "repeat_median_ms": statistics.median(repeats)}


if __name__ == "__main__":
    scheduled._LOWER_CACHE.clear()
    x86_native._SCHEDULED_UNARY_PACKAGE_CACHE.clear()
    rows = [measure(shape) for shape in ((51,), (3, 17), (2, 3, 17), (5, 23))]
    print(json.dumps(rows, indent=2))
