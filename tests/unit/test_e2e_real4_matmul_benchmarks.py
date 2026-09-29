from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from benchmarks.rocm import benchmark_rocm_e2e_real_matmul as rocm_bench

from benchmarks.rocm.benchmark_rocm_e2e_real_matmul import decide
from benchmarks.x86.benchmark_x86_e2e_real_matmul import _summarize, verdict


def _x86_row(*, correct: bool, non_regression: bool) -> dict:
    return {
        "correctness": {"passed": correct},
        "timing": {"non_regression_10pct": non_regression},
    }


def test_x86_e2e_real4_requires_correctness_and_the_existing_ratchet() -> None:
    assert verdict([_x86_row(correct=True, non_regression=True)]) == "promote"
    assert verdict([_x86_row(correct=True, non_regression=False)]) == "retain"
    assert verdict([_x86_row(correct=False, non_regression=True)]) == "reject"
    summary = _summarize([1.0, 1.0, 1.0], [1.09, 1.09, 1.09])
    assert summary["non_regression_10pct"] is True
    assert summary["scheduled_over_production"] == 1.09


def test_rocm_e2e_real4_never_promotes_host_wall_evidence() -> None:
    assert decide(correct=True, tflops=12.5, direct_ratio=0.90, selector_eligible=False) == "retain"
    assert decide(correct=True, tflops=12.5, direct_ratio=0.90, selector_eligible=True) == "promote"
    assert decide(correct=True, tflops=8.01, direct_ratio=1.0, selector_eligible=True) == "reject"
    assert decide(correct=False, tflops=20.0, direct_ratio=1.0, selector_eligible=True) == "reject"


def test_rocm_scheduled_raw_launcher_uses_packaged_entry_symbol(monkeypatch) -> None:
    a = np.ones((16, 16), dtype=np.float16)
    b = np.ones((16, 16), dtype=np.float16)
    monkeypatch.setattr(rocm_bench, "_inputs", lambda case: (a, b, None))
    artifact = SimpleNamespace(
        macro_tile_m=32, macro_tile_n=64,
        graph_digest="graph", schedule_ir_digest="schedule",
        schedule_digest="decision", tile_digest="tile",
    )
    package = SimpleNamespace(
        image=SimpleNamespace(payload=b"scheduled", target_ir_digest="target",
                              image_digest="image"),
        descriptor=SimpleNamespace(entry_symbol="tessera_rocm_matmul_1234"),
    )
    monkeypatch.setattr(
        rocm_bench, "_compile_scheduled",
        lambda m, n, k: (artifact, package, 1.0),
    )
    symbols = []

    class FakeDeviceCase:
        def __init__(self, hip, payload, case, tile, lhs, rhs, bias, *,
                     entry_symbol="gemm"):
            symbols.append((payload, entry_symbol))

        def download(self):
            return a.astype(np.float32) @ b.astype(np.float32)

        def close(self):
            pass

    monkeypatch.setattr(rocm_bench, "DeviceCase", FakeDeviceCase)
    row = rocm_bench._correctness_row(None, b"direct", (16, 16, 16))
    assert row["passed"]
    assert symbols == [
        (b"direct", "gemm"),
        (b"scheduled", package.descriptor.entry_symbol),
    ]
