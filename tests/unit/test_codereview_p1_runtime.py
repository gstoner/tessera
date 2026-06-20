"""P1 runtime / IO regressions from the full-source code review.

  * TesseraRuntime — an explicitly requested library (constructor arg or
    TESSERA_RUNTIME_LIB) that fails to load used to silently fall back to the
    mock backend, hiding a real deployment error. Must raise.  [FIX]
  * TesseraRuntime._telemetry_events — was an unbounded list (leak in long
    runs); now a bounded deque.  [FIX]
  * autotune._measure_gemm_wall_clock — timed a numpy reference GEMM but
    labeled the result method="on_device" for the named backend (Decision #21);
    now labeled "wall_clock_reference".  [FIX]
"""

from __future__ import annotations

import collections

import pytest

from tessera.autotune import _measure_gemm_wall_clock
from tessera.compiler.autotune_v2 import TuningConfig, TuningResult
from tessera.runtime import TesseraRuntime, TesseraRuntimeError


def test_explicit_lib_path_failure_raises(monkeypatch):
    monkeypatch.delenv("TESSERA_RUNTIME_LIB", raising=False)
    with pytest.raises(TesseraRuntimeError):
        TesseraRuntime(lib_path="/no/such/libtessera_runtime.so")


def test_explicit_env_lib_failure_raises(monkeypatch):
    monkeypatch.setenv("TESSERA_RUNTIME_LIB", "/no/such/libtessera_runtime.so")
    with pytest.raises(TesseraRuntimeError):
        TesseraRuntime()


def test_mock_and_default_do_not_raise(monkeypatch):
    monkeypatch.delenv("TESSERA_RUNTIME_LIB", raising=False)
    assert TesseraRuntime(mock=True)._mock_mode is True
    # No explicit lib + nothing found on the default search → mock, no raise.
    assert TesseraRuntime()._mock_mode is True


def test_telemetry_events_is_bounded():
    rt = TesseraRuntime(mock=True)
    assert isinstance(rt._telemetry_events, collections.deque)
    assert rt._telemetry_events.maxlen is not None and rt._telemetry_events.maxlen > 0


def test_wall_clock_measurement_label_is_honest():
    cfg = TuningConfig(tile_m=64, tile_n=64, tile_k=32, num_warps=4, num_stages=2)
    result = TuningResult(cfg, 0.0, 0.0, 0.0, 0, "ok", "", "roofline")
    out = _measure_gemm_wall_clock(result, (64, 64, 64), dtype="bf16", backend="apple_cpu")
    # must not claim an on-device measurement it did not make
    assert out.method == "wall_clock_reference"
    assert "reference" in out.reason
