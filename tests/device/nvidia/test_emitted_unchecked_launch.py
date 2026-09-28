"""An emitted sm_120 lane must not report a launch that never ran as success.

Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (``NVIDIA-EMITTED-UNCHECKED-LAUNCH``).
Measured on this box (CUDA 13.4 / driver 610.88): after an invalid-configuration
launch, ``cudaDeviceSynchronize`` returns SUCCESS and the error sits only in the
last-error slot. 55 emitted entries judged their launch by the sync alone, so a
kernel that never executed returned ``rc 1`` and its (uninitialised) output was
handed back as a result.

Each lane here is run twice with every launch in its emitted source made invalid
(grid ``dim3(0u)``, a configuration error raised at launch):

1. **negative control** -- the post-launch slot reads rewritten to
   ``cudaSuccess`` (the sync-only judgment the entries used to make): the call
   must *succeed*, which is the defect, observed rather than assumed;
2. the shipped emitter's source: the call must *fail* (``RuntimeError`` from the
   run function's ``rc != 1`` check).

Every per-process artifact cache in ``nvidia_cuda`` is emptied for each run, so
the transformed source is what compiles.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = pytest.mark.hardware_nvidia


def _require_sm120():
    from tessera import runtime as rt

    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("exact sm_120 device is unavailable")


def _invalidate_launches(source: str) -> str:
    """Every ``<<<grid, ...>>>`` launch with its grid replaced by ``dim3(0u)``."""
    out, i = [], 0
    while True:
        j = source.find("<<<", i)
        if j < 0:
            out.append(source[i:])
            return "".join(out)
        k, depth = j + 3, 0
        while True:
            c = source[k]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 0:
                break
            k += 1
        out.append(source[i:j + 3])
        out.append("dim3(0u)")
        i = k


def _sync_only(source: str) -> str:
    """The judgment the entries used to make: no slot read after a launch."""
    return re.sub(r"cudaGetLastError\(\)", "cudaSuccess", source)


def _reset_artifact_caches(monkeypatch) -> None:
    from tessera.compiler.emit import nvidia_cuda as N

    for name, value in list(vars(N).items()):
        if not (name.endswith("_artifact") or name.endswith("_artifacts")
                or name in ("_EMITTED_ARTIFACTS", "_EMITTED_KEYS",
                            "_EMITTED_KEYS_BY_OBJ", "_LIB_CACHE")):
            continue
        if isinstance(value, dict):
            monkeypatch.setattr(N, name, {})
        elif value is None or isinstance(value, str):
            monkeypatch.setattr(N, name, None)


def _patch(monkeypatch, emitter: str, index: int | None, transform) -> None:
    from tessera.compiler.emit import nvidia_cuda as N

    original = getattr(N, emitter)

    def patched(*a, _o=original, **k):
        out = _o(*a, **k)
        if index is None:
            return transform(out)
        items = list(out)
        items[index] = transform(items[index])
        return tuple(items)

    monkeypatch.setattr(N, emitter, patched)


def _rng(*shape: int) -> np.ndarray:
    return (np.random.default_rng(sum(shape)).standard_normal(shape) * .1).astype(np.float32)


def _lanes() -> dict[str, tuple[str, int | None, Callable[[], Any]]]:
    from tessera.compiler.emit import nvidia_cuda as N

    x2 = _rng(8, 33)
    q4, k4, v4 = _rng(1, 2, 9, 16), _rng(1, 2, 11, 16), _rng(1, 2, 11, 16)
    pages = _rng(3, 4, 2, 8)
    table = np.array([2, 0, 1], np.int32)
    tok = np.array([3, 0, 7, 1], np.int32)
    return {
        "flash_fwd": ("_synthesize_flash_fwd_cuda", None,
                      lambda: N.run_flash_attention_forward(q4, k4, v4, scale=.25)),
        "flash_bwd_atomic": ("_synthesize_flash_bwd_cuda", None,
                             lambda: N.run_flash_attention_backward(
                                 q4, q4, k4, v4, scale=.25, route="atomic")),
        "softmax": ("_synthesize_softmax_cuda", None, lambda: N.run_row_softmax(x2)),
        "norm": ("_synthesize_norm_cuda", None, lambda: N.run_row_norm(x2, "rmsnorm", 1e-5)),
        "reduce": ("_synthesize_reduce_cuda", None, lambda: N.run_row_reduce(x2, "sum")),
        "reduce_timed": ("_synthesize_reduce_cuda", None,
                         lambda: N.measure_row_reduce_device(x2, "sum", reps=2)),
        "linear_attn": ("_synthesize_linear_attn_cuda", None,
                        lambda: N.run_linear_attention(q4, q4, q4)),
        "moe_dispatch": ("_synthesize_moe_cuda", None,
                         lambda: N.run_moe_dispatch_f32(_rng(8, 5), tok)),
        "paged_kv_read": ("_synthesize_paged_kv_read_cuda", None,
                          lambda: N.run_paged_kv_cache_read_f32(pages, table, 1, 9)),
        "paged_kv_read_timed": ("_synthesize_paged_kv_read_cuda", None,
                                lambda: N.measure_paged_kv_cache_read_device_f32(
                                    pages, table, 1, 9, reps=2)),
        "relu_bias": ("_synthesize_relu_bias_cuda", None,
                      lambda: N.run_relu_bias_f32(x2, x2[0])),
        "fused_epilogue": ("_synthesize_fused_epilogue_cuda", 1,
                           lambda: N.run_fused_epilogue_f32(x2, x2[0], "gelu")),
        "gated_epilogue": ("_synthesize_gated_epilogue_cuda", 2,
                           lambda: N.run_gated_epilogue_f32(x2, x2, "silu")),
        "gated_epilogue_timed": ("_synthesize_gated_epilogue_cuda", 2,
                                 lambda: N.measure_gated_epilogue_device(
                                     x2, x2, "silu", reps=2)),
        "conv2d": ("_synthesize_conv2d_nhwc_cuda", None,
                   lambda: N.run_conv2d_nhwc_f32(_rng(1, 5, 5, 3), _rng(3, 3, 3, 4))),
        "rope": ("_synthesize_posenc_cuda", None,
                 lambda: N.run_rope_f32(_rng(4, 8), _rng(4, 8))),
        "control_for": ("_synthesize_control_flow_cuda", None,
                        lambda: N.run_control_for_f32(_rng(17), trip=3)),
    }


_LANE_NAMES = ("flash_fwd", "flash_bwd_atomic", "softmax", "norm", "reduce",
               "reduce_timed", "linear_attn", "moe_dispatch", "paged_kv_read",
               "paged_kv_read_timed", "relu_bias", "fused_epilogue",
               "gated_epilogue", "gated_epilogue_timed", "conv2d", "rope",
               "control_for")


@pytest.mark.parametrize("lane", _LANE_NAMES)
def test_a_launch_that_never_ran_is_reported(lane, monkeypatch):
    _require_sm120()
    lanes = _lanes()
    assert set(lanes) == set(_LANE_NAMES)
    emitter, index, run = lanes[lane]

    # 1. negative control: the sync-only judgment calls the dead launch a success.
    with monkeypatch.context() as m:
        _reset_artifact_caches(m)
        _patch(m, emitter, index, lambda s: _sync_only(_invalidate_launches(s)))
        run()
    # 2. the shipped entry reads the slot and reports it.
    with monkeypatch.context() as m:
        _reset_artifact_caches(m)
        _patch(m, emitter, index, _invalidate_launches)
        with pytest.raises(RuntimeError):
            run()
    # 3. and the unmodified lane still runs.
    with monkeypatch.context() as m:
        _reset_artifact_caches(m)
        run()
