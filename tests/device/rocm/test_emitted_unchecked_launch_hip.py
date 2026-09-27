"""A HIP launch that never ran must not be reported as success (gfx1151, gfx1201).

Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (the ROCm half of
``NVIDIA-EMITTED-UNCHECKED-LAUNCH``). The gfx1151 paged-KV gather, the direct
paged attention (the two routes the paged-KV warm start serves) and the
ReplaySSM summary step (``su``) judged their launches by ``hipDeviceSynchronize``
/ the event sync alone. They now read the sticky HIP slot after each launch
group.

Each lane runs with every launch made invalid (grid ``dim3(0u)``):

* the shipped source must FAIL (the run function's ``rc != 1`` check raises);
* the sync-only control (post-launch reads rewritten to ``hipSuccess``) is run
  and its outcome is recorded, not asserted: whether ``hipDeviceSynchronize``
  on this ROCm/WSL2 stack reports a configuration error is the fact being
  measured, and the fix must not depend on it.

Then the unmodified lane must still run. Every per-process artifact cache in
``rocm_hip`` is emptied so the transformed source is what compiles.
"""
from __future__ import annotations

import os
import re
import shutil

import numpy as np
import pytest


def _live_rdna_arch() -> str | None:
    """gfx1151, or gfx1201 when its device proof is requested and the build
    arch is set to it (so the emitted HIP compiles for the chip that runs it)."""
    from tessera import runtime as rt

    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    try:
        live = rt._rocm_live_arch() if os.path.isfile(hipcc) else None
    except Exception:  # noqa: BLE001 - no ROCm runtime is "not this host"
        return None
    if live == "gfx1151":
        return live
    if (live == "gfx1201" and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") == "1"
            and rt._rocm_chip() == "gfx1201"):
        return live
    return None


pytestmark = pytest.mark.skipif(
    _live_rdna_arch() is None,
    reason="requires a live gfx1151 device (or gfx1201 with TESSERA_GFX1201_DEVICE_PROOF=1 "
           "and TESSERA_ROCM_CHIP=gfx1201) and hipcc")


def _invalidate_launches(source: str) -> str:
    """Every ``hipLaunchKernelGGL(k, grid, ...)`` and ``k<<<grid, ...>>>`` with its
    grid replaced by ``dim3(0u)``."""
    def replace_arg(text: str, start: int) -> tuple[str, int]:
        k, depth = start, 0
        while True:
            c = text[k]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 0:
                return "dim3(0u)", k
            k += 1

    out, i = [], 0
    pattern = re.compile(r"<<<|hipLaunchKernelGGL\(")
    while True:
        m = pattern.search(source, i)
        if m is None:
            out.append(source[i:])
            return "".join(out)
        if m.group(0) == "<<<":
            out.append(source[i:m.end()])
            grid, i = replace_arg(source, m.end())
        else:
            first_comma = source.index(",", m.end())      # past the kernel name
            out.append(source[i:first_comma + 1])
            grid, i = replace_arg(source, first_comma + 1)
        out.append(grid)


def _sync_only(source: str) -> str:
    return re.sub(r"hipGetLastError\(\)", "hipSuccess", source)


def _reset(monkeypatch) -> None:
    from tessera.compiler.emit import rocm_hip as R

    for name, value in list(vars(R).items()):
        if name.endswith("_artifact") and (value is None or isinstance(value, str)):
            monkeypatch.setattr(R, name, None)
        elif name in ("_PAGED_ARTIFACTS", "_LIB_CACHE"):
            monkeypatch.setattr(R, name, {})


def _patch(monkeypatch, emitter: str, transform) -> None:
    from tessera.compiler.emit import rocm_hip as R

    original = getattr(R, emitter)
    monkeypatch.setattr(R, emitter, lambda *a, _o=original, **k: transform(_o(*a, **k)))


def _lanes():
    from tessera.compiler.emit import rocm_hip as R

    rng = np.random.default_rng(1151)
    pages = (rng.standard_normal((3, 4, 2, 16)) * .1).astype(np.float32)
    table = np.array([2, 0, 1], np.int32)
    idx = np.array([5, 0, 9, 3, 11], np.int64)
    q = (rng.standard_normal((2, 1, 16)) * .1).astype(np.float32)

    def summary():
        device = R.RocmReplayDeviceState(np.zeros((1, 4, 3)), -np.linspace(.2, 1, 4),
                                         capacity=1)
        d = np.abs(rng.standard_normal((1, 4))).astype(np.float32) * .2
        x = rng.standard_normal((1, 4)).astype(np.float32)
        b = rng.standard_normal((1, 3)).astype(np.float32)
        return device.summary_step(d, x, b, b)

    return {
        "paged_kv_read": ("_synthesize_paged_kv_read_hip",
                          lambda: R.run_paged_kv_cache_read_f32(pages, table, idx, reps=2)),
        "paged_attention_direct": ("_synthesize_paged_attention_direct_hip",
                                   lambda: R.run_paged_attention_direct_f32(
                                       q, pages, pages, table, idx, scale=.25,
                                       causal=True, reps=2)),
        "ssm_summary_step": ("_synthesize_ssm_replay_device_hip", summary),
    }


@pytest.mark.parametrize("lane", ("paged_kv_read", "paged_attention_direct",
                                  "ssm_summary_step"))
def test_a_hip_launch_that_never_ran_is_reported(lane, monkeypatch, record_property):
    emitter, run = _lanes()[lane]
    with monkeypatch.context() as m:
        _reset(m)
        _patch(m, emitter, lambda s: _sync_only(_invalidate_launches(s)))
        try:
            run()
            control = "sync-only judgment reported the dead launch as success"
        except RuntimeError:
            control = "sync-only judgment reported the dead launch as a failure"
    record_property("sync_only_control", control)
    print(f"{lane}: {control}")
    with monkeypatch.context() as m:
        _reset(m)
        _patch(m, emitter, _invalidate_launches)
        with pytest.raises(RuntimeError):
            run()
    with monkeypatch.context() as m:
        _reset(m)
        run()
