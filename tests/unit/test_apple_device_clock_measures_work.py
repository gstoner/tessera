"""A device clock that does not move with the host wall is not a measurement.

`ts_record_dispatch_gpu_elapsed` preferred `cb.kernelStartTime`/`kernelEndTime`
and treated `GPUStartTime`/`GPUEndTime` as a fallback, on a comment claiming the
first pair was "the completed compute-kernel interval". The SDK says the
opposite by omission: `GPUStartTime` carries an `@abstract` ("the host time in
seconds that GPU starts executing this command buffer") and `kernelStartTime` is
a bare, undocumented declaration.

Measured on an M1 Max (2026-08-31) with only a kernel's loop count varying:

    iters      kernelS/E        GPUS/E       encoder     host wall   kern/wall
    5,000         54,583       498,375       498,375       764,417       0.071
    320,000       65,833     9,390,833     9,390,792     9,833,750       0.007

`kernelStartTime`/`kernelEndTime` is flat across a 64x workload.
`GPUStartTime`/`GPUEndTime` tracks the host wall *and* agrees with an
independent stage-boundary counter-sample clock to the nanosecond
(9,390,833 vs 9,390,792) -- two mechanisms agreeing that closely is what makes
it a measurement rather than a plausible number.

**No bound could have caught this.** An under-reading clock looks exactly like a
small kernel: both sit far below the host wall, which is why Apple's acceptance
band is one-sided (`accept_apple_device_ns`). What catches it is metamorphic --
vary the workload and require the two clocks to move together.

On Tessera's own resident-attention lane, scaling KV length 19 -> 1216:

    before   58,166 -> 28,875 ns   (0.50x, while the wall rose 1.55x)
    after   864,000 -> 1,505,333 ns (1.74x, wall 1.53x)

The assertion is *tracking*, not growth, and the difference is not cosmetic --
see `_MIN_TRACKING`. Requiring absolute growth made this test pass alone and
fail inside the full Apple sweep, because at these sizes a dispatch is
overhead-dominated and a 64x KV increase does not reliably produce 64x more
measurable time. In that failing run the host wall fell too, so the premise was
wrong rather than the clock.
"""
from __future__ import annotations

import numpy as np
import pytest


#: KV lengths spanning 64x. Wide on purpose: the defect this guards against
#: reported a *flat* ~30-60 us across exactly this range, so a narrow sweep
#: would sit inside the noise and prove nothing.
_KV_LENGTHS = (19, 76, 304, 1216)

#: The device clock must TRACK THE WALL across the sweep, not hit an absolute
#: growth target.
#:
#: A first version asserted the device time grows >= 1.25x over the 64x sweep,
#: and it was flaky -- correct in isolation (1.74x) and failing inside the full
#: Apple sweep (0.55x). The reason is instructive rather than incidental: at
#: these sizes a dispatch costs ~0.7-1.0 ms of fixed overhead, so a 64x KV
#: increase is not 64x more measurable time, and how much of that overhead is
#: warm depends on what ran before. In the failing run the HOST WALL fell too
#: (0.64x) -- the workload genuinely was not growing, so the premise was wrong,
#: not the clock.
#:
#: The invariant that actually holds is agreement: whatever the workload does,
#: the device clock and the host wall must move together. That is precisely
#: what the defect violated -- device 0.50x while the wall rose 1.55x, a
#: divergence of 0.32 -- and it holds in both the isolated run (1.74/1.53 =
#: 1.14) and the sweep (0.55/0.64 = 0.86).
_MIN_TRACKING = 0.5
_MAX_TRACKING = 2.0


def _run(sk, agpu):
    outer, q_heads, kv_heads, sq, dim = 2, 4, 2, 5, 64
    bq, bkv = outer * q_heads, outer * kv_heads
    rng = np.random.default_rng(7)
    st = np.float32
    q = np.ascontiguousarray((rng.normal(size=(bq, sq, dim)) * .2).astype(st))
    k = np.ascontiguousarray((rng.normal(size=(bkv, sk, dim)) * .2).astype(st))
    v = np.ascontiguousarray((rng.normal(size=(bkv, sk, dim)) * .2).astype(st))
    bias = np.ascontiguousarray((rng.normal(size=(bq, sq, sk)) * .1).astype(st))
    qd, kd, vd, bd = (agpu.device_tensor(x) for x in (q, k, v, bias))
    kw = dict(dtype="f32", B=bq, q_heads=q_heads, kv_heads=kv_heads, Sq=sq,
              Sk=sk, D=dim, causal=True, window_size=9, logit_softcap=3.0)
    return qd, kd, vd, bd, kw


@pytest.mark.hardware_apple_gpu
def test_the_device_clock_moves_with_the_host_wall():
    """The check that found the defect, kept so it cannot return."""
    from tessera import apple_gpu_batched as agpu
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    assert set_dispatch_telemetry_enabled(True)

    device_ns, wall_ns, sources = [], [], []
    for sk in _KV_LENGTHS:
        qd, kd, vd, bd, kw = _run(sk, agpu)
        for _ in range(3):          # warm: exclude pipeline construction
            with agpu.batched_session() as s:
                agpu.flash_attn_variant_enc(s, qd, kd, vd, bd, **kw)
        clear_dispatch_telemetry()
        with agpu.batched_session() as s:
            agpu.flash_attn_variant_enc(s, qd, kd, vd, bd, **kw)
        t = read_dispatch_telemetry()
        assert t["device_time_ns"] is not None, (
            f"Sk={sk}: no accepted device time; either the clock was refused "
            "by the acceptance band or telemetry did not record")
        device_ns.append(t["device_time_ns"])
        wall_ns.append(t["wall_time_ns"])
        sources.append(t["timing_source"])

    device_growth = device_ns[-1] / device_ns[0]
    wall_growth = wall_ns[-1] / wall_ns[0]
    tracking = device_growth / wall_growth
    assert _MIN_TRACKING <= tracking <= _MAX_TRACKING, (
        f"the device clock did not move with the host wall across a "
        f"{_KV_LENGTHS[-1] // _KV_LENGTHS[0]}x KV sweep: device "
        f"{device_ns[0]} -> {device_ns[-1]} ns ({device_growth:.2f}x) against "
        f"wall {wall_ns[0]} -> {wall_ns[-1]} ns ({wall_growth:.2f}x), "
        f"tracking {tracking:.2f}. sources={sources}. This is the "
        "kernelStartTime defect: a clock reporting a plausible number that is "
        "not a measurement of the kernel. Note the two are allowed to diverge "
        "in MAGNITUDE -- the wall carries submission overhead the GPU interval "
        "excludes -- but not in DIRECTION.")


@pytest.mark.hardware_apple_gpu
def test_the_documented_gpu_clock_is_the_one_selected():
    """`GPUStartTime`/`GPUEndTime`, not the undocumented kernel pair.

    **This guards the fallback, not the clock's identity.** The source label is
    derived from whether `ts_gpu_interval` succeeded, not from which property
    it read, so an edit that changes the property while leaving the label alone
    passes here -- verified by mutation. The tracking check above is what
    catches that, and it did.

    What this does catch is the subtler regression: `GPUStartTime` reading zero
    because nothing forced the property to publish, so the code silently falls
    back to the undocumented pair. That failure keeps producing plausible
    numbers and would otherwise only show up as a slow drift in recorded
    evidence.
    """
    from tessera import apple_gpu_batched as agpu
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    assert set_dispatch_telemetry_enabled(True)
    qd, kd, vd, bd, kw = _run(304, agpu)
    for _ in range(3):
        with agpu.batched_session() as s:
            agpu.flash_attn_variant_enc(s, qd, kd, vd, bd, **kw)
    clear_dispatch_telemetry()
    with agpu.batched_session() as s:
        agpu.flash_attn_variant_enc(s, qd, kd, vd, bd, **kw)
    source = read_dispatch_telemetry()["timing_source"]
    assert source != "metal_kernel_interval", (
        "fell back to the undocumented kernelStartTime pair, which was "
        "measured flat across a 64x workload. If GPUStartTime is genuinely "
        "unavailable here, that is a finding to record -- not a fallback to "
        "take silently.")
    assert source in {"metal_command_buffer_interval", "metal4_timestamp_heap",
                      "metal4_mpsgraph_envelope"}, source
