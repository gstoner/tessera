from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.ssm_replay import replay_lifecycle_descriptor


def test_rocm_replay_lifecycle_maps_persistent_workspace_and_ring_ownership() -> None:
    lifecycle = replay_lifecycle_descriptor(
        target="rocm_gfx1151",
        batch=1,
        channels=4,
        state_dim=3,
        capacity=8,
        async_slots=2,
    ).as_metadata_dict()
    assert lifecycle["abi_id"] == "tessera.rocm.replay_ssm.resident.f32.v1"
    assert lifecycle["ownership"] == "session_private"
    assert lifecycle["ring_ownership"] == \
        "handle_exclusive_until_wait_or_release"
    assert lifecycle["flush_semantics"] == \
        "fold_live_ring_into_checkpoint_then_clear"
    assert lifecycle["rollback_semantics"] == \
        "rewind_cursor_and_invalidate_rejected_tail"
    assert lifecycle["teardown"] == "drain_pending_then_release"
    assert lifecycle["state"]["workspace"]["lifetime"] == "session"
    assert lifecycle["state"]["workspace"]["initialization"] == "preserve"


def test_rocm_replay_handle_exposes_shared_lifecycle_off_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tessera.compiler.emit import rocm_hip

    class _Unavailable:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("no device")

    monkeypatch.setattr(rocm_hip, "RocmReplayDeviceState", _Unavailable)
    handle = rt.rocm_ssm_replay_state_handle(
        1, 4, 3, -np.ones(4), capacity=8, async_slots=2
    )
    telemetry = handle.lifecycle_telemetry()
    assert telemetry["resident_inputs"] is False
    assert telemetry["ring_slots"] == 2
    assert telemetry["pending_submissions"] == 0
    assert telemetry["lifecycle_descriptor"]["state"]["target"] == \
        "rocm_gfx1151"
    handle.close()
    assert handle.lifecycle_telemetry()["closed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        handle.append(np.ones((1, 4)), np.ones((1, 4)), np.ones((1, 3)))
