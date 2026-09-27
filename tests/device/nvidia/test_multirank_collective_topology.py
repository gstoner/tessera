from __future__ import annotations

import numpy as np
import pytest

from tessera.collectives import adapter
from tests._support.nvidia import nvidia_cuda_host_ready


@pytest.mark.hardware_nvidia
def test_nccl_topology_executes_or_rejects_missing_exact_devices() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    collective = adapter(backend="nccl", world_size=2, device_ordinals=(0, 1))
    status = collective.status()
    if not status.available:
        assert status.status == "backend_unavailable"
        # The probe reports the first missing prerequisite. Without libnccl
        # (Super-Bear has CUDA but no NCCL package) it never reaches device
        # enumeration, so the device-count refusal cannot be observed here:
        # the refusal must still be honest and reach the executor, and the
        # host cannot evaluate the topology lane -- a skip, never a pass.
        if "runtime libraries must both be loadable" in status.reason:
            with pytest.raises(RuntimeError, match="must both be loadable"):
                collective.all_reduce([np.ones(8, np.float32), np.ones(8, np.float32)])
            pytest.skip("NCCL is not installed on this host; the multi-rank "
                        "topology lane cannot be evaluated here")
        assert "only 1 device(s) are visible" in status.reason
        with pytest.raises(RuntimeError, match="only 1 device"):
            collective.all_reduce([np.ones(8, np.float32), np.ones(8, np.float32)])
        return
    try:
        result = collective.all_reduce([
            np.arange(8, dtype=np.float32),
            np.arange(8, dtype=np.float32) + 1,
        ])
        expected = np.arange(8, dtype=np.float32) * 2 + 1
        for rank_result in result:
            np.testing.assert_array_equal(rank_result, expected)
    finally:
        collective.close()
