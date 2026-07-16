"""Host-free NVIDIA control-flow source and rejection contracts."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import nvidia_cuda as nv


def test_control_source_has_four_single_launch_entries():
    source = nv._synthesize_control_flow_cuda()
    for name in ("control_for", "control_if", "control_while", "control_scan"):
        assert f"tessera_nvidia_{name}_f32" in source
    assert source.count("<<<") == 4


def test_control_abi_rejects_bad_shapes_before_cuda():
    with pytest.raises(ValueError, match="rank-1 f32"):
        nv.run_control_for_f32(np.zeros((2, 2), np.float32), trip=2)
    with pytest.raises(ValueError, match="N<=1024"):
        nv.run_control_while_f32(np.zeros(1025, np.float32), max_iters=2, limit=1)
    with pytest.raises(ValueError, match=r"\[trip,N\]"):
        nv.run_control_scan_f32(np.zeros(4, np.float32), np.zeros((3, 5), np.float32))
