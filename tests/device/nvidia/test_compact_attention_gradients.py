"""Exact SM120 compact requested-gradient ABI integration."""
import pytest
from tessera.compiler import nvidia_native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.nvidia import nvidia_cuda_host_ready
from benchmarks.record_device_ring_protocol import Device
from benchmarks.nvidia.benchmark_compact_attention_gradients import run_case

pytestmark=[pytest.mark.hardware_nvidia,pytest.mark.skipif(
    not nvidia_cuda_host_ready() or not nvidia_native.tools_available() or find_tessera_opt() is None,
    reason="requires matching compiler and exact SM120 GPU")]

@pytest.mark.parametrize("shape,causal,wrt,bias", [
    ((1,2,1,3,5,4,3),False,("q",),None),
    ((1,2,1,8,129,8,6),True,("v",),None),
    ((2,4,2,3,5,4,3),True,("bias","q","v"),(1,4,1,1)),
])
def test_compact_requested_outputs_end_to_end(shape,causal,wrt,bias):
    row=run_case(Device("nvidia"),shape,causal,wrt,repetitions=4,bias_shape=bias)
    assert row["checked_host_and_resident_passed"]
    assert row["allocations"]["compact"]["retained_gradient_buffers"]==len(wrt)
    assert row["resources"]["compact"]["gradient_elements"]<=row["resources"]["complete"]["gradient_elements"]
