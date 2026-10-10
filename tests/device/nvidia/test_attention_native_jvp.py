"""Owning RTX5070 public native attention forward-AD route."""
import os
import pytest
from benchmarks.nvidia.benchmark_public_attention_jvp import run

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_NVIDIA_DEVICE_PROOF")!="1",
                              reason="requires owning RTX5070 device proof lane")

@pytest.mark.parametrize("order,wrt,sk,causal",[
    (("q","k","v"),("q",),5,False),
    (("v","q","k"),("k","q"),5,False),
    (("k","v","q"),("v",),129,True),
    (("v","k","q"),("q","k","v"),129,True),
])
def test_public_attention_jvp(order,wrt,sk,causal):
    assert run(order,wrt,sk,causal)["correctness"]=="passed_before_timing"
