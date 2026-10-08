"""Native graph replay retains compiled images, addresses and generations."""
import ctypes as ct
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.resident_rocm_movement import ResidentMovementCall
from tests.unit.test_public_movement_frontend import paged, dispatched, inputs
from benchmarks.rocm.benchmark_captured_movement import paged_full
from benchmarks.rocm.benchmark_paged_softmax_edge import normalized

def test_capture_fork_guard_precedes_inherited_lock():
    owner=ResidentMovementCall.__new__(ResidentMovementCall)
    owner.pid=os.getpid()+1;owner.closed=False
    owner.lock=None
    with pytest.raises(ValueError,match="fork"):owner.capture()

def test_capture_missing_library_has_explicit_boundary():
    import threading
    owner=ResidentMovementCall.__new__(ResidentMovementCall)
    owner.pid=os.getpid();owner.closed=False;owner.lock=threading.RLock()
    owner.lib=SimpleNamespace()
    with pytest.raises(RuntimeError,match="graph replay support"):owner.capture()

@pytest.mark.parametrize("value",[0,1,None,"yes"])
def test_capture_execution_flag_is_strict_boolean(value):
    owner=ResidentMovementCall.__new__(ResidentMovementCall)
    owner.pid=os.getpid();owner.closed=False
    with pytest.raises(TypeError,match="captured execution"):owner.execute(captured=value)

DEVICE=os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF")=="1"

def build_case(architecture,family,large=False):
    if family=="full":
        rng=np.random.default_rng(61108)
        args=(rng.normal(size=(32,16,8,128)).astype(np.float32),
              rng.integers(0,32,64,dtype=np.int32))
        source=paged_full
    else:
        args,_=inputs(family,large)
        source=paged if family=="paged" else dispatched
    fn=ts.jit(target="rocm_"+architecture,native_required=True)(source)
    if family!="dispatched":
        def reference(values):
            x,table=values
            gathered=x[table].reshape(-1,*x.shape[2:])
            return gathered if family=="full" else gathered[1:6]
    else:
        def reference(values):return values[1][values[0]]
    return fn,args,reference

@pytest.mark.skipif(not DEVICE,reason="owning ROCm native movement proof required")
@pytest.mark.parametrize("family,large,softmax",[
    ("paged",False,False),("paged",True,False),
    ("dispatched",False,False),("dispatched",True,False),
    ("paged",False,True),("paged",True,True),("full",True,True),
] if os.environ.get("TESSERA_ROCM_CHIP")!="gfx1201" else [
    ("paged",False,False),("paged",True,False),
    ("paged",False,True),("paged",True,True),("full",True,True),
])
def test_compiler_owned_graph_replay_and_lifetime(family,large,softmax):
    arch=rt._rocm_live_arch()
    assert arch==os.environ["TESSERA_ROCM_CHIP"] and arch in {"gfx1151","gfx1201"}
    fn,args,reference=build_case(arch,family,large)
    fn(*args)
    if softmax:
        consumer=ts.jit(target="rocm_"+arch,native_required=True)(normalized)
        owner=fn.prepare_native_paged_softmax(consumer,*args)
        raw_reference=reference
        def reference(values):
            x=raw_reference(values).astype(np.float64)
            e=np.exp(x-x.max(-1,keepdims=True))
            return e/e.sum(-1,keepdims=True)
        def check(value,expected):
            np.testing.assert_allclose(value,expected,rtol=3e-5,atol=2e-6)
    else:
        owner=fn.prepare_native_movement(*args)
        def check(value,expected):
            np.testing.assert_array_equal(value.view(np.uint32),expected.view(np.uint32))
    expected=reference(args)
    try:
        with pytest.raises(RuntimeError,match="rc=10"):owner.execute(captured=True)
        empty=owner.prepared.resident()
        try:
            with pytest.raises(RuntimeError,match="rc=10"):empty.capture()
        finally:empty.close()
        assert owner.capture()==(2 if softmax else 1)
        assert owner.capture()==(2 if softmax else 1)
        # The native prepared handle may retire: resident owner retains copied
        # compiler ABI/image and leases independently.
        owner.prepared.close()
        def forbidden(*a,**k):raise AssertionError("compiler during warm graph replay")
        with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
            retained=None
            for captured in (False,True,False,True):
                value,receipt=owner.execute(captured=captured)
                check(value,expected)
                assert receipt["submission"]==("native_hip_graph" if captured else "native_direct")
                assert receipt["kernel_elapsed_ms"]>0
                if softmax and not captured:
                    assert receipt["producer_kernel_elapsed_ms"]>0
                    assert receipt["consumer_kernel_elapsed_ms"]>0
                retained=value if retained is None else retained
            stale=owner.execute(download=False,captured=True)[0]
            owner.execute(captured=False)
            with pytest.raises(RuntimeError,match="rc=10"):stale.to_host()
            changed=list(args)
            source,index=owner.prepared.input_positions
            changed[source]=np.ascontiguousarray(args[source]*np.float32(-1.25))
            changed[index]=args[index][::-1].copy()
            owner.upload(tuple(changed))
            check(owner.execute(captured=True)[0],reference(tuple(changed)))
            check(retained,expected)
            owner.close()
            with pytest.raises(ValueError,match="closed"):owner.capture()
            with pytest.raises(ValueError,match="closed"):owner.execute(captured=True)
    finally:owner.close()
