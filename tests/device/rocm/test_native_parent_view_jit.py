"""Exact gfx1201 public one-parent gate/up package and portable replay."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_parent_view_native as native


def gate_up_3_17(parent):
    left = ts.ops.slice(parent, [0, 0], [3, 17])
    right = ts.ops.slice(parent, [0, 17], [3, 17])
    return ts.ops.silu_mul(left, right)


def gate_up_3_17_reverse(parent):
    left = ts.ops.slice(parent, [0, 0], [3, 17])
    right = ts.ops.slice(parent, [0, 17], [3, 17])
    return ts.ops.silu_mul(right, left)


def gate_up_16_64(parent):
    left = ts.ops.slice(parent, [0, 0], [16, 64])
    right = ts.ops.slice(parent, [0, 64], [16, 64])
    return ts.ops.silu_mul(left, right)


def gate_up_16_64_reverse(parent):
    left = ts.ops.slice(parent, [0, 0], [16, 64])
    right = ts.ops.slice(parent, [0, 64], [16, 64])
    return ts.ops.silu_mul(right, left)


def gate_up_256_12288(parent):
    left = ts.ops.slice(parent, [0, 0], [256, 12288])
    right = ts.ops.slice(parent, [0, 12288], [256, 12288])
    return ts.ops.silu_mul(left, right)


def gate_up_256_12288_reverse(parent):
    left = ts.ops.slice(parent, [0, 0], [256, 12288])
    right = ts.ops.slice(parent, [0, 12288], [256, 12288])
    return ts.ops.silu_mul(right, left)


def function(rows, width, reverse):
    return {
        (3, 17, False): gate_up_3_17,
        (3, 17, True): gate_up_3_17_reverse,
        (16, 64, False): gate_up_16_64,
        (16, 64, True): gate_up_16_64_reverse,
        (256, 12288, False): gate_up_256_12288,
        (256, 12288, True): gate_up_256_12288_reverse,
    }[(rows, width, reverse)]


def closure_function(rows, width, reverse):
    if reverse:
        def gate_up_reverse(parent):
            left = ts.ops.slice(parent, [0, 0], [rows, width])
            right = ts.ops.slice(parent, [0, width], [rows, width])
            return ts.ops.silu_mul(right, left)
        return gate_up_reverse

    def gate_up(parent):
        left = ts.ops.slice(parent, [0, 0], [rows, width])
        right = ts.ops.slice(parent, [0, width], [rows, width])
        return ts.ops.silu_mul(left, right)
    return gate_up


def oracle(parent, reverse):
    left, right = np.split(parent.astype(np.float64), 2, axis=1)
    gate, up = (right, left) if reverse else (left, right)
    return ((gate / (1.0 + np.exp(-gate))) * up).astype(np.float32)


@pytest.mark.parametrize("rows,width", [(3, 17), (16, 64), (256, 12288)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("capture", ["literal", "closure"])
def test_public_parent_view_jit_and_portable_replay(rows, width, reverse, capture, monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201", "exact gfx1201 host required"
    parent = np.random.default_rng(1710).uniform(-5, 5, (rows, 2 * width)).astype(np.float32)
    expected = oracle(parent, reverse)
    original = parent.copy()
    frontend = function(rows, width, reverse) if capture == "literal" else closure_function(rows, width, reverse)
    fn = ts.jit(target="rocm_gfx1201")(frontend)

    def forbidden(*a, **k):
        pytest.fail("parent-view used eager arithmetic, legacy execution or warm compiler")

    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        actual = fn(parent)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_equal(parent, original)
    assert fn.execution_kind == "native_gpu"
    artifact = fn.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "rocm_parent_view_native_descriptor"
    native.validate_parent_view_runtime_artifact(artifact)
    held = actual.copy()
    changed = parent * np.float32(0.75)
    expected = oracle(changed, reverse)
    monkeypatch.setattr(native, "run_tessera_opt", forbidden)
    monkeypatch.setattr(native, "find_tessera_opt", forbidden)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        again = fn(changed)
    np.testing.assert_allclose(again, expected, rtol=3e-6, atol=2e-7)
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    output = np.empty((rows, width), np.float32)
    names = [binding.name for binding in restored.launch_descriptor.buffers]
    receipt = rt.launch(restored, {"buffers": {names[0]: changed, names[1]: output},
                                   "scalars": {"N": rows * width}})
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_allclose(output, expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_equal(actual, held)


def test_public_parent_view_fresh_process_replay(tmp_path):
    import os
    import subprocess
    import sys
    assert rt._rocm_live_arch() == "gfx1201", "exact gfx1201 host required"
    parent = np.random.default_rng(1711).uniform(-5, 5, (3, 34)).astype(np.float32)
    fn = ts.jit(target="rocm_gfx1201")(gate_up_3_17)
    np.testing.assert_allclose(fn(parent), oracle(parent, False), rtol=3e-6, atol=2e-7)
    path = tmp_path / "parent-view.json"
    path.write_text(fn.runtime_artifact().to_json())
    np.save(tmp_path / "parent.npy", parent)
    script = r'''
import json,sys,subprocess
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_parent_view_native as native
root=Path(sys.argv[1])
artifact=rt.RuntimeArtifact.from_json((root/'parent-view.json').read_text())
parent=np.load(root/'parent.npy')
gate,up=np.split(parent.astype(np.float64),2,axis=1)
expected=((gate/(1+np.exp(-gate)))*up).astype(np.float32)
def forbidden(*a,**k):raise AssertionError('fresh replay used compiler or eager arithmetic')
native.find_tessera_opt=forbidden
native.run_tessera_opt=forbidden
subprocess.run=forbidden
np.exp=forbidden
rt._execute_rocm_compiled_silu_mul=forbidden
names=[b.name for b in artifact.launch_descriptor.buffers]
output=np.empty((3,17),np.float32)
receipt=rt.launch(artifact,{'buffers':{names[0]:parent,names[1]:output},'scalars':{'N':51}})
assert receipt['ok'] and receipt['execution_kind']=='native_gpu',receipt
np.testing.assert_allclose(output,expected,rtol=3e-6,atol=2e-7)
print(json.dumps({'architecture':rt._rocm_live_arch(),'execution_kind':receipt['execution_kind'],
                  'image_digest':artifact.native_image.image_digest}))
'''
    env = {**os.environ, "TESSERA_OPT": "/unavailable/compiler", "TESSERA_ROCM_OPT": "/unavailable/compiler"}
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                               capture_output=True, text=True, check=True)
    assert '"execution_kind": "native_gpu"' in completed.stdout
