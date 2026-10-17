"""Owning gfx1201 explicit BF16 storage / FP32 compute parent-view proof."""
import os
import json

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_parent_view_native as native

pytestmark = pytest.mark.skipif(
    os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 exact-device environment",
)


def function(rows, width, reverse):
    def gate_up(parent):
        left = ts.ops.slice(parent, [0, 0], [rows, width])
        right = ts.ops.slice(parent, [0, width], [rows, width])
        left_f32 = ts.ops.cast(left, "fp32")
        right_f32 = ts.ops.cast(right, "fp32")
        product = ts.ops.silu_mul(right_f32, left_f32) if reverse else ts.ops.silu_mul(left_f32, right_f32)
        return ts.ops.cast(product, "bf16")
    return gate_up


def oracle(parent, width, reverse):
    left = parent[:, :width].astype(np.float64)
    right = parent[:, width:].astype(np.float64)
    gate, up = (right, left) if reverse else (left, right)
    return (gate / (1 + np.exp(-gate)) * up).astype(ml_dtypes.bfloat16)


def assert_result(actual, expected):
    assert actual.dtype == np.dtype(ml_dtypes.bfloat16)
    # Independent f64 oracle, final BF16 rounding; allow one BF16 relative ULP.
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32),
                               rtol=1 / 128, atol=1e-5)


@pytest.mark.parametrize("rows,width", [(3, 17), (16, 64), (256, 12288)])
@pytest.mark.parametrize("reverse", [False, True])
def test_bf16_parent_jit_changed_inputs_and_portable_replay(rows, width, reverse, monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201"
    rng = np.random.default_rng(120117)
    parent = rng.normal(0, 0.7, (rows, 2 * width)).astype(ml_dtypes.bfloat16)
    before = parent.copy()
    compiled = ts.jit(target="rocm_gfx1201")(function(rows, width, reverse))
    expected = oracle(parent, width, reverse)
    def forbid_eager(*args, **kwargs):
        pytest.fail("BF16 parent-view execution evaluated eager SiLU arithmetic")
    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbid_eager)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbid_eager)
        result = compiled(parent)
    assert compiled.execution_kind == "native_gpu"
    assert_result(result, expected)
    held = result.copy()
    np.testing.assert_array_equal(parent, before)
    artifact = compiled.runtime_artifact()
    assert artifact.launch_descriptor.abi_id == native.PARENT_VIEW_BF16_ABI
    assert artifact.native_image.pipeline_name == "tessera-rocm-executable"

    def forbidden(*args, **kwargs):
        pytest.fail("warm or serialized BF16 parent-view replay invoked compiler lowering")

    monkeypatch.setattr(native, "run_tessera_opt", forbidden)
    monkeypatch.setattr(native, "find_tessera_opt", forbidden)
    changed = rng.normal(0, 0.9, parent.shape).astype(ml_dtypes.bfloat16)
    changed_before = changed.copy()
    expected_changed = oracle(changed, width, reverse)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbid_eager)
        warm = compiled(changed)
    assert_result(warm, expected_changed)
    np.testing.assert_array_equal(changed, changed_before)
    np.testing.assert_array_equal(result, held)
    assert not np.shares_memory(result, warm)

    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    output = np.empty((rows, width), dtype=ml_dtypes.bfloat16)
    member = json.loads(restored.launch_descriptor.provenance["native_parent_view"]["member_json"])
    receipt = rt.launch(restored, {"buffers": dict(zip(member["bindings"], [changed, output], strict=True)),
                                  "scalars": {"N": rows * width}})
    assert receipt.get("ok") and receipt["execution_kind"] == "native_gpu", receipt
    assert_result(output, expected_changed)


def test_bf16_parent_fresh_process_replay_without_compiler(tmp_path):
    import subprocess
    import sys

    assert rt._rocm_live_arch() == "gfx1201"
    parent = np.random.default_rng(1717).normal(0, 0.7, (3, 34)).astype(ml_dtypes.bfloat16)
    compiled = ts.jit(target="rocm_gfx1201")(function(3, 17, False))
    assert_result(compiled(parent), oracle(parent, 17, False))
    (tmp_path / "artifact.json").write_text(compiled.runtime_artifact().to_json())
    np.save(tmp_path / "parent-bits.npy", parent.view(np.uint16))
    script = r"""
import json,sys,subprocess
from pathlib import Path
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_parent_view_native as native
root=Path(sys.argv[1])
parent=np.load(root/'parent-bits.npy').view(ml_dtypes.bfloat16)
gate,up=np.split(parent.astype(np.float64),2,axis=1)
expected=(gate/(1+np.exp(-gate))*up).astype(ml_dtypes.bfloat16)
def forbidden(*args,**kwargs):raise AssertionError('fresh BF16 replay invoked compiler or eager arithmetic')
native.find_tessera_opt=forbidden
native.run_tessera_opt=forbidden
subprocess.run=forbidden
np.exp=forbidden
rt._execute_rocm_compiled_silu_mul=forbidden
artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())
names=[buffer.name for buffer in artifact.launch_descriptor.buffers]
output=np.empty((3,17),dtype=ml_dtypes.bfloat16)
receipt=rt.launch(artifact,{'buffers':dict(zip(names,[parent,output],strict=True)),'scalars':{'N':51}})
assert receipt['ok'] and receipt['execution_kind']=='native_gpu',receipt
np.testing.assert_allclose(output.astype(np.float32),expected.astype(np.float32),rtol=1/128,atol=1e-5)
print(json.dumps({'architecture':rt._rocm_live_arch(),'execution_kind':receipt['execution_kind']}))
"""
    env = {**os.environ, "TESSERA_OPT": "/unavailable/compiler", "TESSERA_ROCM_OPT": "/unavailable/compiler"}
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                               capture_output=True, text=True, check=True)
    assert '"execution_kind": "native_gpu"' in completed.stdout
