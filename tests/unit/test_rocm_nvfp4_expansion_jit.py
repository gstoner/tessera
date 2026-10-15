"""Public Graph capture, native JIT and serialized expansion replay."""
import os
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
from tessera.compiler.rocm_nvfp4_expand_native import NVFP4_EXPAND_ABI

DEVICE = pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                            reason="exact gfx1201 owning device")


@ts.jit(target="rocm_gfx1201")
def expand(codes, scales, globals_):
    return ts.ops.nvfp4_expand(codes, scales, globals_,
        row_offsets=[0, 3, 7], numeric_policy=nvfp4_expansion_policy())


def operands(k=48):
    codes = np.resize(np.arange(256, dtype=np.uint8), (7, k//2))
    bits = np.resize(np.arange(127, dtype=np.uint8), (7, k//16))
    bits[0, 0] = 128
    return codes, bits.view(ml_dtypes.float8_e4m3fn), np.array([.000123456789, .007654321], np.float64)


def assert_bits(got, expected):
    np.testing.assert_array_equal(got[0].view(np.uint8), expected[0].view(np.uint8))
    np.testing.assert_array_equal(got[1].view(np.uint32), expected[1].view(np.uint32))


def test_frontend_two_results_without_eager_oracle(monkeypatch):
    from tessera.compiler import nvfp4_expansion as reference
    from tessera.compiler.trace import trace, to_graph_ir_module
    def forbidden(*a, **kw):
        pytest.fail("Graph tracing called the eager expansion oracle")
    monkeypatch.setattr(reference, "reference_nvfp4_expand", forbidden)
    module = to_graph_ir_module(trace(expand._fn, *operands()), target="rocm_gfx1201")
    assert list(map(str, module.functions[0].result_types)) == [
        "tensor<7x48xf8E4M3FN>", "tensor<3x7xf32>"]
    assert [a.ir_type.dtype for a in module.functions[0].args] == [
        "uint8", "fp8_e4m3", "fp64"]


@DEVICE
@pytest.mark.parametrize("k", [16, 48, 80])
def test_public_jit_cached_image_and_serialized_replay(k, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import nvfp4_expansion as reference
    assert rt._rocm_live_arch() == "gfx1201"
    args = operands(k)
    expected = reference.reference_nvfp4_expand(*args, row_offsets=[0, 3, 7],
                                                numeric_policy=nvfp4_expansion_policy())
    def forbidden(*a, **kw):
        pytest.fail("compiled execution called the eager expansion oracle")
    monkeypatch.setattr(reference, "reference_nvfp4_expand", forbidden)
    first = expand(*args)
    artifact = expand.runtime_artifact()
    assert expand.execution_kind == "native_gpu"
    assert artifact.metadata["canonical_executable"]
    assert artifact.launch_descriptor.abi_id == NVFP4_EXPAND_ABI
    digest = artifact.native_image.image_digest
    warm = expand(globals_=args[2], scales=args[1], codes=args[0])
    assert expand.runtime_artifact().native_image.image_digest == digest
    assert_bits(first, expected)
    assert_bits(warm, expected)
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    names = [b.name for b in sorted(restored.launch_descriptor.buffers, key=lambda b: b.ordinal)]
    arrays = [*args, *[np.empty_like(x) for x in expected]]
    receipt = rt.launch(restored, {"buffers": dict(zip(names, arrays)), "scalars": {}})
    assert receipt.get("ok") and receipt["execution_kind"] == "native_gpu", receipt
    assert_bits(arrays[3:], expected)


@DEVICE
@pytest.mark.parametrize("kind", ["nan_scale", "negative_scale", "zero_global", "overflow", "underflow", "wrong_dtype"])
def test_public_invalid_inputs_do_not_reach_hip(kind, monkeypatch):
    from tessera import runtime as rt
    args = list(operands())
    if kind == "nan_scale":
        args[1][1, 0] = np.nan
    elif kind == "negative_scale":
        args[1][1, 0] = -1
    elif kind == "zero_global":
        args[2][0] = 0
    elif kind == "overflow":
        args[2][0] = np.finfo(np.float64).max
    elif kind == "underflow":
        args[2][0] = np.nextafter(0., 1.)
    else:
        args[0] = args[0].astype(np.int8)
    def forbidden(*a, **kw):
        pytest.fail("invalid import reached HIP")
    monkeypatch.setattr(rt, "_load_hip_for_launch", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="native|NVFP4|nvfp4|LEGALITY_TARGET_CAPABILITY"):
        expand(*args)


@DEVICE
def test_replay_rejects_alias_and_tampered_lineage_before_hip(monkeypatch):
    from tessera import runtime as rt
    args = operands(16)
    expand(*args)
    artifact = expand.runtime_artifact()
    desc = artifact.launch_descriptor
    names = [b.name for b in sorted(desc.buffers, key=lambda b: b.ordinal)]
    # Values alias the packed root's allocation while exposing the required
    # output shape/dtype; use a larger shared backing to keep both views valid.
    backing = np.zeros((7, 16), dtype=np.uint8)
    codes = backing[:, :8].copy()
    values = np.zeros((7, 16), dtype=ml_dtypes.float8_e4m3fn)
    scale_out = np.zeros((1, 7), np.float32)
    def forbidden(*a, **kw):
        pytest.fail("bad replay reached HIP")
    monkeypatch.setattr(rt, "_load_hip_for_launch", forbidden)
    damaged = replace(artifact, graph_ir=artifact.graph_ir + " ")
    receipt = rt.launch(damaged, {"buffers": dict(zip(names, [codes, args[1], args[2], values, scale_out])),
                                 "scalars": {}})
    assert not receipt["ok"] and "lineage" in receipt["reason"], receipt
    # Force a same-sized input/output overlap using K16/N7 scale allocation.
    global_backing = np.zeros(7, np.float64)
    globals_ = global_backing[:2]
    globals_[:] = [1., 2.]
    scale_alias = global_backing.view(np.float32)[:7].reshape(1, 7)
    receipt = rt.launch(artifact, {"buffers": dict(zip(names, [args[0], args[1], globals_, values, scale_alias])),
                                  "scalars": {}})
    assert not receipt["ok"] and "alias" in receipt["reason"], receipt


@DEVICE
def test_serialized_replay_in_fresh_process_without_compiler(tmp_path):
    import json
    import subprocess
    import sys
    from tessera.compiler.nvfp4_expansion import reference_nvfp4_expand
    args = operands(48)
    expected = reference_nvfp4_expand(*args, row_offsets=[0, 3, 7],
                                     numeric_policy=nvfp4_expansion_policy())
    expand(*args)
    (tmp_path / "artifact.json").write_text(expand.runtime_artifact().to_json())
    np.savez(tmp_path / "arrays.npz", codes=args[0], scale_bits=args[1].view(np.uint8),
             globals=args[2], value_bits=expected[0].view(np.uint8),
             plane_bits=expected[1].view(np.uint32))
    child = r'''
import json
from pathlib import Path
import sys
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler import nvfp4_expansion as reference
from tessera.compiler import rocm_nvfp4_expand_native as native
def forbidden(*a, **kw):
    raise AssertionError("serialized replay tried compilation or eager arithmetic")
reference.reference_nvfp4_expand = forbidden
native.package_nvfp4_expansion = forbidden
assert rt._rocm_live_arch() == "gfx1201"
root = Path(sys.argv[1])
artifact = rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
with np.load(root/"arrays.npz", allow_pickle=False) as data:
    codes = data["codes"]
    scales = data["scale_bits"].view(ml_dtypes.float8_e4m3fn)
    globals_ = data["globals"]
    values = np.empty(data["value_bits"].shape, dtype=ml_dtypes.float8_e4m3fn)
    plane = np.empty(data["plane_bits"].shape, np.float32)
    names = [b.name for b in sorted(artifact.launch_descriptor.buffers, key=lambda b: b.ordinal)]
    receipt = rt.launch(artifact, {"buffers": dict(zip(names, [codes,scales,globals_,values,plane])), "scalars": {}})
    assert receipt.get("ok") and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_array_equal(values.view(np.uint8), data["value_bits"])
    np.testing.assert_array_equal(plane.view(np.uint32), data["plane_bits"])
print(json.dumps({"architecture": "gfx1201", "fresh_process": True,
                  "compiler_and_oracle": "forbidden", "bitwise_outputs": "passed"}))
'''
    env = {**os.environ, "TESSERA_OPT": "/unavailable/tessera-opt",
           "TESSERA_ROCM_OPT": "/unavailable/tessera-rocm-opt"}
    result = subprocess.run([sys.executable, "-c", child, str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["bitwise_outputs"] == "passed"


def test_capability_proof_is_named_and_exact_architecture():
    from tessera.compiler.capabilities import TARGET_CAPABILITIES, supports_op
    from tessera.compiler.backend_manifest import manifest_for
    for target in TARGET_CAPABILITIES:
        result = supports_op(target, "tessera.nvfp4_expand", dtype="uint8", rank=2)
        if target == "rocm_gfx1201":
            assert result.supported and result.runtime_status == "ready"
            assert not supports_op(target, "tessera.nvfp4_expand", dtype="fp64", rank=2).supported
        else:
            assert not result.supported
    proved = [entry for entry in manifest_for("nvfp4_expand")
              if entry.status == "device_verified_jit"]
    assert len(proved) == 1 and proved[0].target == "rocm_gfx1201"
    assert proved[0].execute_compare_fixture == "tests/unit/test_rocm_nvfp4_expansion_jit.py"


@pytest.mark.parametrize("target", ["rocm_gfx1151", "nvidia_sm120", "apple_gpu", "x86"])
def test_unsupported_sibling_never_executes_eager_expansion(target, monkeypatch):
    from tessera.compiler import nvfp4_expansion as reference
    def forbidden(*a, **kw):
        pytest.fail("unsupported native target executed the eager expansion oracle")
    monkeypatch.setattr(reference, "reference_nvfp4_expand", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        sibling = ts.jit(expand._fn, target=target)
        sibling(*operands())
