from __future__ import annotations

import copy

import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.native_jvp import (
    NativeJVPArtifact,
    build_native_jvp_artifact,
    child_digest,
)
from tessera.compiler.native_jvp_plugins import (
    native_jvp_plugin_owners,
    plan_native_jvp_family,
)
from tessera.compiler.graph_ir import IROp
from tessera.compiler.graph_ir import _format_attr_value, tensor_ir_type


def _reduce_child() -> dict:
    return {
        "target": "x86",
        "compiler_path": "x86_reduce_compiled",
        "executable": True,
        "execution_kind": "native_cpu",
        "execution_mode": "cpu_avx512",
        "arg_names": ["x"],
        "ops": [{
            "op_name": "tessera.sum",
            "result": "out",
            "operands": ["x"],
            "kwargs": {"axis": -1},
        }],
    }


def _package() -> NativeJVPArtifact:
    child = _reduce_child()
    steps = [
        {"id": "primal", "child_digest": child_digest(child),
         "child_metadata": child, "inputs": ["primal_input"], "output_index": -1},
        {"id": "tangent", "child_digest": child_digest(child),
         "child_metadata": child, "inputs": ["tangent_input"], "output_index": -1},
    ]
    return build_native_jvp_artifact(
        target="x86",
        architecture="zen5_avx512",
        family="reduce",
        source_graph_ir='module attributes {tessera.frontend.authority = "tracer"} {}',
        paired_jvp_ir="func.func @f__jvp()",
        wrt_indices=(0,),
        arg_names=("primal_input", "tangent_input"),
        steps=steps,
    )


def test_native_jvp_contract_is_content_addressed_and_tamper_evident():
    package = _package()
    package.validate()
    stale = copy.deepcopy(dict(package.contract))
    stale["family"] = "spectral"
    with pytest.raises(ValueError, match="stale content identity"):
        NativeJVPArtifact(stale).validate()


def test_graph_float_attributes_use_mlir_scientific_syntax():
    assert _format_attr_value(1e-5) == "1.0e-05"


def test_canonical_complex_dtype_uses_builtin_mlir_element_syntax():
    ty = tensor_ir_type(("4",), "complex64")
    assert ty.dtype == "complex64"
    assert str(ty) == "tensor<4xcomplex<f32>>"


def test_native_jvp_rejects_nontracer_graph_authority():
    child = _reduce_child()
    with pytest.raises(ValueError, match="tracer-owned"):
        build_native_jvp_artifact(
            target="x86", architecture="zen5_avx512", family="reduce",
            source_graph_ir="module {}", paired_jvp_ir="func.func @f__jvp()",
            wrt_indices=(0,), arg_names=("x",),
            steps=({"id": "primal", "child_digest": child_digest(child),
                    "child_metadata": child, "inputs": ["x"]},),
        )


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1250", "unknown"])
def test_rocm_native_jvp_fails_closed_without_gfx1151_proof(architecture: str):
    child = _reduce_child()
    with pytest.raises(ValueError, match="fail closed"):
        build_native_jvp_artifact(
            target="rocm", architecture=architecture, family="reduce",
            source_graph_ir='module attributes {tessera.frontend.authority = "tracer"} {}',
            paired_jvp_ir="func.func @f__jvp()", wrt_indices=(0,),
            arg_names=("primal_input", "tangent_input"),
            steps=({"id": "primal", "child_digest": child_digest(child),
                    "child_metadata": child, "inputs": ["primal_input"]},),
        )


def test_runtime_executes_bound_primal_and_tangent_children(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_execute_x86_compiled_reduce",
        lambda artifact, args: np.asarray(args[0]).sum(axis=-1),
    )
    package = _package()
    result = runtime.launch(
        runtime.RuntimeArtifact(metadata=package.runtime_metadata()),
        (np.arange(6, dtype=np.float32).reshape(2, 3),
         np.ones((2, 3), dtype=np.float32)),
    )
    assert result["ok"]
    assert result["execution_mode"] == "cpu_avx512"
    primal, tangent = result["output"]
    np.testing.assert_array_equal(primal, [3.0, 12.0])
    np.testing.assert_array_equal(tangent, [3.0, 3.0])


def test_runtime_rejects_child_metadata_substitution():
    package = _package()
    stale = copy.deepcopy(dict(package.contract))
    stale["steps"][0]["child_metadata"]["compiler_path"] = "native_cpu"
    # Rehashing only the parent cannot forge the independently bound child.
    stale.pop("artifact_hash")
    import hashlib
    import json
    stale["artifact_hash"] = hashlib.sha256(
        json.dumps(stale, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = runtime.launch(
        runtime.RuntimeArtifact(metadata={
            **package.runtime_metadata(), "native_jvp": stale,
        }),
        (np.ones((2, 3), np.float32), np.ones((2, 3), np.float32)),
    )
    assert not result["ok"]
    assert "child package digest mismatch" in result["reason"]


def test_native_jvp_families_have_one_explicit_plugin_owner():
    owners = native_jvp_plugin_owners()
    assert owners["fft"] == "_plan_fft"
    assert owners["dct"] == "_plan_dct"
    assert owners["spectral_conv"] == "_plan_compound_spectral"
    assert owners["layer_norm"] == "_plan_normalization"
    assert len(owners) == len(set(owners))


def test_jvp_plugin_produces_digest_bound_plan_outside_jitfn():
    source = IROp(
        result="out", op_name="tessera.sum", operands=["%x"],
        operand_types=["tensor<2x3xf32>"], result_type="tensor<2xf32>",
        kwargs={"axis": -1},
    )
    plan = plan_native_jvp_family(
        source=source,
        primal_inputs=(np.ones((2, 3), dtype=np.float32),),
        wrt_indices=(0,), target="x86", architecture="zen5_avx512",
        execution_mode="cpu_avx512",
    )
    assert plan.family == "reduce"
    assert [step["id"] for step in plan.steps] == ["primal", "tangent"]
    assert all(len(step["child_digest"]) == 64 for step in plan.steps)
