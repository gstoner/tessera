import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt
from tessera.compiler import execution_matrix as EM
from tessera.compiler.backend_manifest import manifest_for
from tessera.nn import varlen as V


HELPERS = (
    "memory_index_score",
    "msa_index_scores",
    "varlen_sdpa",
    "score_combine",
)


def _artifact(target, path, op_name, arg_names, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": path,
        "executable": True,
        "arg_names": list(arg_names),
        "output_name": "o",
        "ops": [{
            "op_name": f"tessera.{op_name}",
            "result": "o",
            "operands": list(arg_names),
            "kwargs": dict(kwargs),
        }],
    })


def _case(op_name):
    rng = np.random.default_rng(101)
    if op_name == "score_combine":
        base = (rng.standard_normal((3, 4)) * 0.2).astype(np.float32)
        delta = (rng.standard_normal((3, 4)) * 0.1).astype(np.float32)
        kwargs = {"gamma": 1.25}
        return (base, delta), ("base", "delta"), kwargs, ts.ops.score_combine(
            base, delta, **kwargs)
    if op_name == "memory_index_score":
        keys = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
        query = rng.standard_normal((1, 2, 3, 5)).astype(np.float32)
        kwargs = {}
        return (keys, query), ("keys", "query"), kwargs, ts.ops.memory_index_score(
            keys, query)
    if op_name == "msa_index_scores":
        q = rng.standard_normal((1, 4, 5, 6)).astype(np.float32)
        k = rng.standard_normal((1, 2, 9, 6)).astype(np.float32)
        kwargs = {"block_size": 4}
        return (q, k), ("q", "k"), kwargs, ts.ops.msa_index_scores(q, k, **kwargs)
    if op_name == "varlen_sdpa":
        q = rng.standard_normal((2, 5, 4)).astype(np.float32)
        k = rng.standard_normal((2, 7, 4)).astype(np.float32)
        v = rng.standard_normal((2, 7, 4)).astype(np.float32)
        cu_q = V.cu_seqlens_from_lengths([2, 3])
        cu_k = V.cu_seqlens_from_lengths([3, 4])
        kwargs = {"causal": True}
        ref = ts.ops.varlen_sdpa(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, **kwargs)
        return (q, k, v, cu_q, cu_k), ("q", "k", "v", "cu_q", "cu_k"), kwargs, ref
    raise AssertionError(op_name)


@pytest.mark.parametrize(
    ("target", "path", "execution_kind"),
    (
        ("x86", "x86_composite_helper_compiled", "native_cpu"),
        ("rocm", "rocm_composite_helper_compiled", "reference_cpu"),
    ),
)
@pytest.mark.parametrize("op_name", HELPERS)
def test_composite_helper_launch_matches_reference(target, path, execution_kind, op_name):
    args, arg_names, kwargs, ref = _case(op_name)
    res = rt.launch(_artifact(target, path, op_name, arg_names, kwargs), args)

    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == path
    assert res["execution_kind"] == execution_kind
    np.testing.assert_allclose(res["output"], ref, rtol=2e-5, atol=2e-5)


def test_execution_matrix_has_composite_helper_rows():
    assert "x86_composite_helper_compiled" in EM.KNOWN_EXECUTORS
    assert "rocm_composite_helper_compiled" in EM.KNOWN_EXECUTORS
    assert EM.lookup("x86", "x86_composite_helper_compiled").executable is True
    assert EM.lookup("rocm", "rocm_composite_helper_compiled").executable is True


@pytest.mark.parametrize("op_name", HELPERS)
def test_composite_helper_manifest_backend_parity(op_name):
    entries = {e.target: e for e in manifest_for(op_name)}
    assert entries["apple_gpu"].status == "fused"
    assert entries["x86"].status == "device_verified_jit"
    assert entries["x86"].execute_compare_fixture == (
        "tests/unit/test_composite_helper_backend_parity.py")
    assert entries["rocm"].status == "device_verified_jit"
    assert entries["rocm"].execute_compare_fixture == (
        "tests/unit/test_composite_helper_backend_parity.py")
    assert entries["nvidia_sm90"].status == "artifact_only"
