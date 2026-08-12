"""MODEL-FUSED-PHYS-1 digest-bound MiniMax MSA package boundary."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    TENSOR_OPAQUE,
)
from tessera.compiler.model_attention import build_msa_physical_package
from tessera.stdlib import attention


def _graph() -> GraphIRModule:
    return GraphIRModule(functions=[GraphIRFunction(
        "minimax_msa",
        args=[
            IRArg("Q", TENSOR_OPAQUE),
            IRArg("K", TENSOR_OPAQUE),
            IRArg("V", TENSOR_OPAQUE),
        ],
        body=[IROp(
            "O",
            "tessera.msa_sparse_attention",
            ["%Q", "%K", "%V"],
            ["tensor<*xf32>", "tensor<*xf32>", "tensor<*xf32>"],
            "tensor<*xf32>",
            kwargs={
                "block_size": 2,
                "top_k": 2,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
                "head_dim": 8,
                "mode": "prefill",
                "causal": True,
            },
        )],
        return_values=["%O"],
    )])


@pytest.mark.parametrize(
    "target,arch,marker,lane",
    [
        ("x86", "zen5_avx512", "tessera_x86.kernel", "x86_msa_compiled"),
        (
            "rocm_gfx1151",
            "gfx1151",
            "tessera_rocm.msa_block_sparse",
            "rocm_sparse_attn_compiled",
        ),
    ],
)
def test_msa_package_binds_exact_schedule_tile_target_lineage(
    target, arch, marker, lane
):
    package = build_msa_physical_package(_graph(), target=target)
    package.validate()
    assert package.contract["architecture"] == arch
    assert package.contract["runtime_lane"] == lane
    assert package.contract["graph_redispatch"] == "prohibited"
    assert package.contract["graph_digest"] in package.schedule_ir
    assert "schedule.attn.kv_outer_sparse" in package.schedule_ir
    assert "tessera_attn.msa_kv_outer_sparse" in package.tile_ir
    assert marker in package.target_ir
    runtime_artifact = package.runtime_artifact()
    assert not (runtime_artifact.metadata or {}).get("ops")


def test_msa_package_rejects_target_artifact_substitution():
    package = build_msa_physical_package(_graph(), target="x86")
    object.__setattr__(package, "target_ir", package.target_ir + "\n// changed")
    with pytest.raises(ValueError, match="stale target_digest"):
        package.validate()


def test_x86_runtime_consumes_bound_launch_contract_without_graph_ops(monkeypatch):
    package = build_msa_physical_package(_graph(), target="x86")
    artifact = package.runtime_artifact()
    rng = np.random.default_rng(17)
    q = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    v = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)

    def flash(qv, kv, vv, *, scale, causal, attn_bias):
        del causal
        keys = np.repeat(kv, qv.shape[1] // kv.shape[1], axis=1)
        values = np.repeat(vv, qv.shape[1] // vv.shape[1], axis=1)
        scores = np.einsum("bhqd,bhkd->bhqk", qv, keys) * scale + attn_bias
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.einsum("bhqk,bhkd->bhqd", probs, values)

    monkeypatch.setattr(rt, "_x86_flash_attn", flash)
    got = rt._execute_x86_compiled_msa(artifact, (q, k, v))
    expected = attention.msa_sparse_attention(
        q, k, v, block_size=2, top_k=2, causal=True
    )
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.hardware_rocm
def test_gfx1151_runtime_consumes_exact_msa_artifact_without_graph_redispatch():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    package = build_msa_physical_package(_graph(), target="rocm_gfx1151")
    rng = np.random.default_rng(23)
    q = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    v = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    result = rt.launch(package.runtime_artifact(), (q, k, v))
    assert result["ok"] and result["execution_kind"] == "native_gpu", result.get(
        "reason"
    )
    expected = attention.msa_sparse_attention(
        q, k, v, block_size=2, top_k=2, causal=True
    )
    np.testing.assert_allclose(result["output"], expected, rtol=2e-5, atol=2e-5)
