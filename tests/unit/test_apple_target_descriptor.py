"""Phase 3 — Apple GPU target descriptor + tensor ABI + contract rules.

Tests the explicit Apple GPU target-descriptor surface: the three distinct
execution contracts (metal_artifact / metal_runtime / mtl4_runtime), the
capability-gated Metal-4 requirements, the explicit tensor/buffer ABI, the
contract guards, and the wiring into Target IR, RuntimeArtifact, query_backend,
and the capability snapshot.
"""

from __future__ import annotations

import pytest

from tessera.compiler import apple_target_descriptor as atd
from tessera.compiler.target_ir import lower_tile_to_target_ir
from tessera.compiler.tile_ir import TileFunction, TileIRModule, TileOp


# ── descriptor + required capabilities ───────────────────────────────────────


def test_descriptor_fixed_identity_fields():
    d = atd.apple_target_descriptor(atd.METAL_RUNTIME)
    assert d["vendor"] == "apple"
    assert d["api"] == "metal"
    assert d["triple"] == "air64-apple-macosx"
    assert d["arch"] == "apple-metal"
    assert d["memory_model"] == "unified_64"
    assert d["execution_contract"] == "metal_runtime"


@pytest.mark.parametrize("ec", atd.EXECUTION_CONTRACTS)
def test_descriptor_accepts_each_contract(ec):
    assert atd.apple_target_descriptor(ec)["execution_contract"] == ec


def test_descriptor_rejects_unknown_contract():
    with pytest.raises(atd.AppleTargetError):
        atd.apple_target_descriptor("vulkan_runtime")


def test_required_caps_classic_lanes_need_no_mtl4_gates():
    # metal_artifact + metal_runtime are classic lanes (no MTL4 command model).
    assert atd.required_capabilities_for(atd.METAL_ARTIFACT) == []
    assert atd.required_capabilities_for(atd.METAL_RUNTIME) == []


def test_required_caps_mtl4_base_and_cooperative():
    base = atd.required_capabilities_for(atd.MTL4_RUNTIME)
    assert set(base) == {
        atd.CAP_COMMAND_QUEUE,
        atd.CAP_COMMAND_ALLOCATOR,
        atd.CAP_COMPILER,
    }
    coop = atd.required_capabilities_for(atd.MTL4_RUNTIME, cooperative_tensor=True)
    # Cooperative matmul ALSO requires the tensor + MSL 4.0 gates.
    assert atd.CAP_TENSOR in coop
    assert atd.CAP_MSL_4_0 in coop
    assert set(base).issubset(coop)


def test_descriptor_rejects_unknown_required_capability():
    with pytest.raises(atd.AppleTargetError):
        atd.apple_target_descriptor(atd.MTL4_RUNTIME, required_capabilities=["bogus_cap"])


# ── tensor / buffer ABI ──────────────────────────────────────────────────────


def test_tensor_abi_row_major_default_strides():
    abi = atd.apple_tensor_abi("fp32", (4, 8))
    assert abi["rank"] == 2
    assert abi["shape"] == [4, 8]
    assert abi["strides"] == [8, 1]  # row-major
    assert abi["offset_bytes"] == 0
    assert abi["resource_kind"] == atd.RESOURCE_MTL_BUFFER
    assert abi["resident"] is False
    assert abi["view"] == "buffer"


def test_tensor_abi_explicit_strides_offset_and_tensor_view():
    abi = atd.apple_tensor_abi(
        "bf16",
        (2, 3, 4),
        strides=[12, 4, 1],
        offset_bytes=64,
        resource_kind=atd.RESOURCE_MTL_TENSOR_VIEW,
        resident=True,
    )
    assert abi["dtype"] == "bf16"
    assert abi["strides"] == [12, 4, 1]
    assert abi["offset_bytes"] == 64
    assert abi["resource_kind"] == "mtl_tensor_view"
    assert abi["resident"] is True
    assert abi["view"] == "tensor"  # tensor-view is a tensor, not a plain buffer


def test_tensor_abi_mtlpackage_tensor_is_tensor_view():
    abi = atd.apple_tensor_abi("fp16", (1, 16), resource_kind=atd.RESOURCE_MTLPACKAGE_TENSOR)
    assert abi["view"] == "tensor"


def test_tensor_abi_rejects_bad_rank_and_kind():
    with pytest.raises(atd.AppleTargetError):
        atd.apple_tensor_abi("fp32", (4, 8), strides=[1])  # rank mismatch
    with pytest.raises(atd.AppleTargetError):
        atd.apple_tensor_abi("fp32", (4,), resource_kind="cuda_global")


# ── contract validation ──────────────────────────────────────────────────────


def test_validate_mtl4_runtime_requires_command_model():
    bad = {
        "execution_contract": atd.MTL4_RUNTIME,
        "required_capabilities": [atd.CAP_TENSOR],  # missing the command-model trio
    }
    with pytest.raises(atd.AppleTargetError):
        atd.validate_descriptor(bad)
    good = atd.apple_target_descriptor(atd.MTL4_RUNTIME)
    assert atd.validate_descriptor(good) is good


def test_validate_classic_lane_must_not_smuggle_mtl4_caps():
    # A metal_runtime (classic MPS) descriptor must not require MTL4 command-model
    # caps — that would imply the wrong lane (MPSGraph != MTL4 command model).
    sneaky = {
        "execution_contract": atd.METAL_RUNTIME,
        "required_capabilities": [atd.CAP_COMMAND_QUEUE],
    }
    with pytest.raises(atd.AppleTargetError):
        atd.validate_descriptor(sneaky)


def test_validate_observed_capabilities_gate():
    desc = atd.apple_target_descriptor(atd.MTL4_RUNTIME, cooperative_tensor=True)
    # Host lacks the tensor gate -> not executable.
    observed = {c: True for c in atd.APPLE_CAPABILITIES}
    observed[atd.CAP_TENSOR] = False
    with pytest.raises(atd.AppleTargetError):
        atd.validate_descriptor(desc, observed_capabilities=observed)
    # Full host -> ok.
    full = {c: True for c in atd.APPLE_CAPABILITIES}
    assert atd.validate_descriptor(desc, observed_capabilities=full) is desc


def test_artifact_only_must_not_claim_runtime_contract():
    # The key contract guard: artifact-only modules must not claim mtl4_runtime.
    md_bad = {
        "runtime_status": "artifact_only",
        "target_descriptor": atd.apple_target_descriptor(atd.MTL4_RUNTIME),
    }
    with pytest.raises(atd.AppleTargetError):
        atd.assert_not_artifact_claiming_runtime(md_bad)
    md_bad_rt = {
        "runtime_status": "artifact_only",
        "target_descriptor": {"execution_contract": atd.METAL_RUNTIME},
    }
    with pytest.raises(atd.AppleTargetError):
        atd.assert_not_artifact_claiming_runtime(md_bad_rt)
    # artifact_only + metal_artifact is consistent.
    md_ok = {
        "runtime_status": "artifact_only",
        "target_descriptor": atd.apple_target_descriptor(atd.METAL_ARTIFACT),
    }
    atd.assert_not_artifact_claiming_runtime(md_ok)  # no raise


# ── Target IR emission ───────────────────────────────────────────────────────


def _apple_gpu_runtime_tile():
    return TileIRModule(functions=[
        TileFunction("flash", body=[
            TileOp("tessera.attn.online_softmax", {
                "source": "tessera.flash_attn", "result": "O",
                "ordinal": 0, "policy": "safe",
            })
        ])
    ])


def _apple_gpu_artifact_tile():
    # An unrecognized op -> not an MPS-runtime pattern -> metal_artifact.
    return TileIRModule(functions=[
        TileFunction("misc", body=[TileOp("tile.unknown_op", {"result": "x"})])
    ])


def test_target_ir_apple_gpu_runtime_emits_metal_runtime_descriptor():
    target = lower_tile_to_target_ir(_apple_gpu_runtime_tile(), target_kind="apple_gpu")
    desc = target.attrs["target_descriptor"]
    assert desc["vendor"] == "apple" and desc["api"] == "metal"
    assert desc["triple"] == "air64-apple-macosx"
    # execution_mode == execution_contract; the MPS lane is metal_runtime, never mtl4.
    assert target.attrs["execution_mode"] == "metal_runtime"
    assert desc["execution_contract"] == "metal_runtime"
    assert desc["execution_contract"] != atd.MTL4_RUNTIME


def test_target_ir_apple_gpu_artifact_emits_metal_artifact_descriptor():
    target = lower_tile_to_target_ir(_apple_gpu_artifact_tile(), target_kind="apple_gpu")
    desc = target.attrs["target_descriptor"]
    assert target.attrs["execution_mode"] == "metal_artifact"
    assert desc["execution_contract"] == "metal_artifact"
    # A pure artifact requires no MTL4 gates.
    assert desc["required_capabilities"] == []
    assert target.verify().ok  # descriptor doesn't break verification


# ── RuntimeArtifact round-trip ───────────────────────────────────────────────


def test_runtime_artifact_json_roundtrip_preserves_descriptor_caps_abi():
    from tessera.runtime import RuntimeArtifact

    desc = atd.apple_target_descriptor(atd.MTL4_RUNTIME, cooperative_tensor=True)
    abi = atd.apple_tensor_abi(
        "bf16", (4, 8), resource_kind=atd.RESOURCE_MTL_TENSOR_VIEW, resident=True
    )
    art = RuntimeArtifact(
        target_ir="...",
        metadata={
            "target": "apple_gpu",
            "target_descriptor": desc,
            "required_capabilities": desc["required_capabilities"],
            "tensor_abi": [abi],
        },
    )
    back = RuntimeArtifact.from_json(art.to_json())
    assert back.metadata["target_descriptor"] == desc
    assert back.metadata["required_capabilities"] == desc["required_capabilities"]
    assert back.metadata["tensor_abi"] == [abi]


# ── compile() containerization (artifact-only, no runtime claim) ─────────────


def test_compile_apple_gpu_is_artifact_only_with_metal_artifact_descriptor():
    from tessera.runtime import compile as rt_compile

    art = rt_compile("some target ir", target="apple_gpu")
    md = art.metadata
    assert md["runtime_status"] == "artifact_only"
    assert md["target_descriptor"]["execution_contract"] == "metal_artifact"
    assert md["required_capabilities"] == []
    # No runtime probe ran during compile -> no observed capabilities claimed.
    assert "observed_capabilities" not in md
    # And the contract guard passes (artifact-only is consistent with metal_artifact).
    atd.assert_not_artifact_claiming_runtime(md)


# ── observed-capability snapshot + query_backend ─────────────────────────────


def test_capability_snapshot_is_always_explicit():
    from tessera._apple_gpu_dispatch import apple_gpu_capabilities_snapshot

    snap = apple_gpu_capabilities_snapshot()
    assert isinstance(snap, dict)
    assert "runtime_available" in snap
    assert isinstance(snap["capabilities"], dict)
    if not snap["runtime_available"]:
        # No silent "Metal 4 full" claim when the runtime is unavailable.
        assert snap["capabilities"] == {}
        assert snap["mtl4_full"] is False
    else:
        # When available, capability keys decode the MTL4 bit names.
        decoded = set(snap["capabilities"])
        assert {"mtl4_command_queue", "mtl_tensor", "msl_4_0"}.issubset(decoded)


def test_query_backend_apple_gpu_includes_observed_capabilities():
    from tessera.runtime import query_backend

    info = query_backend("apple_gpu")
    assert "observed_capabilities" in info
    snap = info["observed_capabilities"]
    assert "runtime_available" in snap and "capabilities" in snap
    # cpu must NOT carry the apple snapshot.
    assert "observed_capabilities" not in query_backend("cpu")
