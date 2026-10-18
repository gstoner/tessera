"""Frozen compiler-produced source reverse artifact checks; no device/compiler."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from tessera import runtime as rt
from tessera.compiler.portable_nvidia_source_reverse import restore
from tessera.compiler.portable_nvidia_tensor_jvp import digest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/native_sm120/source_reverse_f16.json"


def artifact():
    return rt.RuntimeArtifact.from_json(FIXTURE.read_text())


def reseal(data):
    for member in data["members"]:
        member["member_digest"] = digest({k: v for k, v in member.items() if k != "member_digest"})
    data["contract_digest"] = digest({k: v for k, v in data.items() if k != "contract_digest"})


def test_portable_source_reverse_restores_offline(monkeypatch):
    with monkeypatch.context() as offline:
        offline.setattr(subprocess, "run", lambda *a, **k: pytest.fail("offline restore ran compiler"))
        offline.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("offline restore loaded device"))
        native, members = restore(artifact().metadata["native_program"])
    assert native.manifest["schema"] == "tessera.native.sm120_source_reverse_outline.v1"
    assert native.manifest["outputs"] == [4] and len(members) == 2
    assert members[0]["abi"]["parameters"] == ["ptr"] * 3 + ["i64"] * 3
    assert members[1]["abi"]["parameters"] == ["ptr"] * 3 + ["i64"] * 2
    assert members[0]["abi"]["block_size"] == 32
    assert members[1]["abi"]["block_size"] == 128


@pytest.mark.parametrize("fault", [
    "schema", "member_count", "image_hash", "target", "parameter_count", "block",
    "dimensions", "input_edge", "graph", "native_entry", "row_schema", "row_body",
    "resource_bool", "seed_dtype", "lifetime", "plan_bool", "contract_digest",
])
def test_portable_source_reverse_rejects_mutations_before_provider(fault, monkeypatch):
    parent = artifact()
    data = copy.deepcopy(parent.metadata["native_program"])
    member = data["members"][0]
    if fault == "schema":
        data["schema"] = "tessera.nvidia.tensor_active_source_jvp.v1"
    elif fault == "member_count":
        data["members"].pop()
    elif fault == "image_hash":
        member["image_digest"] = "0" * 64
    elif fault == "target":
        member["ptx"] = member["ptx"].replace("sm_120a", "sm_90a")
        member["image_digest"] = digest({"not": "a source reverse image"})
    elif fault == "parameter_count":
        member["abi"]["parameters"].pop()
    elif fault == "block":
        member["abi"]["block_size"] = 128
    elif fault == "dimensions":
        member["abi"]["dimensions"] = [17, 19, 35]
    elif fault == "input_edge":
        member["abi"]["inputs"] = [0, 1]
    elif fault == "graph":
        member["graph_ir"] = "module {}"
    elif fault == "native_entry":
        member["entry"] += "_other"
    elif fault == "row_schema":
        data["members"][1]["tile_ir"] = data["members"][1]["tile_ir"].replace("schema = 6", "schema = 1")
    elif fault == "row_body":
        data["members"][1]["target_ir"] = data["members"][1]["target_ir"].replace("tile_body_hash", "old_body_hash")
    elif fault == "resource_bool":
        member["resources"]["registers_per_thread"] = True
    elif fault in ("seed_dtype", "lifetime", "plan_bool"):
        plan = json.loads(data["plan_json"])
        if fault == "seed_dtype":
            plan["buffers"][2]["storage"] = "f16"
        elif fault == "lifetime":
            plan["buffers"][3]["last_read"] = 0
        else:
            plan["buffers"][0]["id"] = False
        data["plan_json"] = json.dumps(plan)
    elif fault == "contract_digest":
        data["contract_digest"] = "0" * 64
    if fault != "contract_digest":
        reseal(data)
    metadata = {**parent.metadata, "native_program": data}
    altered = rt.RuntimeArtifact(graph_ir=parent.graph_ir, metadata=metadata)
    from tessera.compiler.nvidia_source_reverse_runtime import execute
    with monkeypatch.context() as refused:
        refused.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("invalid artifact reached provider"))
        refused.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid artifact ran compiler"))
        with pytest.raises(ValueError):
            execute(altered, (None, None, None))


def test_composite_vjp_declaration_is_exact_target_and_real_graph_ops():
    from tessera.compiler.native_vjp_composites import native_vjp_composite_declarations
    from tessera.compiler.op_catalog import OP_SPECS
    declarations = native_vjp_composite_declarations()
    assert "source_normalization_matmul" in declarations
    declaration = declarations["source_normalization_matmul"]
    declaration.validate()
    assert set(declaration.target_consumers) == {"nvidia_sm120"}
    assert declaration.graph_consumers == ("tessera.rmsnorm", "tessera.relu", "tessera.matmul")
    for name in declaration.graph_consumers:
        assert name.removeprefix("tessera.") in OP_SPECS


def test_composite_execution_does_not_advertise_a_single_norm_contract():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("nvidia_sm120", "nvidia_sm120_source_reverse_compiled")
    assert row is not None and row.op_family == "source_normalization_matmul"
    assert row.backward_aliases == ()
    assert row.device_proof == "device_verified_jit"


def test_cached_metadata_uses_full_payload_and_an_owned_snapshot(monkeypatch):
    from tessera.compiler import nvidia_source_reverse_runtime as runtime
    data = copy.deepcopy(artifact().metadata["native_program"])
    runtime._certificates.clear()
    original_restore = runtime.restore
    calls = []
    def counted(value):
        calls.append(True)
        return original_restore(value)
    monkeypatch.setattr(runtime, "restore", counted)
    cached = runtime._checked(data)
    assert runtime._checked(copy.deepcopy(data)) is cached
    assert len(calls) == 1
    data["members"][0]["abi"]["block_size"] = 128
    assert cached[2][0]["abi"]["block_size"] == 32
    # Retaining the old display digest must never hit the valid cache entry.
    with pytest.raises(ValueError):
        runtime._checked(data)
    assert len(calls) == 2


def test_cached_metadata_is_bounded():
    from tessera.compiler import nvidia_source_reverse_runtime as runtime
    from tessera.compiler.portable_nvidia_source_reverse import manifest, seal_members
    data = artifact().metadata["native_program"]
    original = json.loads(data["plan_json"])
    records = [{k: v for k, v in member.items() if k not in {"abi", "provenance", "member_digest"}}
               for member in data["members"]]
    runtime._certificates.clear()
    for index in range(runtime._CERTIFICATE_LIMIT + 2):
        plan = copy.deepcopy(original)
        plan["source_graph_ir"] += "\n" * (index + 1)
        text = json.dumps(plan)
        payload = manifest(text, seal_members(text, records))
        runtime._checked(payload)
    assert len(runtime._certificates) == runtime._CERTIFICATE_LIMIT
