"""Compiler-free structural replay checks; PTX stubs here are never executed."""
from copy import deepcopy
import json
import subprocess

import pytest

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.portable_source_attention_reverse import digest, restore, seal, sha
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse, ReverseMember
from tests.unit.test_native_source_attention_reverse_outline import source

pytestmark = pytest.mark.compiler_route


def portable_program(program):
    """Construct signature-only PTX stubs; they are never device evidence."""
    import re
    members = []
    for graph, schedule, tile, target, native in program.lower_members():
        entry = re.findall(r'\bllvm.func @(\w+)\(', tile)[0]
        signature = re.search(r'llvm.func @' + re.escape(entry) + r'\(([^)]*)\)', tile)
        assert signature is not None
        count = len(signature[1].split(",")) if signature[1].strip() else 0
        params = ",\n".join(".param .u64 p" + str(j) for j in range(count))
        image = (".version 9.0\n.target sm_120a\n.address_size 64\n.visible .entry "
                 + entry + "(\n" + params + "\n) { ret; }\n").encode()
        members.append(ReverseMember(graph, schedule, tile, target, native, entry, image, sha(image.decode())))
    return seal(program.manifest, members)


@pytest.fixture(scope="module")
def portable(compiler_toolchain):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    return portable_program(export_native_source_attention_reverse(source(), tool=tool))


def reseal(data):
    for field in ("source", "paired", "program"):
        data["plan"][field + "_graph_sha256"] = sha(data["plan"][field + "_graph_ir"])
    for record in data["members"]:
        record["image_digest"] = sha(record["ptx"])
        record["provenance"] = {f + "_sha256": sha(record[f]) for f in (
            "graph_ir", "schedule_ir", "tile_ir", "target_ir", "native_ir")}
        record["member_digest"] = digest({k: v for k, v in record.items() if k != "member_digest"})
    data["contract_digest"] = digest({k: v for k, v in data.items() if k != "contract_digest"})
    return data


def test_restore_needs_neither_compiler_nor_provider(portable, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    def forbidden(*args, **kwargs):
        pytest.fail("offline restore invoked compiler/provider")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", forbidden)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    data, plan, members, dims = restore(json.loads(json.dumps(portable)))
    assert dims == (1, 2, 1, 3, 5, 4, 3) and len(members) == 4
    assert data["plan"] == plan
    with PreparedSourceAttentionReverse.from_manifest(data) as owner:
        assert owner._lib is None and owner._handle == 0
        mutated = owner.manifest()
        mutated["plan"]["outputs"].reverse()
        assert owner.manifest()["plan"]["outputs"] == [11, 9, 10]


@pytest.mark.parametrize("fault", ["lifetime", "seed_order", "gradient_order", "bool_index", "root_call",
    "program_return", "extra_operation", "source_epsilon", "source_causal", "checkpoint_scale",
    "checkpoint_seed", "row_geometry", "row_target_seal", "abi_geometry", "block_size", "ptx_width",
    "ptx_target", "ptx_entry", "local_bindings", "original_result_type", "source_dropout", "source_softcap",
    "source_accum", "source_causal_type", "source_scale_type", "source_norm_eps_type", "source_seed_type"])
def test_rehashed_inconsistent_contract_rejected_before_provider(portable, fault, monkeypatch):
    from tessera import runtime as rt
    data = deepcopy(portable)
    plan, records = data["plan"], data["members"]
    if fault == "lifetime":
        plan["buffers"][5]["last_read"] = 1
    elif fault == "seed_order":
        plan["cotangent_indices"].reverse()
    elif fault == "gradient_order":
        plan["outputs"].reverse()
    elif fault == "bool_index":
        plan["requested_gradients"][0] = False
    elif fault == "root_call":
        plan["program_graph_ir"] = plan["program_graph_ir"].replace("__member_0(%arg0)", "__member_0(%arg1)", 1)
    elif fault == "program_return":
        plan["program_graph_ir"] = plan["program_graph_ir"].replace("return %3, %2#1, %2#2", "return %3, %2#2, %2#1", 1)
    elif fault == "extra_operation":
        plan["program_graph_ir"] = plan["program_graph_ir"].replace("  func.func @", "  unknown.operation\n  func.func @", 1)
    elif fault == "source_epsilon":
        plan["source_graph_ir"] = plan["source_graph_ir"].replace("eps = 1.000000e-05", "eps = 1.000000e-01", 1)
    elif fault == "source_causal":
        plan["source_graph_ir"] = plan["source_graph_ir"].replace('head_dim = 4', 'causal = true, head_dim = 4', 1)
    elif fault in ("source_dropout", "source_softcap"):
        field = "dropout_p" if fault == "source_dropout" else "softcap"
        plan["source_graph_ir"] = plan["source_graph_ir"].replace('head_dim = 4', field + ' = 5.000000e-01 : f32, head_dim = 4', 1)
    elif fault in ("source_accum", "source_norm_eps_type"):
        old, new = ('accum = "fp32"', 'accum = "fp16"') if fault == "source_accum" else (
            'eps = 1.000000e-05 : f64', 'eps = "bad"')
        plan["source_graph_ir"] = plan["source_graph_ir"].replace(old, new, 1)
        records[0]["graph_ir"] = records[0]["graph_ir"].replace(old, new, 1)
        plan["member_graphs"][0] = records[0]["graph_ir"]
        plan["program_graph_ir"] = plan["program_graph_ir"].replace(old, new, 1)
    elif fault in ("source_causal_type", "source_scale_type", "source_seed_type"):
        name = {"source_causal_type": "causal", "source_scale_type": "scale", "source_seed_type": "dropout_seed"}[fault]
        plan["source_graph_ir"] = plan["source_graph_ir"].replace('head_dim = 4', name + ' = "bad", head_dim = 4', 1)
    elif fault == "checkpoint_scale":
        records[2]["tile_ir"] = records[2]["tile_ir"].replace("scale = 5.000000e-01", "scale = 2.500000e-01")
    elif fault == "checkpoint_seed":
        records[2]["tile_ir"] = records[2]["tile_ir"].replace("lse_cotangent = true", "lse_cotangent = false")
    elif fault == "row_geometry":
        records[3]["tile_ir"] = records[3]["tile_ir"].replace("rows = 6 : i64", "rows = 7 : i64")
    elif fault == "row_target_seal":
        records[3]["target_ir"] = records[3]["target_ir"].replace("tile_body_hash = \"", "tile_body_hash = \"0", 1)
    elif fault == "abi_geometry":
        records[0]["abi"]["dimensions"][0] += 1
    elif fault == "block_size":
        records[1]["abi"]["block"][0] = 32
    elif fault == "ptx_width":
        records[2]["ptx"] = records[2]["ptx"].replace(".param .u64", ".param .u32", 1)
    elif fault == "ptx_target":
        records[0]["ptx"] = records[0]["ptx"].replace(".target sm_120a", ".target sm_80")
    elif fault == "ptx_entry":
        records[0]["ptx"] = records[0]["ptx"].replace(records[0]["entry"], "wrong_entry")
    elif fault == "local_bindings":
        records[0]["graph_ir"] = records[0]["graph_ir"].replace('"input_0"', '"wrong_input"')
        plan["member_graphs"][0] = records[0]["graph_ir"]
        plan["program_graph_ir"] = plan["program_graph_ir"].replace('"input_0"', '"wrong_input"', 1)
    else:
        plan["source_graph_ir"] = plan["source_graph_ir"].replace("tensor<1x2x3x3xf16>", "tensor<1x2x3x3xf32>", 1)
    def forbidden(*args, **kwargs):
        pytest.fail("inconsistent manifest reached CUDA provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    with pytest.raises(ValueError):
        PreparedSourceAttentionReverse.from_manifest(reseal(data))


def test_component_receipt_abi_does_not_alias_validated_cache(portable, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import source_attention_reverse_runtime as native
    import os
    class MockOwner:
        _closed = False
        receipt = {"execution_kind": "native_gpu"}
        def __call__(self, *args):
            return (0,)
    owner = MockOwner()
    monkeypatch.setitem(native._live, (os.getpid(), portable["contract_digest"]), owner)
    artifact = rt.RuntimeArtifact(graph_ir=portable["plan"]["source_graph_ir"], metadata={
        "target": "nvidia_sm120", "compiler_path": native.COMPILER_PATH,
        "arg_names": list(native.ARGUMENTS), "native_program": portable})
    # Mock provider plumbing only; no device execution/attestation is claimed.
    first = native.execute(artifact, (None,) * 5)
    first.component_receipts[0]["abi"]["block"][0] = 1
    second = native.execute(artifact, (None,) * 5)
    assert second.component_receipts[0]["abi"]["block"] == [128, 1, 1]
