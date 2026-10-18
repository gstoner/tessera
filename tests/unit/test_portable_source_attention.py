"""Host-free: portable source-attention schema and ABI validation without tools."""
import copy
import lzma
from pathlib import Path
import subprocess

import pytest

from tessera import runtime as rt
from tessera.compiler.native_artifact import ArtifactContractError
from tessera.compiler.portable_source_attention import digest, restore
from tessera.compiler.source_attention_runtime import _checked, execute

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/native_attention_prefix_20261018/portable"


@pytest.fixture(params=["f16", "bf16"])
def artifact(request):
    return rt.RuntimeArtifact.from_json(lzma.decompress(
        (PACKET / (request.param + ".runtime.json.xz")).read_bytes()))


def forbid_tools(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("portable validation invoked a compiler or native provider")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)


def test_portable_native_members_validate_without_compiler_or_device(artifact, monkeypatch):
    forbid_tools(monkeypatch)
    data = artifact.metadata["native_program"]
    restored, plan, members, dims = restore(data)
    assert restored["contract_digest"] == data["contract_digest"]
    assert plan["source_graph_ir"] == artifact.graph_ir
    assert dims == (1, 2, 1, 3, 5, 4, 3)
    assert len(members) == 2
    assert members[1].descriptor.provenance["row_lse"] == "f32[B,Hq,Sq]"


@pytest.mark.parametrize("mutation", [
    "private_lifetime", "norm_input_dtype", "norm_epsilon", "attention_guard",
    "native_architecture", "crossed_images", "member_graph",
])
def test_rehashed_portable_contract_rejects_semantic_mismatch(artifact, monkeypatch, mutation):
    forbid_tools(monkeypatch)
    original = artifact.metadata["native_program"]
    _checked(original)  # Prime the certificate cache before changing the payload.
    data = copy.deepcopy(original)
    if mutation == "private_lifetime":
        data["plan"]["buffers"][3]["last_read"] = 0
    elif mutation == "norm_input_dtype":
        data["members"][0]["descriptor"]["buffers"][0]["dtype"] = "fp32"
    elif mutation == "norm_epsilon":
        data["members"][0]["descriptor"]["provenance"]["epsilon"] = 0.5
    elif mutation == "attention_guard":
        data["members"][1]["descriptor"]["shape_guards"][0]["value"] = 99
    elif mutation == "native_architecture":
        data["members"][0]["image"]["architecture"] = "sm_90"
    elif mutation == "crossed_images":
        data["members"][0]["image"] = copy.deepcopy(data["members"][1]["image"])
    else:
        data["members"][0]["graph_ir"] += "\n"
    for member in data["members"]:
        member["descriptor"].pop("descriptor_digest", None)
    data["contract_digest"] = digest({k: v for k, v in data.items() if k != "contract_digest"})
    with pytest.raises((ValueError, ArtifactContractError)):
        _checked(data)


@pytest.mark.parametrize("mutation", ["target", "graph", "arguments", "placement"])
def test_parent_artifact_mismatch_fails_before_native_acquisition(artifact, monkeypatch, mutation):
    forbid_tools(monkeypatch)
    metadata = copy.deepcopy(artifact.metadata)
    graph = artifact.graph_ir
    if mutation == "target":
        metadata["target"] = "rocm_gfx1201"
    elif mutation == "graph":
        graph += "\n"
    elif mutation == "arguments":
        metadata["arg_names"] = ["v", "q", "k"]
    else:
        metadata["result_placement"] = "cuda"
    changed = rt.RuntimeArtifact(graph_ir=graph, metadata=metadata)
    with pytest.raises(ValueError, match="parent Graph"):
        execute(changed, (None, None, None))
