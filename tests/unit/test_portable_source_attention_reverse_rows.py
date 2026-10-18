"""Compiler-free v2 source-AD replay guards; PTX stubs are never executed."""
from copy import deepcopy
import json
import re
import subprocess

import pytest

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.portable_source_attention_reverse import restore, seal, sha
from tessera.compiler.prepared_source_attention_reverse import ReverseMember
from tests.unit.test_native_source_attention_reverse_rows import rows_source
from tests.unit.test_portable_source_attention_reverse import reseal

pytestmark = pytest.mark.compiler_route


@pytest.fixture(scope="module", params=["f16", "bf16"])
def portable_rows(compiler_toolchain, request):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    program = export_native_source_attention_reverse(rows_source(request.param), tool=tool)
    members = []
    for i, (graph, schedule, tile, target, native) in enumerate(program.lower_members()):
        entry = re.findall(r"\bllvm.func @(\w+)\(", tile)[0]
        params = ",\n".join(".param .u64 p" + str(j) for j in range((4, 12, 14, 16, 5)[i]))
        image = (".version 9.0\n.target sm_120a\n.address_size 64\n.visible .entry "
                 + entry + "(\n" + params + "\n) { ret; }\n").encode()
        members.append(ReverseMember(graph, schedule, tile, target, native, entry, image, sha(image.decode())))
    return seal(program.manifest, members)


def test_rows_restore_is_compiler_and_provider_free(portable_rows, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    def forbidden(*args, **kwargs):
        pytest.fail("offline rows restore invoked compiler/provider")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", forbidden)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    data, plan, members, dims = restore(json.loads(json.dumps(portable_rows)))
    assert data["schema"] == "tessera.nvidia.source_attention_reverse_program.v2"
    assert len(members) == 5 and len(plan["buffers"]) == 13
    assert dims == (1, 2, 1, 3, 5, 4, 3)
    assert data["members"][2]["abi"]["block"] == [32, 1, 1]
    assert data["members"][3]["abi"]["block"] == [128, 1, 1]


@pytest.mark.parametrize("fault", [
    "schema", "lifetime", "row_storage", "rounded_output_edge", "missing_row_member",
    "row_block", "backward_grid", "row_scale", "row_causal", "request", "ptx_width"])
def test_rows_rehashed_inconsistency_rejected_before_provider(portable_rows, fault, monkeypatch):
    from tessera import runtime as rt
    data = deepcopy(portable_rows)
    plan, records = data["plan"], data["members"]
    if fault == "schema":
        data["schema"] = "tessera.nvidia.source_attention_reverse_program.v1"
    elif fault == "lifetime":
        plan["buffers"][8]["last_read"] = 2
    elif fault == "row_storage":
        plan["buffers"][8]["storage"] = "f16"
        plan["buffers"][8]["bytes"] //= 2
    elif fault == "rounded_output_edge":
        plan["steps"][2]["inputs"][4] = 6
    elif fault == "missing_row_member":
        records.pop(2)
    elif fault == "row_block":
        records[2]["abi"]["block"][0] = 128
    elif fault == "backward_grid":
        records[3]["abi"]["grid"][0] += 1
    elif fault == "row_scale":
        records[2]["tile_ir"] = records[2]["tile_ir"].replace("scale = 5.000000e-01", "scale = 1.000000e+00")
    elif fault == "row_causal":
        records[2]["tile_ir"] = records[2]["tile_ir"].replace("causal = false", "causal = true")
    elif fault == "request":
        plan["source_graph_ir"] = plan["source_graph_ir"].replace(
            "tessera.native.source_attention_reverse_rows_request, ", "")
    elif fault == "ptx_width":
        records[3]["ptx"] = records[3]["ptx"].replace(".param .u64 p0", ".param .u32 p0")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid rows program reached provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    with pytest.raises(ValueError):
        restore(reseal(data))

