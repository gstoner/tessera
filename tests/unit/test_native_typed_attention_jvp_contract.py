"""Typed native JVP roles and numerical policy are sealed before physical lowering."""
import json
import re

import pytest
from tessera.compiler.native_attention_jvp import _lower_graph
from tessera.compiler.native_gpu_storage import _decode_image
from tessera.compiler.scheduled_matmul import run_tessera_opt
from tests.device.nvidia.test_native_typed_attention_jvp import graph

pytestmark = pytest.mark.compiler_route


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide_output", [False, True])
def test_native_typed_jvp_manifest_has_role_specific_dtypes(storage, wide_output, compiler_toolchain):
    tool = compiler_toolchain.require_tessera_opt("tessera-graph-to-schedule", "tessera-schedule-to-tile")
    output = "f32" if wide_output else storage
    ir = _lower_graph(graph(storage, output, True), compiler=tool)
    match = re.search(r'tessera\.native_tensor_contract = "((?:\\.|[^"\\])*)"', ir)
    assert match
    manifest = json.loads(_decode_image(match[1]).decode())
    root = "fp16" if storage == "f16" else "bf16"
    result = "fp32" if wide_output else root
    assert [row["dtype"] for row in manifest["arguments"] if row["kind"] == "tensor"] == (
        [root]*3 + [result, "fp32"] + [root]*3 + [result])
    assert ir.count("tile.alloc_shared") == (2 if wide_output else 3)


@pytest.mark.parametrize("fault", ["output", "lse", "direction"])
def test_typed_jvp_rejects_inconsistent_storage_roles(fault, compiler_toolchain):
    tool = compiler_toolchain.require_tessera_opt("tessera-graph-to-schedule", "tessera-schedule-to-tile")
    source = graph("f16", "bf16" if fault == "output" else "f16", True)
    if fault == "lse":
        source = source.replace("tensor<1x2x3xf32>", "tensor<1x2x3xf16>")
    elif fault == "direction":
        q, kv, wrong = "tensor<1x2x3x16xf16>", "tensor<1x1x5x16xf16>", "tensor<1x2x3x16xbf16>"
        source = source.replace("%dq: "+q, "%dq: "+wrong)
        old = f",{q},{kv},{kv}) -> {q}"
        assert old in source
        source = source.replace(old, f",{wrong},{kv},{kv}) -> {q}")
    with pytest.raises(RuntimeError):
        _lower_graph(source, compiler=tool)


@pytest.mark.parametrize("old,new", [
    (', storage = "f16"', ', storage = "bf16"'),
    ('row_lse_storage = "f32"', 'row_lse_storage = "f16"'),
    ('algorithm = "cooperative_saved_lse_unrounded_value_moments_v1"',
     'algorithm = "cooperative_saved_lse_moments_v1"'),
])
def test_typed_jvp_schedule_replay_seals_storage_and_correction_policy(old, new, compiler_toolchain):
    tool = compiler_toolchain.require_tessera_opt("tessera-graph-to-schedule", "tessera-schedule-to-tile")
    schedule = run_tessera_opt(tool, graph("f16", "f16", True), "--tessera-graph-to-schedule")
    assert old in schedule
    with pytest.raises(RuntimeError, match="contract changed"):
        run_tessera_opt(tool, schedule.replace(old, new, 1), "--tessera-schedule-to-tile")
