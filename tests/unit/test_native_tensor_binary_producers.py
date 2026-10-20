"""Original binary producer SSA must survive the native tensor program route."""
import base64
from dataclasses import replace
import json
import re

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def lhs_add(a, residual, b):
    return ts.ops.matmul(ts.ops.add(a, residual), b, output_dtype="fp32")


def rhs_add(a, b, residual):
    return ts.ops.matmul(a, ts.ops.add(b, residual), output_dtype="fp32")


def both_add(a, residual_a, b, residual_b):
    return ts.ops.matmul(ts.ops.add(a, residual_a), ts.ops.add(b, residual_b),
                         output_dtype="fp32")


def add_then_relu(a, residual, b):
    return ts.ops.matmul(ts.ops.relu(ts.ops.add(a, residual)), b, output_dtype="fp32")


def duplicate_intermediate(a, residual, b):
    value = ts.ops.add(a, residual)
    return ts.ops.matmul(ts.ops.add(value, value), b, output_dtype="fp32")


CASES = {
    "lhs": (lhs_add, [(17, 32), (17, 32), (32, 11)], [[0, 1], [3, 2]]),
    "rhs": (rhs_add, [(17, 32), (32, 11), (32, 11)], [[1, 2], [0, 3]]),
    "both": (both_add, [(17, 32), (17, 32), (32, 11), (32, 11)],
             [[0, 1], [2, 3], [4, 5]]),
    "chain": (add_then_relu, [(17, 32), (17, 32), (32, 11)],
              [[0, 1], [3], [4, 2]]),
    "duplicate": (duplicate_intermediate, [(17, 32), (17, 32), (32, 11)],
                  [[0, 1], [3, 3], [4, 2]]),
}


def binary_source(case, storage):
    function, shapes, _ = CASES[case]
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    values = tuple(np.zeros(shape, dtype=dtype) for shape in shapes)
    fn = ts.jit(target="nvidia_sm120")(function)
    graph = fn._traced_autodiff_module(values, {})
    graph = replace(graph, module_attrs={**graph.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    return graph.to_mlir(target="nvidia_sm120", canonical=True)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_native_binary_program_query_retains_original_graph(case, storage):
    source = binary_source(case, storage)
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = true" in queried.stdout
    # Querying a clone must not replace the original binary computation.
    assert queried.stdout.count("tessera.add") == source.count("tessera.add")
    assert "tensor_stage_" not in queried.stdout


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_native_binary_program_frame_preserves_operand_roles(case, storage):
    compiled = _run(binary_source(case, storage),
                    "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert compiled.returncode == 0, compiled.stderr
    match = re.search(r'tessera.native.tensor_program_frame_json = "([^"]+)"',
                      compiled.stdout)
    assert match, "native Schedule/Tile must export the actual checked buffer frame"
    frame = json.loads(base64.b64decode(match[1]))
    _, shapes, expected_inputs = CASES[case]
    assert frame["roots"] == list(range(len(shapes)))
    assert [stage["inputs"] for stage in frame["stages"]] == expected_inputs
    assert frame["stages"][-1]["output"] == frame["output"]
    assert compiled.stdout.count("tile.program_invoke ") == len(expected_inputs)
    assert "tile.elementwise_kernel " in compiled.stdout
    # The ordinary half consumers keep typed fragments; FP32 uses its native
    # scalar-FMA consumer rather than being silently converted into half.
    if storage != "f32":
        assert "tile.view " in compiled.stdout and "tile.fragment_pack " in compiled.stdout


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_native_binary_standard_pipeline_reaches_target(case, storage):
    source = binary_source(case, storage)
    lowered = _run(source, "--tessera-nvidia-pipeline-sm120",
                   "--lower-tile-to-nvidia=sm=120")
    assert lowered.returncode == 0, lowered.stderr
    assert "tile.elementwise_kernel" not in lowered.stdout
    assert lowered.stdout.count("tessera_nvidia.program_invoke ") == len(CASES[case][2])
    required_threads = re.findall(
        r"nvvm.reqntid\s*=\s*array<i32:\s*128,\s*1,\s*1>", lowered.stdout)
    assert len(required_threads) == source.count("tessera.add")


def _binary_leaf():
    tiled = _run(binary_source("lhs", "f16"),
                 "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert tiled.returncode == 0, tiled.stderr
    projected = _run(tiled.stdout, "--tessera-schedule-to-tile=tensor-member=0")
    assert projected.returncode == 0, projected.stderr
    return projected.stdout


@pytest.mark.parametrize("change,diagnostic", [
    ("count", "binary frame buffers"),
    ("threads", "128-thread kernel"),
    ("storage", "binary frame buffers"),
    ("operand_order", "native parameter order"),
])
def test_binary_physical_contract_rejects_inconsistent_leaf(change, diagnostic):
    source = _binary_leaf()
    if change == "count":
        assert "elements = 544 : i64" in source
        source = source.replace("elements = 544 : i64", "elements = 543 : i64")
        source = re.sub(
            r"(tessera.native.program_scalars = array<i64:)\s*544>",
            r"\g<1> 543>", source)
    elif change == "threads":
        source, count = re.subn(
            r"(nvvm.reqntid = array<i32:)\s*128,\s*1,\s*1>",
            r"\g<1> 64, 1, 1>", source)
        assert count == 1
    elif change == "storage":
        assert 'storage = "f16"' in source
        source = source.replace('storage = "f16"', 'storage = "bf16"')
    else:
        source, count = re.subn(
            r"(tile.elementwise_kernel\s+)(%\w+),\s*(%\w+),",
            lambda match: match[1] + match[3] + ", " + match[2] + ",", source)
        assert count == 1
    refused = _run(source)
    assert refused.returncode != 0
    assert diagnostic in refused.stderr


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("profile_request", ["layout", "rhs_order", "accum"])
def test_binary_program_does_not_normalize_conflicting_matrix_contract(storage, profile_request):
    source = binary_source("lhs", storage)
    attributes = {
        "layout": 'tessera.layout = "col_major"',
        "rhs_order": 'rhs_storage_order = "col_major"',
        "accum": 'numeric_policy = {accum = "fp64"}',
    }
    marker = 'output_dtype = "fp32"'
    assert source.count(marker) == 1
    source = source.replace(marker, attributes[profile_request] + ", " + marker)
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = false" in queried.stdout
    assert ("col_major" if profile_request != "accum" else "fp64") in queried.stdout


def test_binary_program_does_not_claim_another_owners_add():
    source = r'''
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @unowned_add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %x = "tessera.add"(%a, %b) {tessera.effect_kind = "pure"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %x : tensor<2x3xf32>
  }
}
'''
    result = _run(source, "--tessera-graph-to-schedule")
    assert result.returncode == 0, result.stderr
    assert "tessera.add" in result.stdout
    assert "schedule.binary_kernel" not in result.stdout


def test_binary_target_does_not_claim_another_owners_tile_kernel():
    source = r'''
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  llvm.func @unowned_add(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr, %n: i64) attributes {nvvm.kernel} {
    tile.elementwise_kernel %a, %b, %out, %n {family = "binary", kind = "add", storage = "f32", output_storage = "f32"} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}
'''
    result = _run(source, "--lower-tile-to-nvidia=sm=120")
    assert result.returncode == 0, result.stderr
    # Preserve the pre-existing pass boundary. This is no execution claim for
    # the unowned carrier; its owning lowering is still required.
    assert "tile.elementwise_kernel" in result.stdout
