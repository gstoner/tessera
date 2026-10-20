"""Bounded producer ownership must retain the original dynamic Graph source."""
import base64
import json
import re

import pytest

from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def bounded_binary_source(storage):
    # Textual frontend is an equally authoritative source. A and residual have
    # equal runtime rows; the declared M bound is capacity, not the active M.
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @bounded_add_matmul(%a: tensor<?x32x{storage}>, %r: tensor<?x32x{storage}>, %b: tensor<32x11x{storage}>) -> tensor<?x11xf32> {{
    %x = "tessera.add"(%a, %r) {{tessera.effect_kind = "pure"}} : (tensor<?x32x{storage}>, tensor<?x32x{storage}>) -> tensor<?x32x{storage}>
    %y = "tessera.matmul"(%x, %b) {{output_dtype = "fp32", rhs_storage_order = "row_major", shape_bounds = [17 : i64, 11 : i64, 32 : i64], tessera.effect_kind = "pure"}} : (tensor<?x32x{storage}>, tensor<32x11x{storage}>) -> tensor<?x11xf32>
    return %y : tensor<?x11xf32>
  }}
}}'''


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_bounded_binary_query_preserves_original_dynamic_source(storage):
    source = bounded_binary_source(storage)
    result = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert result.returncode == 0, result.stderr
    assert "tessera.native.tensor_program_admission = true" in result.stdout
    assert f"tensor<?x32x{storage}>" in result.stdout
    assert "tensor<?x11xf32>" in result.stdout
    assert "tessera.add" in result.stdout and "tensor_stage_" not in result.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_bounded_binary_standard_pipeline_owns_both_native_stages(storage):
    result = _run(bounded_binary_source(storage), "--tessera-nvidia-pipeline-sm120",
                  "--lower-tile-to-nvidia=sm=120")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("tessera_nvidia.program_invoke ") == 2
    records = re.findall(r'tessera.native.tensor_program_frame_json = "([^"]+)"', result.stdout)
    assert len(records) == 1, "the complete producer program needs one native controller"
    frame = json.loads(base64.b64decode(records[0]))
    assert frame["schema"] == "tessera.native.tensor_program_frame.v2"
    assert frame["roots"] == [0, 1, 2]
    assert [stage["inputs"] for stage in frame["stages"]] == [[0, 1], [3, 2]]
    assert frame["stages"][-1]["output"] == frame["output"]
    # A bounded owner must distinguish capacity from active view geometry;
    # the static v1 frame cannot express runtime equality and derived launches.
    assert frame["buffers"][0]["shape"] == [17, 32]
    assert frame["buffers"][frame["output"]]["shape"] == [17, 11]


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_bounded_program_packages_original_source_and_dynamic_member_images(storage):
    from tessera.compiler.native_tensor_program import compile_native_tensor_program

    program = compile_native_tensor_program(bounded_binary_source(storage))
    assert program is not None
    assert program.frame["schema"] == "tessera.native.tensor_program_frame.v2"
    assert program.frame["row_bound"] == 17
    assert program.frame["row_root"] == 0
    assert program.frame["row_buffers"] == [1, 1, 0, 1, 1]
    producer, consumer = program.frame["stages"]
    assert producer["scalar_row_multipliers"] == [32]
    assert producer["grid_row_multipliers"] == [32, 0, 0]
    assert producer["grid_row_divisors"] == [128, 1, 1]
    assert consumer["scalars"] == [17, 11, 32, 32, 11, 11]
    assert consumer["scalar_row_multipliers"] == [1, 0, 0, 0, 0, 0]
    assert consumer["grid_row_multipliers"] == [0, 1, 0]
    assert consumer["grid_row_divisors"] == [1, 16, 1]
    assert len(program.images) == 2 and all(b".target sm_120a" in image for image in program.images)
    assert f"tensor<?x32x{storage}>" in program.source_ir
    assert "tile.view " in program.tile_ir and "tile.fragment_pack " in program.tile_ir
    assert "memref<?x32x" in program.target_ir


@pytest.mark.parametrize("original,replacement", [
    ("tessera.native.tensor_program_row_root = 0 : i64", "tessera.native.tensor_program_row_root = 3 : i64"),
    ("tessera.native.tensor_program_row_bound = 17 : i64", "tessera.native.tensor_program_row_bound = 16 : i64"),
    ("tessera.native.tensor_program_row_buffers = array<i64: 1, 1, 0, 1, 1>",
     "tessera.native.tensor_program_row_buffers = array<i64: 1, 0, 0, 1, 1>"),
    ("scalar_row_multipliers = array<i64: 32>", "scalar_row_multipliers = array<i64: 33>"),
    ("grid_row_divisors = array<i64: 128, 1, 1>", "grid_row_divisors = array<i64: 0, 1, 1>"),
])
def test_bounded_controller_rejects_inconsistent_active_contract(original, replacement):
    lowered = _run(bounded_binary_source("f16"), "--tessera-nvidia-pipeline-sm120",
                   "--lower-tile-to-nvidia=sm=120")
    assert lowered.returncode == 0, lowered.stderr
    assert original in lowered.stdout
    changed = _run(lowered.stdout.replace(original, replacement))
    assert changed.returncode != 0
    assert "bounded" in changed.stderr


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("trace_rows", [1, 17])
def test_public_bound_policy_is_projected_by_native_compiler(storage, trace_rows):
    import ml_dtypes
    import numpy as np
    import tessera as ts
    from tests.unit.test_native_tensor_binary_producers import lhs_add

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    values = (np.zeros((trace_rows, 35), dtype=dtype),
              np.zeros((trace_rows, 35), dtype=dtype), np.zeros((35, 19), dtype=dtype))
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(lhs_add)
    try:
        program = fn.compile_native_tensor_program(*values)
        assert program.frame["row_bound"] == 65
        assert program.frame["row_buffers"] == [1, 1, 0, 1, 1]
        assert f"tensor<{trace_rows}x35x{storage}>" in program.source_ir
        assert "tessera.native.tensor_program_row_bound_request = 65 : i64" in program.source_ir
        assert f"tensor<?x35x{storage}>" in program.schedule_ir
        assert program.frame["stages"][1]["scalars"] == [65, 19, 35, 35, 19, 19]
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_bounded_host_ssa_distinguishes_active_roots_and_owned_capacity(storage):
    lowered = _run(bounded_binary_source(storage), "--tessera-graph-to-schedule")
    assert lowered.returncode == 0, lowered.stderr
    assert re.search(r"^\s*func.func @bounded_add_matmul\(%\w+: memref<\?x32x" + storage + r">", lowered.stdout, re.M)
    assert lowered.stdout.count("memref.dim ") == 1
    assert lowered.stdout.count("memref.subview ") == 2
    assert f"memref.alloc() : memref<17x32x{storage}>" in lowered.stdout
    assert "memref.alloc() : memref<17x11xf32>" in lowered.stdout
    assert f"to memref<?x32x{storage}>" in lowered.stdout
    assert "to memref<?x11xf32>" in lowered.stdout


@pytest.mark.parametrize("change", ["capacity_roots", "row_source", "zero_view", "view_deallocation", "early_retirement", "malformed_later_view"])
def test_bounded_active_ssa_cannot_erase_extents_or_escape_lifetime(change):
    lowered = _run(bounded_binary_source("f16"), "--tessera-graph-to-schedule")
    assert lowered.returncode == 0, lowered.stderr
    source = lowered.stdout
    if change == "capacity_roots":
        source = source.replace("memref<?x32xf16>", "memref<17x32xf16>")
    elif change == "row_source":
        source = source.replace("memref.dim %arg0,", "memref.dim %arg1,", 1)
    elif change == "zero_view":
        rows = re.search(r"(%\w+) = memref.dim ", source)[1]
        zero = re.search(r"(%\w+) = arith.constant 0 : index", source)[1]
        source = source.replace(f"[{rows}, 32]", f"[{zero}, 32]", 1)
    elif change == "view_deallocation":
        view, allocation = re.search(r"(%\w+) = memref.subview (%\w+).* : memref<17x32xf16> to memref<\?x32xf16>", source).groups()
        source = source.replace(f"memref.dealloc {allocation} : memref<17x32xf16>",
                                f"memref.dealloc {view} : memref<?x32xf16>", 1)
    else:
        lines = source.splitlines(keepends=True)
        invocations = [i for i, line in enumerate(lines) if "tile.program_invoke " in line]
        assert len(invocations) == 2
        if change == "early_retirement":
            release = next(i for i, line in enumerate(lines) if "memref.dealloc " in line)
            line = lines.pop(release)
            lines.insert(invocations[1], line)
        else:
            lines.insert(invocations[1], '    %bad_view = "memref.subview"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -1, 32>, static_strides = array<i64: 1, 1>}> : () -> memref<?x32xf16>\n')
        source = "".join(lines)
    assert source != lowered.stdout
    refused = _run(source)
    assert refused.returncode > 0, refused.stderr  # a diagnostic, never SIGABRT/SIGSEGV


def test_static_native_frame_cannot_acquire_bounded_kernel_metadata():
    from tests.unit.test_native_tensor_binary_producers import binary_source

    lowered = _run(binary_source("lhs", "f16"), "--tessera-graph-to-schedule",
                   "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert lowered.returncode == 0, lowered.stderr
    changed = lowered.stdout.replace("nvvm.kernel", "nvvm.kernel, tessera.native.program_grid_row_multipliers = array<i64: 0, 1, 0>", 1)
    assert changed != lowered.stdout
    refused = _run(changed)
    assert refused.returncode > 0
    assert "static frame cannot carry bounded" in refused.stderr
