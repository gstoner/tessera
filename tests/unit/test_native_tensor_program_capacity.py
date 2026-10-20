"""Native admission and typed programs obey the retained frame ABI capacities."""
import re

import pytest
from tests.unit.test_native_tensor_program_pipeline import _run, relu_product, traced_source

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def chain_source(stages, *, huge_rhs=False):
    a = "tensor<1x2147483647xf16>" if huge_rhs else "tensor<1x16xf16>"
    b = "tensor<2147483647x2147483647xf16>" if huge_rhs else "tensor<16x1xf16>"
    out = "tensor<1x2147483647xf32>" if huge_rhs else "tensor<1x1xf32>"
    value, storage = ("%b", b) if huge_rhs else ("%a", a)
    lines = ['module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {',
             f"func.func @chain(%a: {a}, %b: {b}) -> {out} {{"]
    for index in range(stages - 1):
        result = f"%p{index}"
        lines.append(f'  {result} = "tessera.relu"({value}) : ({storage}) -> {storage}')
        value = result
    lhs, rhs = ("%a", value) if huge_rhs else (value, "%b")
    lines += [f'  %o = "tessera.matmul"({lhs}, {rhs}) {{output_dtype = "fp32"}} : ({a}, {b}) -> {out}',
              f"  return %o : {out}", "}", "}"]
    return "\n".join(lines)


@pytest.mark.parametrize("stages", [64, 65])
def test_native_admission_matches_frame_stage_capacity(stages):
    queried = _run(chain_source(stages), "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert f"tessera.native.tensor_program_admission = {str(stages <= 64).lower()}" in queried.stdout
    scheduled = _run(chain_source(stages), "--tessera-graph-to-schedule")
    if stages <= 64:
        assert scheduled.returncode == 0, scheduled.stderr
        assert scheduled.stdout.count("tile.program_invoke ") == stages
    else:
        assert scheduled.returncode != 0
        assert "64 stages and 128 buffers" in scheduled.stderr


def add_roots(source, count, storage):
    match = re.search(r"(?m)^(\s*func.func @relu_product\()([^)]*)", source)
    assert match
    extra = "".join(f", %unused_{index}: {storage}" for index in range(count))
    return source[:match.end()] + extra + source[match.end():]


@pytest.mark.parametrize("extra", [124, 125])
def test_typed_program_matches_frame_buffer_capacity(extra):
    scheduled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    changed = add_roots(scheduled.stdout, extra, "memref<1x1xf16>")
    verified = _run(changed, "--verify-each")
    if extra == 124:
        assert verified.returncode == 0, verified.stderr
    else:
        assert verified.returncode != 0
        assert "64 stages and 128 buffers" in verified.stderr


@pytest.mark.parametrize("extra", [2, 3])
def test_typed_program_checks_total_aligned_arena_without_allocating(extra):
    scheduled = _run(traced_source(relu_product, "f16"), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    changed = add_roots(scheduled.stdout, extra, "memref<2147483647x2147483647xf16>")
    verified = _run(changed, "--verify-each")
    if extra == 2:
        assert verified.returncode == 0, verified.stderr
    else:
        assert verified.returncode != 0
        assert "aligned arena exceeds its 64-bit capacity" in verified.stderr


def test_typed_program_checks_stage_count_independently_of_graph_admission():
    scheduled = _run(chain_source(64), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    line = next(line for line in scheduled.stdout.splitlines() if "tile.program_invoke " in line)
    added = line.replace(", %alloc ", ", %extra_stage ")
    original = line.replace(" %arg0,", " %extra_stage,")
    changed = scheduled.stdout.replace(line,
        "    %extra_stage = memref.alloc() : memref<1x16xf16>\n" + added + "\n" + original, 1)
    changed = re.sub(r"(?m)^(    return .*)$",
                     "    memref.dealloc %extra_stage : memref<1x16xf16>\n" + r"\1", changed, count=1)
    verified = _run(changed, "--verify-each")
    assert verified.returncode != 0
    assert "64 stages and 128 buffers" in verified.stderr


def test_native_admission_checks_arena_before_member_generation():
    source = chain_source(3, huge_rhs=True)
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = false" in queried.stdout
    scheduled = _run(source, "--tessera-graph-to-schedule")
    assert scheduled.returncode != 0
    assert "aligned arena exceeds its 64-bit capacity" in scheduled.stderr
