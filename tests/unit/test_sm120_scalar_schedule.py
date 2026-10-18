"""Native Graph/Schedule/Tile ancestry for mixed adjoint contractions."""
import subprocess

import pytest


def graph_source(a="f32", b="f16", output="f16", transpose_a=False, transpose_b=True,
                 extra=""):
    m, n, k = 17, 19, 35
    a_shape = f"{k}x{m}" if transpose_a else f"{m}x{k}"
    b_shape = f"{n}x{k}" if transpose_b else f"{k}x{n}"
    attributes = (f"transposeA = {str(transpose_a).lower()}, "
                  f"transposeB = {str(transpose_b).lower()}" + extra)
    return f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @adjoint_product(%a: tensor<{a_shape}x{a}>, %b: tensor<{b_shape}x{b}>) -> tensor<{m}x{n}x{output}> {{
    %r = "tessera.matmul"(%a, %b) {{{attributes}}} :
      (tensor<{a_shape}x{a}>, tensor<{b_shape}x{b}>) -> tensor<{m}x{n}x{output}>
    return %r : tensor<{m}x{n}x{output}>
  }}
}}"""


def invoke(compiler_toolchain, text, flag):
    tool = compiler_toolchain.require_tessera_opt(flag)
    return subprocess.run([str(tool), flag, "-"], input=text, text=True,
                          capture_output=True, timeout=60)


@pytest.mark.compiler_route
@pytest.mark.parametrize("half", ["f16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
def test_native_schedule_preserves_mixed_adjoint_contract(
    compiler_toolchain, half, side, output, orientation,
):
    a, b = ("f32", half) if side == "a" else (half, "f32")
    scheduled = invoke(compiler_toolchain, graph_source(a, b, output, *orientation),
                       "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    assert 'physical_contract = "nvidia_sm120_scalar_fma_v1"' in scheduled.stdout
    assert f'storage = "{a}"' in scheduled.stdout
    assert f'storage_b = "{b}"' in scheduled.stdout
    tile = invoke(compiler_toolchain, scheduled.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "tile.scalar_matmul_kernel" in tile.stdout
    assert f"a_element_type = {a}" in tile.stdout
    assert f"b_element_type = {b}" in tile.stdout
    assert f"output_element_type = {output}" in tile.stdout
    target = invoke(compiler_toolchain, tile.stdout, "--lower-tile-to-nvidia=sm=120")
    assert target.returncode == 0, target.stderr
    assert "tile.scalar_matmul_kernel" not in target.stdout
    assert "fma" in target.stdout and "mma.sync" not in target.stdout


@pytest.mark.compiler_route
def test_scalar_schedule_rejects_semantic_policy_substitution(compiler_toolchain):
    graph = graph_source(extra=', numeric_policy = {accum = "fp32", execution_mode = "approximate"}')
    result = invoke(compiler_toolchain, graph, "--tessera-graph-to-schedule")
    assert result.returncode != 0


@pytest.mark.compiler_route
def test_scalar_schedule_contract_is_bound_to_original_graph(compiler_toolchain):
    scheduled = invoke(compiler_toolchain, graph_source(), "--tessera-graph-to-schedule")
    assert scheduled.returncode == 0, scheduled.stderr
    altered = scheduled.stdout.replace(
        'physical_contract = "nvidia_sm120_scalar_fma_v1"', 'physical_contract = ""')
    assert altered != scheduled.stdout
    result = invoke(compiler_toolchain, altered, "--tessera-schedule-to-tile")
    assert result.returncode != 0


@pytest.mark.compiler_route
def test_scalar_schedule_preserves_explicit_rowmajor_rhs(compiler_toolchain):
    implicit = invoke(compiler_toolchain, graph_source(), "--tessera-graph-to-schedule")
    explicit = invoke(compiler_toolchain, graph_source(extra=', rhs_storage_order = "row_major"'),
                      "--tessera-graph-to-schedule")
    assert implicit.returncode == 0, implicit.stderr
    assert explicit.returncode == 0, explicit.stderr
    implicit_tile = invoke(compiler_toolchain, implicit.stdout, "--tessera-schedule-to-tile")
    explicit_tile = invoke(compiler_toolchain, explicit.stdout, "--tessera-schedule-to-tile")
    assert implicit_tile.returncode == explicit_tile.returncode == 0
    assert implicit_tile.stdout == explicit_tile.stdout
