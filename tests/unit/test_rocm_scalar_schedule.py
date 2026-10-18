"""ROCm native mixed contractions preserve FP32 operands through final stores."""
import subprocess

import pytest

from tests.unit.test_sm120_scalar_schedule import graph_source


@pytest.mark.compiler_route
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("half", ["f16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
def test_rocm_mixed_graph_reaches_native_scalar_contraction(
        compiler_toolchain, arch, half, side, output, orientation):
    a, b = ("f32", half) if side == "a" else (half, "f32")
    graph = graph_source(a, b, output, *orientation).replace(
        "nvidia_sm120", "rocm_" + arch).replace('"sm_120"', '"' + arch + '"')
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    def run(text, flag):
        result = subprocess.run([str(tool), flag, "-"], input=text, text=True,
                                capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return result.stdout
    schedule = run(graph, "--tessera-graph-to-schedule")
    assert f'physical_contract = "rocm_{arch}_scalar_fma_v1"' in schedule
    tile = run(schedule, "--tessera-schedule-to-tile")
    assert "tile.scalar_matmul_kernel" in tile
    assert f"a_element_type = {a}" in tile
    assert f"b_element_type = {b}" in tile
    assert f"output_element_type = {output}" in tile
    target = run(tile, "--lower-tile-to-rocm=arch=" + arch)
    assert "tile.scalar_matmul_kernel" not in target
    assert "wmma" not in target.lower()
    assert "scalar_matmul" in target or "fma" in target

    gpu = run(target, "--generate-rocm-gemm-f32-kernel")
    assert "gpu.func" in gpu and "math.fma" in gpu
    assert "tessera_rocm.scalar_matmul" not in gpu
    assert f"memref<?x{a}>" in gpu
    assert f"memref<?x{b}>" in gpu
    assert f"memref<?x{output}>" in gpu


@pytest.mark.compiler_route
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("mutation", ["arch", "digest", "geometry"])
def test_rocm_scalar_tile_rejects_contract_mutation(compiler_toolchain, arch, mutation):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    source = graph_source().replace("nvidia_sm120", "rocm_" + arch).replace(
        '"sm_120"', '"' + arch + '"')
    for flag in ("--tessera-graph-to-schedule", "--tessera-schedule-to-tile"):
        result = subprocess.run([str(tool), flag, "-"], input=source, text=True,
                                capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        source = result.stdout
    if mutation == "arch":
        selected = "gfx1201" if arch == "gfx1151" else "gfx1151"
    else:
        selected = arch
        if mutation == "digest":
            import re
            source, count = re.subn(r'tessera.schedule_hash = "[a-f0-9]{64}"',
                                    'tessera.schedule_hash = "invalid"', source)
            assert count == 1
        else:
            source = source.replace("tessera.macro_tile_m = 16", "tessera.macro_tile_m = 32")
    result = subprocess.run([str(tool), "--lower-tile-to-rocm=arch=" + selected, "-"],
                            input=source, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "ROCm scalar contraction requires" in result.stderr


@pytest.mark.compiler_route
@pytest.mark.parametrize("target,arch", [("rocm_gfx1151", "gfx1151"),
    ("rocm_gfx1201", "gfx1201"), ("nvidia_sm120", "sm_120")])
@pytest.mark.parametrize("position", ["producer", "result"])
def test_standalone_scalar_schedule_cannot_discard_unhandled_graph_work(compiler_toolchain, target, arch, position):
    source = graph_source(a="f32", b="f16", output="f32", transpose_a=False, transpose_b=False)
    source = source.replace("nvidia_sm120", target).replace('"sm_120"', '"' + arch + '"')
    if position == "producer":
        source = source.replace('    %r = "tessera.matmul"(%a, %b)',
            '    %p = "tessera.relu"(%a) : (tensor<17x35xf32>) -> tensor<17x35xf32>\n'
            '    %r = "tessera.matmul"(%p, %b)')
    else:
        source = source.replace("    return %r : tensor<17x19xf32>",
            '    %p = "tessera.relu"(%r) : (tensor<17x19xf32>) -> tensor<17x19xf32>\n'
            '    return %p : tensor<17x19xf32>')
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    result = subprocess.run([str(tool), "--tessera-graph-to-schedule", "-"],
                            input=source, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Graph->Schedule" in result.stderr
