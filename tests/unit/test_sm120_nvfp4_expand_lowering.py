"""sm_120 FP4 expansion route and contract tests (SM120-FP4-EXPAND-NATIVE-20261103).

A compiler-route test: successful IR emission is not numerical evidence. The
bitwise device proof is tests/device/nvidia/test_native_nvfp4_expand_sm120.py
on the RTX 5070; gfx1201/gfx1151 proofs of the ROCm image do not transfer.
"""
import json
import re
import subprocess

import pytest

from tessera.compiler.nvfp4_expansion import mxfp4_expansion_policy, nvfp4_expansion_policy
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def _graph(n=7, k=32, offsets=(0, 3, 7), policy=None):
    from tessera.compiler.nvidia_nvfp4_expand_native import build_sm120_nvfp4_expansion_graph

    return build_sm120_nvfp4_expansion_graph(n, k, list(offsets),
                                             numeric_policy=policy or nvfp4_expansion_policy())


def test_builder_names_the_sm120_target_pair():
    graph = _graph()
    assert json.loads(graph.module_attrs["tessera.target"]) == "nvidia_sm120"
    assert json.loads(graph.module_attrs["tessera.arch"]) == "sm_120"


def test_public_capability_stays_unsupported_on_sm120():
    # Only the explicit package route admits the Graph; a ready/artifact row
    # would let @jit(target="nvidia_sm120") fall back to the eager oracle.
    from tessera.compiler.capabilities import supports_op

    assert not supports_op("nvidia_sm120", "tessera.nvfp4_expand", dtype="uint8", rank=2).supported


def test_execution_matrix_row_names_its_device_fixture():
    from tessera.compiler.execution_matrix import _MATRIX
    from tessera.compiler.nvidia_nvfp4_expand_native import COMPILER_PATH

    row = _MATRIX[("nvidia_sm120", COMPILER_PATH)]
    assert row.executor_id == "nvidia_native_descriptor"
    assert row.evidence_target == "nvidia_sm120" and row.device_proof == "device_verified_abi"
    assert row.numerical_fixture == "tests/device/nvidia/test_native_nvfp4_expand_sm120.py"


@pytest.fixture(scope="module")
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching compiler required")
    return tool


def _tile(compiler, graph, target="nvidia_sm120"):
    source = graph.to_mlir(verify=False, target=target, canonical=True)
    return run_tessera_opt(compiler, run_tessera_opt(
        compiler, source, "--tessera-graph-to-schedule"), "--tessera-schedule-to-tile")


@pytest.mark.parametrize("policy", [nvfp4_expansion_policy(), mxfp4_expansion_policy()])
def test_schedule_to_tile_emits_the_nvvm_kernel_entry(compiler, policy):
    from tessera.compiler.nvidia_nvfp4_expand_native import _contract

    tile = _tile(compiler, _graph(k=64, policy=policy))
    assert re.search(r"llvm\.func @tessera_nvfp4_expand_[0-9a-f]{16}\(", tile)
    assert "nvvm.kernel" in tile and "func.func" not in tile
    names, n, k, offsets, _, _, mx = _contract(tile)
    assert (names, n, k, offsets) == (("codes", "scales", "globals", "values", "scale_plane"),
                                      7, 64, (0, 3, 7))
    assert mx is (policy == mxfp4_expansion_policy())
    assert ('scale_storage = "e8m0_bits"' if mx else 'scale_storage = "e4m3_bits"') in tile


def test_graph_contract_refuses_a_mismatched_target_pair(compiler):
    graph = _graph()
    graph.module_attrs["tessera.arch"] = json.dumps("gfx1201")
    source = graph.to_mlir(verify=False, target="nvidia_sm120", canonical=True)
    result = subprocess.run([str(compiler), "--tessera-graph-to-schedule"], input=source,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "NVFP4 ingest needs an isolated gfx1201 checkpoint entry" in result.stderr


def test_package_refuses_a_rocm_graph_before_any_tool():
    from tessera.compiler.nvidia_nvfp4_expand_native import package_sm120_nvfp4_expansion
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph

    graph = build_nvfp4_expansion_graph(7, 32, [0, 7], numeric_policy=nvfp4_expansion_policy(),
                                        target="rocm_gfx1201")
    with pytest.raises(ValueError, match="nvidia_sm120 / sm_120"):
        package_sm120_nvfp4_expansion(graph)


@pytest.fixture(scope="module")
def nvidia_opt():
    from tessera.compiler.nvidia_native import _tool

    tool = _tool("tessera-nvidia-opt")
    if tool is None:
        pytest.skip("matching NVIDIA target compiler required")
    return str(tool)


def _lower(nvidia_opt, tile, pipeline="--tessera-lower-to-nvidia-sm120"):
    return subprocess.run([nvidia_opt, pipeline], input=tile, capture_output=True, text=True)


def test_nvidia_lowering_materializes_the_kernel(compiler, nvidia_opt):
    result = _lower(nvidia_opt, _tile(compiler, _graph(policy=mxfp4_expansion_policy())))
    assert result.returncode == 0, result.stderr
    assert "tile.nvfp4_expand_kernel" not in result.stdout
    assert "nvvm.read.ptx.sreg.ctaid.x" in result.stdout
    assert "arith.truncf" in result.stdout


def test_nvidia_lowering_refuses_a_rocm_tile(compiler, nvidia_opt):
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph

    graph = build_nvfp4_expansion_graph(7, 32, [0, 7], numeric_policy=nvfp4_expansion_policy(),
                                        target="rocm_gfx1201")
    result = _lower(nvidia_opt, _tile(compiler, graph, target="rocm_gfx1201"))
    assert result.returncode != 0
    assert "NVIDIA_NVFP4_EXPAND_ARCH_MISMATCH" in result.stderr


def test_nvidia_lowering_refuses_another_sm(compiler, nvidia_opt):
    result = _lower(nvidia_opt, _tile(compiler, _graph()), "--tessera-lower-to-nvidia-sm90")
    assert result.returncode != 0
    assert "NVIDIA_NVFP4_EXPAND_ARCH_MISMATCH" in result.stderr
