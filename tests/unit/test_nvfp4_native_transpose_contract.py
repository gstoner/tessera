"""Native orientation sealing; no GPU execution in this suite."""
import os
import re
from pathlib import Path
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera.compiler.scheduled_matmul import find_tessera_opt, lower_scheduled_matmul, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="matching native compiler required")


def oriented(a, b, sa, sb):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [1, 16], "format": "ue4m3"},
        transposeA=True, transposeB=True)


def artifact():
    traced = trace(oriented, ((31, 7), "nvfp4"), ((5, 31), "nvfp4"),
                   ((2, 7), "uint8"), ((5, 2), "uint8"))
    module = to_graph_ir_module(traced, name="oriented", target="nvidia_sm120")
    return lower_scheduled_matmul(module, target="nvidia_sm120")


@pytest.mark.parametrize("flag", ["transposeA", "transposeB"])
def test_schedule_orientation_cannot_change_after_hashing(flag):
    a = artifact()
    assert (a.m, a.n, a.k) == (7, 5, 31)
    line = next(line for line in a.schedule_ir.splitlines() if "= schedule.matmul" in line)
    assert f"{flag} = true" in line
    changed = a.schedule_ir.replace(line, line.replace(f"{flag} = true", f"{flag} = false"))
    with pytest.raises(RuntimeError, match="altered after hashing"):
        run_tessera_opt(find_tessera_opt(), changed, "--tessera-schedule-to-tile")


@pytest.mark.parametrize("flag", ["transposeA", "transposeB"])
def test_tile_orientation_is_typed(flag):
    tool = Path(os.environ.get("TESSERA_NVIDIA_OPT", "missing"))
    if not tool.is_file():
        pytest.skip("matching NVIDIA target compiler required")
    a = artifact()
    changed = re.sub(rf'{flag} = true', f'{flag} = "wrong"', a.tile_ir)
    assert changed != a.tile_ir
    with pytest.raises(RuntimeError, match="orientation flags must be boolean"):
        run_tessera_opt(tool, changed, "--tessera-lower-to-nvidia-sm120")


@pytest.mark.parametrize("original,wrong", [("tensor<2x7xui8>", "tensor<7x2xui8>"),
                                             ("tensor<5x2xui8>", "tensor<2x5xui8>")])
def test_native_graph_rejects_untransposed_scale_shapes(original, wrong):
    a = artifact()
    changed = a.graph_ir.replace(original, wrong)
    assert changed != a.graph_ir
    with pytest.raises(RuntimeError, match="UE4M3 scales"):
        run_tessera_opt(find_tessera_opt(), changed, "--canonicalize")


@pytest.mark.parametrize("level", ["schedule", "tile"])
@pytest.mark.parametrize("flag", ["transposeA", "transposeB"])
def test_orientation_types_verified_without_target_lowering(level, flag):
    a = artifact()
    ir = a.schedule_ir if level == "schedule" else a.tile_ir
    line = next(line for line in ir.splitlines() if f"{level}.matmul" in line)
    changed = ir.replace(line, line.replace(f"{flag} = true", f"{flag} = {chr(34)}wrong{chr(34)}"))
    assert changed != ir
    with pytest.raises(RuntimeError, match="orientation flags must be boolean"):
        run_tessera_opt(find_tessera_opt(), changed, "--canonicalize")


@pytest.mark.parametrize("transpose_b", [False, True])
def test_shared_rhs_transposed_a_keeps_native_batch_abi(transpose_b):
    from tests.device.nvidia.test_nvfp4_transpose_jit import oriented_product
    from tessera.compiler import nvidia_native
    b_shape = (19, 129) if transpose_b else (129, 19)
    sb_shape = (19, 9) if transpose_b else (9, 19)
    traced = trace(oriented_product(True, transpose_b, "shared_rhs_rows"),
                   ((3, 129, 17), "nvfp4"), (b_shape, "nvfp4"),
                   ((3, 9, 17), "uint8"), (sb_shape, "uint8"))
    module = to_graph_ir_module(traced, name="shared_oriented", target="nvidia_sm120")
    assert nvidia_native.supports_nvfp4_matmul(module)
    a = lower_scheduled_matmul(module, target="nvidia_sm120")
    assert (a.m, a.n, a.k) == (51, 19, 129)
    assert 'batching = "shared_rhs_rows"' in a.tile_ir
    assert "@tessera_tile_matmul_nvfp4_batched" in a.tile_ir


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False),
                                                    (False, True), (True, True)])
def test_shared_lhs_batch_keeps_native_roles_and_sealed_policy(transpose_a, transpose_b):
    from tests.device.nvidia.test_nvfp4_transpose_jit import oriented_product
    from tessera.compiler import nvidia_native
    a_shape = (129, 17) if transpose_a else (17, 129)
    b_shape = (3, 19, 129) if transpose_b else (3, 129, 19)
    sa_shape = (9, 17) if transpose_a else (17, 9)
    sb_shape = (3, 19, 9) if transpose_b else (3, 9, 19)
    traced = trace(oriented_product(transpose_a, transpose_b, "shared_lhs"),
                   (a_shape, "nvfp4"), (b_shape, "nvfp4"),
                   (sa_shape, "uint8"), (sb_shape, "uint8"))
    module = to_graph_ir_module(traced, name="shared_lhs", target="nvidia_sm120")
    assert nvidia_native.supports_nvfp4_matmul(module)
    a = lower_scheduled_matmul(module, target="nvidia_sm120")
    assert (a.m, a.n, a.k) == (51, 19, 129)
    assert 'batching = "shared_lhs"' in a.tile_ir
    assert "@tessera_tile_matmul_nvfp4_batched" in a.tile_ir
    changed = (a.schedule_ir.replace('batching = "shared_lhs"', 'batching = "independent_rhs"')
               .replace("tensor<17x129x", "tensor<3x17x129x")
               .replace("tensor<129x17x", "tensor<3x129x17x")
               .replace("tensor<17x9x", "tensor<3x17x9x")
               .replace("tensor<9x17x", "tensor<3x9x17x"))
    assert changed != a.schedule_ir
    with pytest.raises(RuntimeError, match="does not match the retained Graph matmul contract"):
        run_tessera_opt(find_tessera_opt(), changed, "--tessera-schedule-to-tile")
