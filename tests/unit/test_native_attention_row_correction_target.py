"""Host-free native Graph/AD/Schedule/Tile residual reaches NVIDIA target math.

PTX assembly and device numerical proof live in the owning NVIDIA device lane;
the required portable compiler lane supplies tessera-opt without a CUDA SDK.
"""
import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_attention_row_correction_schedule import stages

pytestmark = pytest.mark.compiler_route


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("bias,seed", [(False, False), (False, True), (True, False), (True, True)])
def test_native_row_correction_reaches_target_math(storage, wide, bias, seed):
    _, _, tile = stages(storage, bias=bias, saved=seed, output="f32" if wide else storage)
    tool = find_tessera_opt()
    assert tool is not None, "production compiler required"
    target = run_tessera_opt(tool, tile, "--tessera-lower-to-nvidia-sm120")
    assert "tile.attention_row_correction_kernel" not in target
    assert "nvvm.ex2 " in target
    assert "llvm.store" in target
    assert "scf.for" in target
    assert "arith.mulf" in target
