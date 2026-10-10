"""Physical storage order must be explicit at the typed B fragment boundary."""
import re
import pytest
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from benchmarks.nvidia.benchmark_row_major_b_core import tile_probe

pytestmark = pytest.mark.skipif(
    find_tessera_opt() is None or native._tool("tessera-nvidia-opt") is None,
    reason="requires native Schedule and NVIDIA lowering tools")


def lower(text):
    return native._run([native._tool("tessera-nvidia-opt"), "--tessera-lower-to-nvidia-sm120"],
                       text.encode()).decode()


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("bounded", [False, True])
def test_row_major_b_pack_materializes_scalar_pitch_gathers(dtype, bounded):
    shape=(17,35,19) if bounded else (16,32,8)
    _, text=tile_probe(shape,dtype,shape[2]+5,bounded=bounded)
    lowered=lower(text)
    assert "nvvm.mma.sync" in lowered
    assert "tile.fragment_pack" not in lowered
    storage="f16" if dtype=="fp16" else "bf16"
    assert len(re.findall(r"llvm.load[^\n]*-> "+storage+r"(?:\s|$)",lowered)) >= 4


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_row_major_b_source_requires_transpose_intent(dtype):
    _,text=tile_probe((16,32,8),dtype,13,bounded=False,transpose=False)
    with pytest.raises(RuntimeError,match="unsupported sm_120 fragment source layout"):
        lower(text)


def test_column_major_b_cannot_silently_ignore_transpose_intent():
    _,text=tile_probe((16,32,8),"fp16",13,bounded=False)
    text=text.replace('order = "row_major", leading_dim = 13>',
                      'order = "col_major", leading_dim = 32>',1)
    with pytest.raises(RuntimeError,match="unsupported sm_120 fragment source layout"):
        lower(text)
