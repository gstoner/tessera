from pathlib import Path
from dataclasses import replace
import pytest
from tessera.compiler.gpu_heap_collection import emit_pool, materialize_pool
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize(
    "slots,width,mode",
    [(0, 4, "allocate"), (True, 4, "collect"), (3, 0, "collect"), (257, 4, "collect"), (3, 4, "unknown")],
)
def test_pool_envelope(slots, width, mode):
    with pytest.raises(ValueError, match="bounded"):
        emit_pool(slots, width, mode)


@pytest.mark.parametrize("mode", ["allocate", "collect"])
def test_native_pool_replay(mode):
    tool = find_tessera_opt()
    if tool is None or not Path("/usr/lib/llvm-23/bin/mlir-opt").exists():
        pytest.skip("native compiler required")
    program = materialize_pool(
        3, 4, mode, compiler=tool, llvm_bin="/usr/lib/llvm-23/bin", backend="nvidia", chip="sm_120"
    )
    program.validate()
    with pytest.raises(ValueError, match="replay"):
        replace(program, width=5).validate()


@pytest.mark.parametrize('mode',['allocate','collect','graph','mark','collect_seeded'])
def test_byte_object_graph_native_replay(mode):
    tool = find_tessera_opt()
    if tool is None or not Path('/usr/lib/llvm-23/bin/mlir-opt').exists():
        pytest.skip('native compiler required')
    program = materialize_pool(4,64,mode,compiler=tool,llvm_bin='/usr/lib/llvm-23/bin',backend='nvidia',chip='sm_120',payload_dtype='int8',references=4)
    assert program.validate()[2].shape == (4,8)
    with pytest.raises(ValueError,match='replay'):
        replace(program,references=2).validate()
