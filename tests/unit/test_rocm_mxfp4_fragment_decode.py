"""Target-specific register export exposes all packed weight bits, including -0."""
import pytest

from tests.unit.test_rocm_bounded_fragment_pack import lower_fragment


CASES = ((16, 16, 0, 0), (15, 13, 0, 0), (9, 7, 0, 0), (1, 1, 0, 0),
         (32, 16, 16, 0), (31, 13, 16, 0), (32, 16, 1, 1), (64, 16, 32, 0))


def fragment_byte_source(pitch):
    """Diagnostic cast names the gfx1201 register ABI; this is not portable IR."""
    return '''
!fb = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "auto">
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>
#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
module {
 gpu.module @packed_bytes_mod {
  gpu.func @packed_fragment_bytes(%b: memref<?xi8>, %out: memref<?xi32>,
      %k: index, %n: index, %row: index, %col: index) kernel {
   %two = arith.constant 2 : index
   %tid = gpu.thread_id x
   %offset = arith.muli %tid, %two : index
   %view = tile.view %b, %row, %col, %k, %n {
    tile.layout = #layout,
    tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = PITCH_VALUE>,
    tile.storage_contract = "e2m1_packed_k_to_e4m3"
   } : (memref<?xi8>, index, index, index, index) -> !tile.tile
   %fragment = tile.fragment_pack %view {role = "b", mma = #mma} : (!tile.tile) -> !fb
   %registers = builtin.unrealized_conversion_cast %fragment : !fb to vector<2xi32>
   vector.store %registers, %out[%offset] : memref<?xi32>, vector<2xi32>
   gpu.return
  }
 }
}
'''.replace('PITCH_VALUE', str(pitch))


@pytest.mark.compiler_route
@pytest.mark.parametrize("pitch", [16, 17])
def test_packed_b_register_export_compiles(compiler_toolchain, pitch):
    lowered = lower_fragment(compiler_toolchain, fragment_byte_source(pitch),
                             "gfx1201", rocdl=True)
    assert "tile.fragment_pack" not in lowered
    assert "vector.store" in lowered and "vector<2xi32>" in lowered
