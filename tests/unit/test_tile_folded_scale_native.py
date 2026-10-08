"""Verifier proof for the internal full-K folded Tile epilogue."""
import subprocess
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt

SOURCE = """
!fc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
func.func @scale(%acc: !fc, %lhs: memref<?xf32>, %rhs: memref<?xi8>, %m: index, %n: index) -> !fc {
 %z = arith.constant 0 : index
 %r = tile.fragment_folded_scale %acc scales(%lhs, %rhs) at(%z, %z) bounds(%m, %n) : !fc, memref<?xf32>, memref<?xi8>
 return %r : !fc
}
"""

@pytest.mark.parametrize("old,new,code", [
    ("memref<?xf32>", "memref<?xf64>", "TILE_FRAGMENT_FOLDED_SCALE_BUFFERS"),
    ("memref<?xi8>", "memref<?xi16>", "TILE_FRAGMENT_FOLDED_SCALE_BUFFERS"),
    ("memref<?xi8>", "memref<2x?xi8>", "TILE_FRAGMENT_FOLDED_SCALE_BUFFERS"),
    ('elem = "f32", acc = "f32"', 'elem = "f16", acc = "f16"', "TILE_FRAGMENT_FOLDED_SCALE_TYPE"),
])
def test_folded_scale_verifier_rejects_wrong_contract(old,new,code):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires current native compiler")
    result = subprocess.run([str(tool), "-"], input=SOURCE.replace(old,new),
                            capture_output=True,text=True)
    assert result.returncode != 0
    assert code in result.stderr, result.stderr

def test_folded_scale_verifier_accepts_typed_contract():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires current native compiler")
    result = subprocess.run([str(tool), "-"], input=SOURCE,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert "tile.fragment_folded_scale" in result.stdout
