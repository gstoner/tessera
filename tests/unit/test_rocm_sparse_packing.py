import itertools

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_sparse_packing import pack_sparse_wmma_inputs


@pytest.mark.parametrize("dtype",[np.float16,ml_dtypes.bfloat16])
def test_sparse_packing_roundtrip_all_pairs_and_register_groups(dtype):
    a=np.zeros((16,32),dtype=dtype)
    pairs=list(itertools.combinations(range(4),2))
    for row in range(16):
        for group in range(8):
            cols=pairs[(row+group)%6]
            a[row,group*4+np.array(cols)]=np.asarray([1,-2],dtype=dtype)
    b=np.arange(512,dtype=np.float32).reshape(32,16).astype(dtype)
    result=pack_sparse_wmma_inputs(a,b)
    av=np.frombuffer(result.a,np.uint16).reshape(32,8)
    bv=np.frombuffer(result.b,np.uint16).reshape(32,16)
    indices=np.frombuffer(result.indices,np.uint32)
    decoded=np.zeros((16,32),np.uint16)
    # Invert lane/register addressing (independent traversal from the producer).
    for lane in range(32):
        for reg in range(4):
            group=(reg//2)*4+(lane//16)*2+reg%2
            code=int(indices[lane])>>(reg*4)&15
            first,second=code&3,code>>2
            assert first<second
            decoded[lane%16,group*4+first]=av[lane,reg*2]
            decoded[lane%16,group*4+second]=av[lane,reg*2+1]
        for reg in range(8):
            for half in range(2):
                k=(reg//4)*16+(lane//16)*8+(reg%4)*2+half
                assert bv[lane,reg*2+half]==b.view(np.uint16)[k,lane%16]
    np.testing.assert_array_equal(decoded,a.view(np.uint16))
    assert isinstance(result.a,bytes) and np.all(indices>>16==0)
    a.fill(9)
    assert not np.array_equal(decoded,a.view(np.uint16))


def test_sparse_packing_refuses_pruning_and_unsupported_storage():
    with pytest.raises(ValueError,match="two stored"):
        pack_sparse_wmma_inputs(np.ones((16,32),np.float16),np.ones((32,16),np.float16))
    with pytest.raises(ValueError,match="storage"):
        pack_sparse_wmma_inputs(np.zeros((16,32),np.float32),np.ones((32,16),np.float32))
    with pytest.raises(ValueError,match="matching"):
        pack_sparse_wmma_inputs(np.zeros((16,16),np.float16),np.ones((32,16),np.float16))


@pytest.mark.compiler_rocm
@pytest.mark.parametrize("dtype",["float16","bfloat16"])
def test_sparse_target_ir_is_consumed_by_native_pass(dtype):
    import os
    import subprocess
    from tessera.compiler.rocm_sparse_packing import sparse_wmma_target_ir
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    compiler=find_tessera_opt()
    if compiler is None:
        pytest.skip("native Tessera compiler required")
    source=sparse_wmma_target_ir(dtype)
    result=subprocess.run([str(compiler),"--lower-tessera-target-to-rocdl"],
        input=source,text=True,capture_output=True,env=os.environ)
    assert result.returncode==0,result.stderr
    assert "tessera_rocm.swmmac" not in result.stdout
    assert "llvm.amdgcn.swmmac.f32.16x16x32." in result.stdout
    for broken in (source.replace('gfx1201','gfx1151'),
                   source.replace('vector<8xf32>', 'vector<4xf32>')):
        result=subprocess.run([str(compiler),"--lower-tessera-target-to-rocdl"],
            input=broken,text=True,capture_output=True)
        assert result.returncode!=0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sparse_schedule_tile_target_ancestry(dtype):
    import subprocess
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    from tessera.compiler.rocm_sparse_packing import sparse_wmma_schedule_ir
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires native compiler")
    schedule = sparse_wmma_schedule_ir(dtype)
    tile = subprocess.check_output([tool, "--tessera-schedule-to-tile"], input=schedule, text=True)
    assert "schedule.sparse_mma" not in tile and "tile.sparse_mma" in tile
    target = subprocess.check_output([tool, "--lower-tile-to-rocm"], input=tile, text=True)
    assert "tile.sparse_mma" not in target and "tessera_rocm.swmmac" in target
    for ir in (schedule, tile):
        bad = subprocess.run([tool], input=ir.replace('arch = "gfx1201"', 'arch = "gfx1151"'), text=True, capture_output=True)
        assert bad.returncode != 0
