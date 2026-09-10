from dataclasses import replace
from pathlib import Path
import os
import pytest
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd
from tessera.compiler.native_ssd import materialize_ssd
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize('backend,chip',[('nvidia','sm_120'),('rocm','gfx1151')])
def test_ssd_gpu_package_replays_schedule_and_abi(backend,chip):
    tool = find_tessera_opt()
    llvm = Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('native GPU toolchain required')
    if backend == 'rocm' and not (Path(os.environ.get('ROCM_PATH','/opt/rocm'))/'llvm/bin/ld.lld').exists():
        pytest.skip('ROCm toolkit linker required; validate on owning ROCm host')
    logical = lower_scheduled_ssd(5,2,3,2,2,compiler=tool)
    native = materialize_ssd(logical,compiler=tool,llvm_bin=llvm,backend=backend,chip=chip)
    specs = native.validate()
    assert [s.name for s in specs] == ['x','decay','b','c','initial','y','carry','checkpoints','scratch']
    assert [s.writable for s in specs[:-1]] == [False]*5+[True]*3
    assert specs[-2].shape == (3,2,3,2)
    assert 'tessera.ssd.source' in native.package.arena_ir
    corrupted = replace(native.package,arena_ir=native.package.arena_ir.replace('arith.mulf','arith.addf'))
    corrupted = replace(corrupted,binding_digest=corrupted._digest())
    with pytest.raises(ValueError,match='replay'):
        replace(native,package=corrupted).validate()
