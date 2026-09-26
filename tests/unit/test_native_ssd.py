from dataclasses import replace
from pathlib import Path
import os
import pytest
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd
from tessera.compiler.native_ssd import materialize_ssd
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.llvm_tools import llvm_bin_dir
from tests._support.environment import require_native_storage_lane


@pytest.mark.parametrize('cooperative',[False,True])
@pytest.mark.parametrize('backend,chip',[('nvidia','sm_120'),('rocm','gfx1151'),('rocm','gfx1201')])
def test_ssd_gpu_package_replays_schedule_and_abi(backend,chip,cooperative):
    if chip == 'gfx1201' and os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1':
        # Sync GFX1201-SSD-CALIBRATION-2026-09-26: the gfx1201 row belongs to
        # the Tajasarus lane, never inferred from a gfx1151 host.
        pytest.skip('gfx1201 SSD packaging runs under TESSERA_GFX1201_DEVICE_PROOF=1')
    tool = find_tessera_opt()
    llvm = llvm_bin_dir()
    if tool is None or llvm is None or not (llvm/'mlir-opt').exists():
        pytest.skip('native GPU toolchain required')
    # Both lanes, not just ROCm. This guarded `rocm` and left `nvidia` bare, so
    # on either ROCm box the nvidia rows failed inside `mlir-opt`'s NVVM
    # serialization — a missing CUDA toolchain reported as a broken compiler.
    require_native_storage_lane(backend)
    logical = lower_scheduled_ssd(5,2,3,2,2,compiler=tool)
    native = materialize_ssd(logical,compiler=tool,llvm_bin=llvm,backend=backend,chip=chip,cooperative=cooperative)
    specs = native.validate()
    assert [s.name for s in specs] == ['x','decay','b','c','initial','y','carry','checkpoints','scratch']
    assert [s.writable for s in specs[:-1]] == [False]*5+[True]*3
    assert specs[-2].shape == (3,2,3,2)
    assert 'tessera.ssd.source' in native.package.arena_ir
    if cooperative:
        assert native.package.arena_ir.count('gpu.barrier') == 2
        assert 'gpu.block_id' in native.package.arena_ir
        assert 'addr_space = 3' in native.package.arena_ir
    corrupted = replace(native.package,arena_ir=native.package.arena_ir.replace('arith.mulf','arith.addf'))
    corrupted = replace(corrupted,binding_digest=corrupted._digest())
    with pytest.raises(ValueError,match='replay'):
        replace(native,package=corrupted).validate()


def test_cooperative_ssd_refuses_excess_state_lanes():
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    logical = lower_scheduled_ssd(1,1,257,1,1,compiler=tool)
    with pytest.raises(RuntimeError,match='256 states'):
        run_tessera_opt(tool,logical.schedule_ir,'--tessera-schedule-to-tile=ssd-gpu=nvidia')


def test_cooperative_ssd_refuses_unowned_module_definitions():
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    logical = lower_scheduled_ssd(2,1,3,1,1,compiler=tool)
    source = logical.schedule_ir.replace('module {',
        'module {\n memref.global "private" @unowned : memref<1xf32> = uninitialized',1)
    with pytest.raises(RuntimeError,match='isolated'):
        run_tessera_opt(tool,source,'--tessera-schedule-to-tile=ssd-gpu=nvidia')


def test_ssd_native_gpu_adjoint_projects_all_roles():
    tool = find_tessera_opt()
    llvm = llvm_bin_dir()
    if tool is None or llvm is None or not (llvm/'mlir-opt').exists():
        pytest.skip('native GPU toolchain required')
    require_native_storage_lane('nvidia')  # this test packages for sm_120 only
    logical = lower_scheduled_ssd(3,2,2,2,2,compiler=tool)
    native = materialize_ssd(logical,compiler=tool,llvm_bin=llvm,backend='nvidia',chip='sm_120',adjoint=True)
    specs = native.validate()
    assert len(specs) == 15
    assert [s.writable for s in specs[:-1]] == [False]*9+[True]*5
    assert specs[5].shape == (2,2,2,2)
    with pytest.raises(ValueError,match='not implemented'):
        materialize_ssd(logical,compiler=tool,llvm_bin=llvm,backend='nvidia',chip='sm_120',adjoint=True,cooperative=True)


def test_ssd_storage_envelope_is_not_the_tape_pilot_limit():
    from tessera.compiler.native_ssd import _shape
    assert _shape('tensor<512x2x32xf32>') == (512,2,32)
    with pytest.raises(ValueError,match='64 MiB'):
        _shape('tensor<16777217xf32>')
    with pytest.raises(ValueError,match='ranked static'):
        _shape('tensor<512xf64>')
