from pathlib import Path
import pytest
from tessera.compiler.resident_gradient_sum import gradient_sum_source, bind_gradient_sum
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.environment import require_native_storage_lane


@pytest.mark.parametrize('shape', [(), (0,), (-1,), (True,), (16777217,)])
def test_gradient_sum_refuses_invalid_or_excessive_storage(shape):
    with pytest.raises(ValueError, match='bound'):
        gradient_sum_source(shape)


def test_gradient_sum_passes_native_replay_and_tensor_contract():
    tool = find_tessera_opt()
    if tool is None or not Path('/usr/lib/llvm-23/bin/mlir-opt').exists():
        pytest.skip('native compiler required')
    require_native_storage_lane('nvidia')  # packages for sm_120 only
    binding = bind_gradient_sum((3, 4), compiler=tool, llvm_bin='/usr/lib/llvm-23/bin',
                                backend='nvidia', chip='sm_120')
    assert binding.specs[2].shape == (3, 4)
    assert binding.specs[2].writable
    binding.package.validate()
