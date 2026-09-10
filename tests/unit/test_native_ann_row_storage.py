from pathlib import Path
import re
import subprocess
import pytest
from benchmarks.record_native_ann_execution import source
from tessera.compiler.native_ann_gpu import _prepare_ann_gpu_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


@pytest.mark.parametrize('backend', ['nvidia', 'rocm'])
def test_parallel_ann_compacts_proved_row_temporaries(backend):
    tool = find_tessera_opt()
    llvm = Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm / 'mlir-opt').exists():
        pytest.skip('native compiler toolchain required')
    program = run_tessera_opt(tool, source(64, 8, 'square'), '--canonicalize')
    text = _prepare_ann_gpu_ir(program, tool, llvm, backend, True, True)
    size = int(re.search(r'tessera.autodiff.temporary_bytes = (\d+)', text)[1])
    assert size <= 4096
    assert 'memref<1x8xf32>' in text
    assert 'gpu.thread_id' in text
    # The unpartitioned program remains outside this bounded storage tier.
    with pytest.raises(subprocess.CalledProcessError):
        _prepare_ann_gpu_ir(program, tool, llvm, backend, True, False)
