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


@pytest.mark.parametrize('backend', ['nvidia', 'rocm'])
def test_expanded_copies_keep_global_thread_row_and_compact_private_row(backend):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    source = 'module attributes {tessera.ann.source = "copy-regression"} {\n  memref.global "private" constant @weights : memref<2x2xf32> = dense<[[1.0, 2.0], [3.0, 4.0]]>\n  func.func @copy_rows(%input: memref<2x2xf32>, %output: memref<2x2xf32>) {\n    %weights = memref.get_global @weights : memref<2x2xf32>\n    %private = memref.alloc() : memref<2x2xf32>\n    memref.copy %weights, %private : memref<2x2xf32> to memref<2x2xf32>\n    memref.copy %private, %output : memref<2x2xf32> to memref<2x2xf32>\n    return\n  }\n}\n'
    lowered = run_tessera_opt(tool, source,
        '--tessera-native-tape-to-gpu=parallel-ann-rows=true backend='+backend)
    row = re.search(r'(%[\w]+) = gpu.thread_id x', lowered)[1]
    end = re.search(r'(%[\w]+) = arith.addi '+re.escape(row)+r', %[\w]+ : index', lowered)[1]
    # Both generated outer copy loops execute exactly this thread's row.
    loops = re.findall(r'scf.for (%[\w]+) = '+re.escape(row)+r' to '+re.escape(end)+r' step', lowered)
    assert len(loops) == 2
    full_row = re.escape(loops[0])
    assert re.search(r'memref.load %[^\[]+\['+full_row+r', %\w+\] : memref<2x2xf32', lowered)
    assert re.search(r'memref.store %\w+, %[^\[]+\['+re.escape(loops[1])+r', %\w+\] : memref<2x2xf32', lowered)
    # Only the private buffer is compacted. Its stores and loads use zero.
    zeroes = re.findall(r'(%\w+) = arith.constant 0 : index', lowered)
    accesses = re.findall(r'memref\.(?:load|store) [^\n]*\[(%\w+), %\w+\] : memref<1x2xf32', lowered)
    assert len(accesses) == 2 and all(index in zeroes for index in accesses)
    assert 'tessera.autodiff.temporary_bytes = 24' in lowered
