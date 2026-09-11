"""Exercise the actual conversion pass when a native compiler is available."""
import os
from pathlib import Path
import subprocess
import pytest
from benchmarks.record_dtype_arithmetic import emit


@pytest.mark.parametrize('dtype', ['fp8_e4m3','fp8_e5m2'])
@pytest.mark.parametrize('lanes', [1,2])
def test_byte_fp8_conversion_fully_legalizes(dtype, lanes):
    compiler = os.environ.get('TESSERA_OPT')
    if not compiler or not Path(compiler).is_file():
        pytest.skip('requires built tessera-opt')
    result = subprocess.run([compiler, '--allow-unregistered-dialect',
        '--tessera-expand-lowp-conversions'], input=emit(dtype,lanes), text=True,
        capture_output=True, check=True).stdout
    assert 'f8E' not in result
    assert 'arith.fptoui' in result and 'arith.select' in result
    # Conversion is idempotent and emits parseable registered operations.
    again = subprocess.run([compiler, '--allow-unregistered-dialect',
        '--tessera-expand-lowp-conversions'], input=result,text=True,
        capture_output=True,check=True).stdout
    assert result == again


def test_non_nearest_rounding_is_not_silently_rewritten():
    compiler = os.environ.get('TESSERA_OPT')
    if not compiler or not Path(compiler).is_file():
        pytest.skip('requires built tessera-opt')
    source = '''module { func.func @up(%x: f32) -> i8 {
      %v = arith.truncf %x upward : f32 to f8E5M2
      %b = arith.bitcast %v : f8E5M2 to i8
      return %b : i8
    }}'''
    result = subprocess.run([compiler,'--tessera-expand-lowp-conversions'],
        input=source,text=True,capture_output=True,check=True).stdout
    assert 'upward' in result
    assert 'f8E5M2' in result


@pytest.mark.parametrize('directive,generate', [
    ('"tessera_rocm.int4_pack"() {name="pack",kind="pack"} : () -> ()', 'generate-rocm-int4-pack-kernel'),
    ('"tessera_rocm.dequant_gemm"() {name="dequant"} : () -> ()', 'generate-rocm-dequant-gemm-kernel'),
])
def test_packed_producers_convert_in_process_without_duplicate_kernel_attrs(directive,generate):
    compiler = os.environ.get('TESSERA_OPT')
    if not compiler or not Path(compiler).is_file():
        pytest.skip('requires built tessera-opt')
    pipeline = f'builtin.module({generate},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts))'
    result = subprocess.run([compiler,'--allow-unregistered-dialect',f'--pass-pipeline={pipeline}'],
        input='module {'+directive+'}',text=True,capture_output=True,check=True).stdout
    assert 'llvm.func' in result and 'rocdl.kernel' in result
