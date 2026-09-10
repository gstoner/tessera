"""Square shapes must not hide an incorrect transpose optimization."""
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('passes',[('--tessera-canonicalize',),('--canonicalize',),()])
def test_identity_permutation_matmul_executes_without_transposition(passes):
    compiler=Path(os.environ.get('TESSERA_OPT','/missing'))
    if not compiler.is_file() or not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():
        pytest.skip('native compiler and CPU JIT required')
    from tessera.compiler.native_gpu_storage import _run
    from tessera import _jit_boundary as jit
    source='''module {
      func.func @identity_mm(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %a = tessera.transpose %x {permutation = array<i64: 0, 1>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %b = tessera.matmul %a, %y : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %b : tensor<2x2xf32>
      }
    }'''
    lowered=_run(compiler,*passes,'--tessera-to-linalg',source=source)
    x=np.array([[1,2],[3,5]],np.float32)
    y=np.array([[2,7],[4,1]],np.float32)
    out=np.empty_like(x)
    handle=jit.compile_module(lowered)
    try:jit.invoke(handle,'identity_mm',[x,y],[out])
    finally:jit.destroy(handle)
    np.testing.assert_array_equal(out,x@y)
