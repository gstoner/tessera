"""Physical HVP packaging from the compiler's captured product ABI."""
import json
from pathlib import Path

from .native_gpu_storage import _run, build_native_gpu_storage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_persistent_tape import _attribute, _shape, _dtype
from .native_storage_contract import attach_tensor_contract


def materialize_native_hvp(source, *, compiler, llvm_bin, backend, chip):
    """Compile a bounded serial CUDA/HIP HVP; no derivative is rebuilt here."""
    if 'tessera.frontend.authority = "tracer"' not in source:
        raise ValueError('native HVP requires tracer-owned source')
    compiler, llvm_bin = Path(compiler), Path(llvm_bin)
    exported = _run(compiler, '--tessera-autodiff-paired=normalize-counted-while=true normalize-data-while=true',
                    '--tessera-autodiff-hvp-prepare',
                    '--tessera-autodiff-forward=export-hvp=true', source=source)
    abi = json.loads(_attribute(exported, 'tessera.autodiff.product_abi'))
    if abi.get('schema') != 1 or abi.get('role') != 'hvp':
        raise ValueError('native HVP requires its compiler product ABI')
    types = abi['inputs'] + abi['results']
    specs = tuple(TensorSpec(f'arg{i}', _dtype(t), _shape(t), i >= len(abi['inputs']))
                  for i, t in enumerate(types)) + (IndexSpec('scratch', 1, 1),)
    native = _run(compiler, '--tessera-to-linalg', source=exported)
    buffered = _run(llvm_bin/'mlir-opt', '--allow-unregistered-dialect',
        '--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
        '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
        '--convert-linalg-to-loops', '--canonicalize', source=native)
    gpu = _run(compiler, '--allow-unregistered-dialect',
               '--tessera-native-tape-to-gpu=backend='+backend, source=buffered)
    first, rest = gpu.split('\n', 1)
    prefix = 'module attributes {'
    if not first.startswith(prefix) or not first.endswith('} {'):
        raise ValueError('native HVP lacks compiler product lineage')
    gpu = attach_tensor_contract('module {\n'+rest, specs, grid=(1,1,1), block=(1,1,1))
    gpu = gpu.replace(prefix, prefix+first[len(prefix):-3]+', ', 1)
    return build_native_gpu_storage(gpu, compiler=compiler, llvm_bin=llvm_bin,
                                    backend=backend, chip=chip)
