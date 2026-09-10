"""Replay-bound serial SSD GPU baseline; cooperative tuning remains separate."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
from .scheduled_ssd import ScheduledSSD
from .native_gpu_storage import NativeGPUStoragePackage, _run, build_native_gpu_storage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_persistent_tape import _shape
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def _prepare(logical, compiler, llvm_bin, backend):
    logical.validate(compiler)
    signature = re.search(r'func.func @ssd\((.*?)\) -> \((.*?)\)',logical.schedule_ir,re.S)
    if signature is None:
        raise ValueError('SSD requires its isolated typed entry')
    inputs = re.findall(r'tensor<[^>]+>',signature[1])
    outputs = re.findall(r'tensor<[^>]+>',signature[2])
    if len(inputs) != 5 or len(outputs) != 3:
        raise ValueError('SSD entry roles disagree')
    names = ('x','decay','b','c','initial','y','carry','checkpoints')
    specs = tuple(TensorSpec(name,'fp32',_shape(type_),i>=5)
        for i,(name,type_) in enumerate(zip(names,inputs+outputs,strict=True))) + (IndexSpec('scratch',1,1),)
    # Function-boundary bufferization otherwise treats inputs as writable and
    # may recycle the initial carry into the loop state. The Schedule inputs
    # are immutable; project that ownership before choosing physical buffers.
    match = re.search(r'(func.func @ssd\()(.*?)(\) ->)',logical.lowered_ir,re.S)
    if match is None:
        raise ValueError('SSD lowered entry signature is missing')
    arguments,count = re.subn(r'(%[\w]+: tensor<[^>]+>)',
        r'\1 {bufferization.writable = false}',match[2])
    if count != 5:
        raise ValueError('SSD lowered input projection disagrees')
    readonly = logical.lowered_ir[:match.start(2)] + arguments + logical.lowered_ir[match.end(2):]
    buffered = _run(llvm_bin/'mlir-opt',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
        '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
        '--canonicalize',source=readonly)
    encoded = '"'+''.join('\\'+format(b,'02X') for b in logical.schedule_ir.encode())+'"'
    buffered = buffered.replace('module {','module attributes {tessera.ssd.source = '+encoded+'} {',1)
    gpu = _run(compiler,'--tessera-native-tape-to-gpu=backend='+backend,source=buffered)
    first,rest = gpu.split('\n',1)
    prefix = 'module attributes {'
    if not first.startswith(prefix) or not first.endswith('} {'):
        raise ValueError('SSD lowered owner metadata is missing')
    attrs = first[len(prefix):-3]
    gpu = attach_tensor_contract('module {\n'+rest,specs,grid=(1,1,1),block=(1,1,1))
    return gpu.replace(prefix,prefix+attrs+', ',1), specs


@dataclass(frozen=True)
class NativeSSD:
    logical: ScheduledSSD
    compiler: Path
    llvm_bin: Path
    package: NativeGPUStoragePackage

    def validate(self):
        self.package.validate()
        if self.package.compiler_digest != self.logical.compiler_digest:
            raise ValueError('SSD physical compiler differs from Schedule owner')
        if hashlib.sha256((self.llvm_bin/'mlir-opt').read_bytes()).hexdigest() != self.package.llvm_digest:
            raise ValueError('SSD physical toolchain identity changed')
        source,specs = _prepare(self.logical,self.compiler,self.llvm_bin,self.package.backend)
        arena = _run(self.compiler,'--allow-unregistered-dialect','--tessera-tile-buffer-reuse',
                     '--tessera-tile-buffer-arena','--canonicalize',source=source)
        if arena != self.package.arena_ir:
            raise ValueError('SSD device artifact disagrees with Schedule replay')
        return specs

    def bind(self):
        specs = self.validate()
        signature = inspect.Signature([inspect.Parameter(s.name,inspect.Parameter.POSITIONAL_ONLY) for s in specs])
        return generate_tensor_binding(self.package,signature)


def materialize_ssd(logical, *, compiler, llvm_bin, backend, chip):
    compiler,llvm_bin = Path(compiler),Path(llvm_bin)
    source,_ = _prepare(logical,compiler,llvm_bin,backend)
    package = build_native_gpu_storage(source,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    result = NativeSSD(logical,compiler,llvm_bin,package)
    result.validate()
    return result
