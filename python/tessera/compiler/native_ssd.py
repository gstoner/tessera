"""Replay-bound serial and opt-in cooperative SSD GPU packages."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
from .scheduled_ssd import ScheduledSSD
from .native_gpu_storage import NativeGPUStoragePackage, _run, build_native_gpu_storage, replay_arena_ir
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def _shape(type_name):
    """SSD storage bound is independent of the small persistent-tape pilot."""
    match = re.fullmatch(r'tensor<((?:[1-9][0-9]*x)+)f32>',type_name)
    if match is None:
        raise ValueError('SSD requires ranked static f32 storage')
    shape = tuple(int(dim) for dim in match[1].split('x') if dim)
    count = 1
    for dim in shape:
        if dim > (1 << 24) or count > (1 << 24)//dim:
            raise ValueError('SSD storage exceeds its 64 MiB per-buffer envelope')
        count *= dim
    return shape


def _prepare(logical, compiler, llvm_bin, backend, cooperative=False, adjoint=False):
    if type(cooperative) is not bool or type(adjoint) is not bool:
        raise ValueError("SSD cooperative mode must be boolean")
    logical.validate(compiler)
    if adjoint and cooperative:
        raise ValueError('cooperative SSD adjoint is not implemented')
    if adjoint:
        from .ssd_checkpoint_ad import lower_checkpoint_vjp
        derivative = lower_checkpoint_vjp(logical,compiler=compiler)
        lowered = derivative.lowered_ir
        entry = 'ssd_vjp'
    else:
        lowered = logical.lowered_ir
        entry = 'ssd'
    signature = re.search(r'func.func @'+entry+r'\((.*?)\) -> \((.*?)\)',lowered,re.S)
    if signature is None:
        raise ValueError('SSD requires its isolated typed entry')
    inputs = re.findall(r'tensor<[^>]+>',signature[1])
    outputs = re.findall(r'tensor<[^>]+>',signature[2])
    input_count,output_count = (9,5) if adjoint else (5,3)
    if len(inputs) != input_count or len(outputs) != output_count:
        raise ValueError('SSD entry roles disagree')
    names = (('x','decay','b','c','initial','saved','dy','df','dcps','dx','ddecay','db','dc','dinitial') if adjoint else ('x','decay','b','c','initial','y','carry','checkpoints'))
    specs = tuple(TensorSpec(name,'fp32',_shape(type_),i>=input_count)
        for i,(name,type_) in enumerate(zip(names,inputs+outputs,strict=True))) + (IndexSpec('scratch',1,1),)
    if cooperative:
        gpu = _run(compiler,'--tessera-schedule-to-tile=ssd-gpu='+backend,source=logical.schedule_ir)
        states = int(_shape(inputs[2])[2])
        lanes = 1 << (states-1).bit_length()
        first,rest = gpu.split('\n',1)
        prefix = 'module attributes {'
        if not first.startswith(prefix) or not first.endswith('} {'):
            raise ValueError('SSD cooperative metadata missing')
        attrs = first[len(prefix):-3]
        gpu = attach_tensor_contract('module {\n'+rest,specs,
            grid=(int(_shape(inputs[0])[1])*int(_shape(inputs[0])[2]),1,1),block=(lanes,1,1))
        return gpu.replace(prefix,prefix+attrs+', ',1), specs
    # Function-boundary bufferization otherwise treats inputs as writable and
    # may recycle the initial carry into the loop state. The Schedule inputs
    # are immutable; project that ownership before choosing physical buffers.
    match = re.search(r'(func.func @'+entry+r'\()(.*?)(\) ->)',lowered,re.S)
    if match is None:
        raise ValueError('SSD lowered entry signature is missing')
    arguments,count = re.subn(r'(%[\w]+: tensor<[^>]+>)',
        r'\1 {bufferization.writable = false}',match[2])
    if count != input_count:
        raise ValueError('SSD lowered input projection disagrees')
    readonly = lowered[:match.start(2)] + arguments + lowered[match.end(2):]
    buffered = _run(llvm_bin/'mlir-opt',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
        '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
        '--convert-linalg-to-loops','--canonicalize',source=readonly)
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
    cooperative: bool = False
    adjoint: bool = False

    def validate(self):
        self.package.validate()
        if self.package.compiler_digest != self.logical.compiler_digest:
            raise ValueError('SSD physical compiler differs from Schedule owner')
        if hashlib.sha256((self.llvm_bin/'mlir-opt').read_bytes()).hexdigest() != self.package.llvm_digest:
            raise ValueError('SSD physical toolchain identity changed')
        source,specs = _prepare(self.logical,self.compiler,self.llvm_bin,self.package.backend,self.cooperative,self.adjoint)
        arena = replay_arena_ir(self.compiler, source)
        if arena != self.package.arena_ir:
            raise ValueError('SSD device artifact disagrees with Schedule replay')
        return specs

    def bind(self):
        specs = self.validate()
        signature = inspect.Signature([inspect.Parameter(s.name,inspect.Parameter.POSITIONAL_ONLY) for s in specs])
        return generate_tensor_binding(self.package,signature)


def materialize_ssd(logical, *, compiler, llvm_bin, backend, chip, cooperative=False, adjoint=False):
    compiler,llvm_bin = Path(compiler),Path(llvm_bin)
    source,_ = _prepare(logical,compiler,llvm_bin,backend,cooperative,adjoint)
    package = build_native_gpu_storage(source,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    result = NativeSSD(logical,compiler,llvm_bin,package,cooperative,adjoint)
    result.validate()
    return result
