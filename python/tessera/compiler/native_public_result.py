"""Compiler-projected capacity storage with device-returned logical shapes.

Exported native AD products generate checked input/output shape sidecars. This
is not automatic capture of arbitrary dynamic tensor-returning Python functions.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import inspect
import json
import math
from pathlib import Path
from .native_gpu_storage import _run, build_native_gpu_storage, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding
from .native_persistent_tape import _attribute
from .native_device_tape import _Buffer


def _prepare(source, compiler, backend, capacity=0, input_capacity=0):
    gpu = _run(compiler, '--tessera-native-tape-to-gpu=backend='+backend+' status-buffer=true public-result-capacity='+str(capacity)+' public-input-capacity='+str(input_capacity), source=source)
    metadata = json.loads(_attribute(gpu, 'tessera.native_result_abi'))
    if metadata.get('schema') != 1:
        raise ValueError('unsupported native result shape ABI')
    names = {'f32':'fp32', 'f64':'fp64', 'i8':'int8', 'i64':'int64'}
    specs: tuple[TensorSpec | IndexSpec, ...] = tuple(TensorSpec('arg'+str(i), names[row['storage']], tuple(row['shape']), row['writable'])
                  for i,row in enumerate(metadata['arguments']))
    specs += (TensorSpec('status','int64',(1,),True), IndexSpec('scratch',1,1))
    first, rest = gpu.split('\n',1)
    prefix = 'module attributes {'
    if not first.startswith(prefix) or not first.endswith('} {'):
        raise ValueError('native result module has no projected ABI')
    attrs = first[len(prefix):-3]
    gpu = attach_tensor_contract('module {\n'+rest,specs,grid=(1,1,1),block=(1,1,1))
    return gpu.replace(prefix,prefix+attrs+', ',1), metadata, specs


def materialize_public_results(source, *, compiler, llvm_bin, backend, chip):
    compiler, llvm_bin = Path(compiler), Path(llvm_bin)
    gpu, _, _ = _prepare(source,compiler,backend)
    package = build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    return NativePublicResult(source,compiler,package)


@dataclass(frozen=True)
class NativePublicResult:
    source: str
    compiler: Path
    package: NativeGPUStoragePackage
    capacity: int = 0
    input_capacity: int = 0

    def validate(self):
        self.package.validate()
        if hashlib.sha256(self.compiler.read_bytes()).hexdigest()!=self.package.compiler_digest:
            raise ValueError('public result compiler identity changed')
        gpu, metadata, specs = _prepare(self.source,self.compiler,self.package.backend,self.capacity,self.input_capacity)
        arena = _run(self.compiler,'--allow-unregistered-dialect','--tessera-tile-buffer-reuse',
                     '--tessera-tile-buffer-arena','--canonicalize',source=gpu)
        if arena != self.package.arena_ir:
            raise ValueError('public result artifact disagrees with native source replay')
        return metadata, specs

    def run(self, *inputs):
        return PublicResultFrame(self,inputs)

    def submit(self, stream, *inputs):
        if type(stream) is not int or not 0<stream<(1<<64):
            raise ValueError('public result submission requires a non-null stream')
        return PublicResultFrame(self,inputs,stream=stream)


class _LogicalView:
    def __init__(self, buffer, shape):
        self._buffer, self._shape = buffer, shape

    @property
    def __cuda_array_interface__(self):
        # Buffer checks its owning frame and pointer on every access.
        result = dict(self._buffer.__cuda_array_interface__)
        result.update(shape=self._shape,data=(self._buffer.pointer.value,True))
        return result


class PublicResultFrame:
    """Owns capacities and immutable host-validated logical result views."""
    def __init__(self, program, inputs, *, stream=None):
        metadata, specs = program.validate()
        if len(inputs)!=sum(not row['writable'] for row in metadata['arguments']):
            raise ValueError('public result input arity disagrees')
        self.program, self.closed, self.buffers = program, False, []
        self._submission=None
        self._metadata=metadata
        signature = inspect.Signature([inspect.Parameter(s.name,inspect.Parameter.POSITIONAL_ONLY) for s in specs])
        self.binding = generate_tensor_binding(program.package,signature)
        self.binding._bound = program.package.bind()
        native = self.binding._bound
        self.check, self.sync = native._check, native._sync
        cuda = program.package.backend == 'nvidia'
        def bind(cu, hip, types):
            fn = getattr(native._driver,cu if cuda else hip)
            fn.argtypes, fn.restype = types, ct.c_int
            return fn
        self.alloc = bind('cuMemAlloc_v2','hipMalloc',[ct.POINTER(ct.c_void_p),ct.c_size_t])
        self.free = bind('cuMemFree_v2','hipFree',[ct.c_void_p])
        if stream is not None:
            self.alloc_async=bind('cuMemAllocAsync','hipMallocAsync',[ct.POINTER(ct.c_void_p),ct.c_size_t,ct.c_void_p])
        self.copy_out = bind('cuMemcpyDtoH_v2','hipMemcpyDtoH',[ct.c_void_p,ct.c_void_p,ct.c_size_t])
        self.context_type = ct.c_void_p if cuda else ct.c_int
        self.current = bind('cuCtxGetCurrent','hipGetDevice',[ct.POINTER(self.context_type)])
        self.context = self.context_type(); self.check(self.current(ct.byref(self.context)))
        try:
            arguments = []
            values = iter(inputs)
            for spec in specs[:-2]:
                if not isinstance(spec, TensorSpec):
                    raise ValueError("native result argument is not a tensor")
                arguments.append(_Buffer(self,spec.shape,spec.dtype,stream=stream) if spec.writable else next(values))
            status = _Buffer(self,(1,),'int64',stream=stream)
            self._arguments,self._status=arguments,status
            if stream is None:
                self.binding(*arguments,status,1)
                self._expose()
            else:
                self._submission=self.binding.submit(stream,*arguments,status,1)
        except BaseException:
            self.close()
            raise

    def _expose(self):
        if self._integer(self._status)!=0:
            raise RuntimeError('public result guard failed; shapes and data are unavailable')
        results=[]
        for row in self._metadata['results']:
            shape=self._integers(self._arguments[row['shape']],row.get('rank',1))
            if any(not 0<=dim<=row['capacity'] for dim in shape) or math.prod(shape)>row['capacity']:
                raise RuntimeError('public result logical extent exceeds capacity')
            results.append(_LogicalView(self._arguments[row['data']],shape))
        self.results=tuple(results)

    def poll(self):
        """Expose logical views only after completion and successful status.

        Completion is queried; small status/shape readbacks happen only after
        it succeeds. Closing unrestricted exported views remains synchronous.
        """
        if self.closed:
            raise ValueError('public result frame is closed')
        current=self.context_type(); self.check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('public result requires its owning device context')
        if hasattr(self,'results'):
            return True
        if self._submission is not None and not self._submission.ticket.poll():
            return False
        self._expose()
        return True

    def _integers(self, buffer, count):
        value = (ct.c_int64 * count)()
        self.check(self.copy_out(ct.byref(value),buffer.pointer,ct.sizeof(value)))
        return tuple(value)

    def _integer(self, buffer):
        return self._integers(buffer,1)[0]

    def close(self):
        if self.closed:
            return
        context = self.context_type(); self.check(self.current(ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError('public result requires its owning device context')
        self.check(self.sync())
        while self.buffers:
            buffer = self.buffers[-1]
            self.check(self.free(buffer.pointer)); buffer.pointer=ct.c_void_p(); self.buffers.pop()
        self.binding.close()
        self.closed=True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


def materialize_ad_public_results(source, *, compiler, llvm_bin, backend, chip, capacity, role='forward', input_capacity=0):
    """Generate a checked public result from an exported native AD product.

    Multiple rank-one through rank-four results are admitted. With input_capacity,
    dynamic inputs bind flat capacity buffers plus native-validated shape sidecars.
    The native producer validates the logical extent against the caller's
    storage budget; Python never synthesizes the result-copy loop.
    """
    if type(capacity) is not int or not 1 <= capacity <= 1024 or role not in ('forward','backward'):
        raise ValueError('AD public results require a bounded capacity and product role')
    if type(input_capacity) is not int or not 0 <= input_capacity <= 1024:
        raise ValueError('AD input capacity must be a bounded integer')
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    exported=_run(compiler,'--tessera-autodiff-paired=box-product-scalars=true export-product='+role,source=source)
    native=_run(compiler,'--tessera-to-linalg',source=exported)
    buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
        '--convert-linalg-to-loops','--canonicalize',source=native)
    gpu,_,_=_prepare(buffered,compiler,backend,capacity,input_capacity)
    package=build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    return NativePublicResult(buffered,compiler,package,capacity,input_capacity)
