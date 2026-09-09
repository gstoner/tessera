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
import threading
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

    def submit(self, stream, *inputs, scoped=False):
        if type(stream) is not int or not 0<stream<(1<<64):
            raise ValueError('public result submission requires a non-null stream')
        return PublicResultFrame(self,inputs,stream=stream,scoped=scoped)


class _LogicalView:
    def __init__(self, buffer, shape):
        self._buffer, self._shape = buffer, shape

    @property
    def pointer(self):
        return self._buffer.pointer

    @property
    def __cuda_array_interface__(self):
        # Buffer checks its owning frame and pointer on every access.
        result = dict(self._buffer.__cuda_array_interface__)
        result.update(shape=self._shape,data=(self._buffer.pointer.value,True))
        return result


class PublicResultFrame:
    """Owns capacities and immutable host-validated logical result views."""
    def __init__(self, program, inputs, *, stream=None, scoped=False, snapshot=False):
        if type(scoped) is not bool or (scoped and stream is None):
            raise ValueError('scoped public frames require asynchronous submission')
        self._scoped, self._stream = scoped, stream
        self._lock = threading.RLock()
        self._retiring = False
        self._owner = None
        self._submissions = []
        metadata, specs = program.validate()
        if len(inputs)!=sum(not row['writable'] for row in metadata['arguments']):
            raise ValueError('public result input arity disagrees')
        self.program, self.closed = program, False
        self.buffers: list[_Buffer] = []
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
            self.free_async=bind('cuMemFreeAsync','hipFreeAsync',[ct.c_void_p,ct.c_void_p])
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
            if snapshot:
                raw,_,_,_,_=self.binding._resident(*arguments,status,1)
                copy=bind('cuMemcpyDtoD_v2' if stream is None else 'cuMemcpyDtoDAsync_v2','hipMemcpyDtoD' if stream is None else 'hipMemcpyDtoDAsync',[ct.c_void_p,ct.c_void_p,ct.c_size_t]+([] if stream is None else [ct.c_void_p]))
                if stream is None:self.check(self.sync())
                self._snapshot_borrows=tuple(inputs)
                owned=[]
                for i,spec in enumerate(specs[:-2]):
                    if not spec.writable:
                        buffer=_Buffer(self,spec.shape,spec.dtype,stream=stream)
                        self.check(copy(buffer.pointer,ct.c_void_p(raw[i]),buffer.nbytes,*(() if stream is None else (ct.c_void_p(stream),))))
                        arguments[i]=buffer
                        owned.append(buffer)
                self._snapshot_inputs=tuple(owned)
            self._arguments,self._status=arguments,status
            if stream is None:
                self.binding(*arguments,status,1)
                self._expose()
            else:
                self._submission=self.binding.submit(stream,*arguments,status,1)
                if scoped:
                    from .native_reader_retirement import TrackedDerivativeGeneration
                    self._owner = TrackedDerivativeGeneration(self, self._submission, tuple(self.buffers), native)
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
        if 'tessera.source_state' in self.program.source:
            contract=json.loads(_attribute(self.program.source,'tessera.source_state'))
            product=json.loads(_attribute(self.program.source,'tessera.autodiff.product_abi')) if 'tessera.autodiff.product_abi' in self.program.source else None
            if contract.get('error_specs') and (product is None or product['role']=='forward'):
                import numpy as np
                from .native_source_state import decode_source_exception
                outputs=[]
                # Only completion sidecars cross back to the host. Tensor
                # results remain resident and unexposed on an exception.
                primal=results if product is None else results[:product['primal_results']]
                for view in primal[-((2 if contract.get('error_dynamic') else 1)+len(contract.get('error_payload_sites',()))):]:
                    interface=view.__cuda_array_interface__
                    array=np.empty(interface['shape'],dtype=interface['typestr'])
                    self.check(self.copy_out(array.ctypes.data,view.pointer,array.nbytes))
                    outputs.append(array)
                if not hasattr(self,'_source_exception'):
                    self._source_exception=decode_source_exception(contract,outputs)
                if self._source_exception is not None:
                    # Polling the same failed frame must not retain every prior
                    # caller and its locals in an ever-growing traceback chain.
                    raise self._source_exception.with_traceback(None)
        self._results=tuple(results)
        if self._scoped:
            assert self._owner is not None
            self._owner._reader_buffers = self._results
        else:
            self.results=self._results

    def poll(self):
        with self._lock:
            return self._poll()

    def _poll(self):
        """Expose logical views only after completion and successful status.

        Completion is queried; small status/shape readbacks happen only after
        it succeeds. Closing unrestricted exported views remains synchronous.
        """
        if self.closed or getattr(self, '_retiring', False):
            raise ValueError('public result frame is closed or retiring')
        current=self.context_type(); self.check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('public result requires its owning device context')
        if hasattr(self,'_results'):
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

    def _ready(self, *, allow_retirement_failure=False):
        if self.closed or (self._retiring and not allow_retirement_failure):
            raise ValueError('public result frame is closed or retiring')
        if getattr(self, '_retirement_poisoned', False) and not allow_retirement_failure:
            raise RuntimeError('failed asynchronous free quarantined the public frame')
        context = self.context_type()
        self.check(self.current(ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError('public result requires its owning device context')

    def read(self, stream):
        with self._lock:
            self._ready()
            if not self._scoped or self._owner is None:
                raise ValueError('public reader scopes require scoped submission')
            if not hasattr(self, '_results'):
                raise ValueError('public readers require successful completion and shape checks')
            return self._owner.read(stream)

    def retire(self, stream):
        with self._lock:
            self._ready()
            if not self._scoped or self._owner is None:
                raise ValueError('unrestricted public views require synchronous close')
            try:
                self._owner.retire(stream)
            finally:
                self._retiring = self._owner.retiring
        return self

    def poll_retired(self):
        with self._lock:
            if self.closed:
                return True
            self._ready(allow_retirement_failure=True)
            if not self._retiring or self._owner is None or not self._owner.poll():
                return False
            if not self.binding.close_if_complete(defer_unload=True):
                return False
            self.closed = True
            return True

    def close(self):
        if self.closed:
            return
        if self._scoped and self._owner is not None:
            if not self._retiring:
                self.retire(self._stream)
            self._owner.wait()
            if not self.poll_retired():
                raise RuntimeError('public frame retirement remains pending')
            return
        context = self.context_type(); self.check(self.current(ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError('public result requires its owning device context')
        self.check(self.sync())
        while self.buffers:
            buffer = self.buffers[-1]
            self.check(self.free(buffer.pointer)); buffer.pointer=ct.c_void_p(); self.buffers.pop()
        self.binding.close()
        self._snapshot_borrows=()
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
    if 'tessera.source_state' in source and json.loads(_attribute(source,'tessera.source_state')).get('error_specs'):
        raise ValueError('GPU source exception AD requires a product-aware checked forward binding')
    return _materialize_ad_product(source,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip,capacity=capacity,role=role,input_capacity=input_capacity)


def _materialize_ad_product(source, *, compiler, llvm_bin, backend, chip, capacity, role, input_capacity=0):
    exported=_run(compiler,'--tessera-autodiff-paired=box-product-scalars=true export-product='+role,source=source)
    native=_run(compiler,'--tessera-to-linalg',source=exported)
    buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'+(' copy-before-write=true' if 'tessera.source_state' in source else ''),
        '--convert-linalg-to-loops','--canonicalize',source=native)
    gpu,_,_=_prepare(buffered,compiler,backend,capacity,input_capacity)
    package=build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    return NativePublicResult(buffered,compiler,package,capacity,input_capacity)


@dataclass(frozen=True)
class NativeSourceVJP:
    """Exception-gated reverse execution over owned input snapshots.

    Only the checked forward's residuals enter backward. Completion metadata
    receives zero seeds. Exception objects are transported, not differentiated.
    """
    _forward: NativePublicResult
    _backward: NativePublicResult

    def _validate_pair(self,cotangents):
        forward_abi=json.loads(_attribute(self._forward.source,'tessera.autodiff.product_abi'))
        backward_abi=json.loads(_attribute(self._backward.source,'tessera.autodiff.product_abi'))
        if (_attribute(self._forward.source,'tessera.autodiff.product_pair')!=
                _attribute(self._backward.source,'tessera.autodiff.product_pair')
                or forward_abi['role']!='forward' or backward_abi['role']!='backward'):
            raise ValueError('source VJP products do not share a generated pair')
        contract=json.loads(_attribute(self._forward.source,'tessera.source_state'))
        public_count=contract['result_count']+len(contract['groups'])
        if not isinstance(cotangents,tuple) or len(cotangents)!=public_count:
            raise ValueError('source VJP needs a cotangent per public result and next state')
        return forward_abi,public_count

    def _backward_inputs(self,forward,cotangents,primal_count,public_count,stream=None):
        cuda=self._forward.package.backend=='nvidia'
        driver=forward.binding._bound._driver
        name=('cuMemsetD8_v2' if cuda else 'hipMemset') if stream is None else ('cuMemsetD8Async' if cuda else 'hipMemsetAsync')
        zero=getattr(driver,name)
        zero.argtypes=[ct.c_void_p,ct.c_ubyte if cuda else ct.c_int,ct.c_size_t]+([] if stream is None else [ct.c_void_p])
        zero.restype=ct.c_int
        seeds=list(cotangents)
        for result in forward.results[public_count:primal_count]:
            buffer=_Buffer(forward,result._shape,result._buffer.dtype,stream=stream)
            forward.check(zero(buffer.pointer,0,buffer.nbytes,*(() if stream is None else (ct.c_void_p(stream),))))
            seeds.append(buffer)
        return (*forward._snapshot_inputs,*seeds,*forward.results[primal_count:])

    def run(self,*inputs,cotangents):
        abi,public_count=self._validate_pair(cotangents)
        forward=PublicResultFrame(self._forward,inputs,snapshot=True)
        try:
            backward=self._backward.run(*self._backward_inputs(forward,cotangents,abi['primal_results'],public_count))
            return SourceVJPFrame(forward,backward,public_count)
        except BaseException:
            forward.close()
            raise

    def submit(self,stream,*inputs,cotangents):
        if type(stream) is not int or not 0<stream<(1<<64):
            raise ValueError('source VJP requires a non-null stream')
        return AsyncSourceVJPFrame(self,stream,inputs,cotangents)


class AsyncSourceVJPFrame:
    """Poll-driven forward/check/backward staging; explicit close may synchronize.

    Inputs/cotangents must be ready on the supplied stream and remain owned.
    Snapshot, zero fills and both launches use that stream. No backward is
    enqueued until the forward's device status and exception completion pass.
    """
    def __init__(self,program,stream,inputs,cotangents):
        self._abi,self._public_count=program._validate_pair(cotangents)
        self._program,self._stream=program,stream
        self._cotangents=cotangents
        self._lock=threading.RLock()
        self._backward=None
        self._error=None
        self.closed=False
        self._forward=PublicResultFrame(program._forward,inputs,stream=stream,snapshot=True)

    def poll(self):
        with self._lock:
            if self.closed:raise ValueError('source VJP frame is closed')
            if self._error is not None:raise self._error.with_traceback(None)
            if hasattr(self,'derivatives'):return True
            try:
                if self._backward is None:
                    if not self._forward.poll():return False
                    args=self._program._backward_inputs(self._forward,self._cotangents,self._abi['primal_results'],self._public_count,self._stream)
                    self._backward=self._program._backward.submit(self._stream,*args)
                if not self._backward.poll():return False
                self.primals=self._forward.results[:self._public_count]
                self.derivatives=self._backward.results
                return True
            except BaseException as error:
                self._error=error
                raise

    def close(self):
        with self._lock:
            if self.closed:return
            if self._backward is not None:self._backward.close()
            self._forward.close()
            self._cotangents=()
            self.closed=True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


class SourceVJPFrame:
    def __init__(self,forward,backward,public_count):
        self._forward,self._backward=forward,backward
        self.primals=forward.results[:public_count]
        self.derivatives=backward.results

    def close(self):
        self._backward.close()
        self._forward.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def materialize_source_vjp(source, *, compiler, llvm_bin, backend, chip, capacity):
    """Bind native exception products; standalone backward admission stays closed."""
    if type(capacity) is not int or not 1<=capacity<=1024:
        raise ValueError('source VJP requires capacity from one through 1024')
    contract=json.loads(_attribute(source,'tessera.source_state'))
    if contract.get('schema')!=1 or not contract.get('error_specs'):
        raise ValueError('source VJP requires a serialized source exception contract')
    if len(contract.get('arguments',()))!=1 or contract.get('groups') not in ([],[[0]]) or contract.get('state_views') or contract.get('object_fields'):
        raise ValueError('source VJP currently requires one input without projected aliases or object fields')
    programs=[_materialize_ad_product(source,compiler=Path(compiler),llvm_bin=Path(llvm_bin),
        backend=backend,chip=chip,capacity=capacity,role=role) for role in ('forward','backward')]
    return NativeSourceVJP(*programs)
