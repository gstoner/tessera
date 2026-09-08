"""Split compiler-produced AD products with persistent CUDA/HIP residual storage.

The physical envelope is static floating/integer tensor storage and bounded for/if regions,
executed serially on one GPU thread. Nested inner states may still be replayed
by the compiler's backward program; exported residuals persist across calls.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import inspect
import json
from pathlib import Path
import re
import threading
from .native_gpu_storage import _run, _decode_image, build_native_gpu_storage, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding, read_tensor_contract, tensor_contract_specs
from .native_device_tape import _Buffer


def _attribute(text,name):
    fields=re.findall(re.escape(name)+r' = "((?:\\.|[^"\\])*)"',text)
    if len(fields)!=1:
        raise ValueError('split tape requires unique compiler ABI and lineage')
    return _decode_image(fields[0]).decode()


def _shape(type_name):
    match=re.fullmatch(r'tensor<((?:[1-9][0-9]*x)*)(f32|f64|i8|i64)>',type_name)
    if not match:
        raise ValueError('persistent GPU tape requires static f32/f64/i8/i64 tensor slots')
    shape=tuple(int(x) for x in match[1].split('x') if x)
    count=1
    for dim in shape:
        if dim>1024 or count>1024//dim:
            raise ValueError('persistent GPU tape slot exceeds 1024 elements')
        count*=dim
    return shape


def _dtype(type_name):
    _shape(type_name)
    return {'f32':'fp32','f64':'fp64','i8':'int8','i64':'int64'}[type_name.split('x')[-1].removeprefix('tensor<').removesuffix('>')]


def _checked_status(package):
    if 'tessera.autodiff.gpu_status' not in package.arena_ir:
        return False
    if _attribute(package.arena_ir,'tessera.autodiff.gpu_status')!='guard-v1':
        raise ValueError('unknown persistent tape GPU status contract')
    return True


def materialize_persistent_tape(source, *, compiler, llvm_bin, backend, chip, checked_status=False):
    """Generate, bufferize and materialize both products from one fresh request."""
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    if backend not in ('nvidia','rocm'):
        raise ValueError('persistent tape requires a CUDA or HIP consumer')
    if type(checked_status) is not bool:
        raise ValueError('checked status selection must be boolean')
    packages=[]
    contracts=[]
    lineages=[]
    for role in ('forward','backward'):
        exported=_run(compiler,'--tessera-autodiff-paired=normalize-counted-while=true normalize-data-while=true box-product-scalars=true export-product='+role,source=source)
        contract=json.loads(_attribute(exported,'tessera.autodiff.product_abi'))
        contracts.append(contract)
        lineages.append(_attribute(exported,'tessera.autodiff.product_pair'))
        shapes=[_shape(t) for t in contract['inputs']+contract['results']]
        native=_run(compiler,'--tessera-to-linalg',source=exported)
        buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect',
            '--convert-elementwise-to-linalg',
            '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
            '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
            '--convert-linalg-to-loops','--canonicalize',source=native)
        gpu=_run(compiler,'--allow-unregistered-dialect','--tessera-native-tape-to-gpu=backend='+backend+(' status-buffer=true' if checked_status else ''),source=buffered)
        inputs=len(contract['inputs'])
        specs=tuple(TensorSpec(f'arg{i}',_dtype(t),shape,i>=inputs)
                    for i,(t,shape) in enumerate(zip(contract['inputs']+contract['results'],shapes,strict=True)))+((TensorSpec('status','int64',(1,),True),) if checked_status else ())+(IndexSpec('scratch',1,1),)
        first,rest=gpu.split('\n',1)
        prefix='module attributes {'
        if not first.startswith(prefix) or not first.endswith('} {'):
            raise ValueError('native tape module lacks its product attributes')
        attributes=first[len(prefix):-3]
        gpu=attach_tensor_contract('module {\n'+rest,specs,grid=(1,1,1),block=(1,1,1))
        gpu=gpu.replace(prefix,prefix+attributes+', ',1)
        packages.append(build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip))
    if lineages[0]!=lineages[1]:
        raise ValueError('persistent tape native products have different lineage')
    f,b=contracts
    if b['inputs']!=f['inputs']+f['results'] or b['results']!=f['inputs']:
        raise ValueError('persistent tape requires gradients for every primal input')
    result=PersistentTapePair(packages[0],packages[1],hashlib.sha256(lineages[0].encode()).hexdigest())
    result.validate()
    return result


@dataclass(frozen=True)
class PersistentTapePair:
    forward: NativeGPUStoragePackage
    backward: NativeGPUStoragePackage
    lineage_digest: str

    def validate(self):
        contracts=[]
        for role,package in zip(('forward','backward'),(self.forward,self.backward),strict=True):
            package.validate()
            c=json.loads(_attribute(package.arena_ir,'tessera.autodiff.product_abi'))
            lineage=_attribute(package.arena_ir,'tessera.autodiff.product_pair')
            if c.get('role')!=role or c.get('schema')!=1 or hashlib.sha256(lineage.encode()).hexdigest()!=self.lineage_digest:
                raise ValueError('persistent tape product identity disagrees')
            manifest=read_tensor_contract(package)
            inputs=len(c['inputs'])
            expected=tuple(TensorSpec(f'arg{i}',_dtype(t),_shape(t),i>=inputs)
                           for i,t in enumerate(c['inputs']+c['results']))+((TensorSpec('status','int64',(1,),True),) if _checked_status(package) else ())+(IndexSpec('scratch',1,1),)
            if (tensor_contract_specs(manifest)!=expected or manifest['grid']!=[1,1,1]
                    or manifest['block']!=[1,1,1]):
                raise ValueError('persistent tape tensor binding disagrees with native product ABI')
            contracts.append(c)
        if (self.forward.backend,self.forward.chip)!=(self.backward.backend,self.backward.chip):
            raise ValueError('persistent tape products require the same backend')
        if _checked_status(self.forward)!=_checked_status(self.backward):
            raise ValueError('persistent tape status contracts disagree')
        f,b=contracts
        if b['inputs']!=f['inputs']+f['results'] or b['results']!=f['inputs']:
            raise ValueError('persistent tape residual ABI disagrees')
        return f,b

    def capture(self,*inputs):
        return PersistentTapeFrame(self,inputs)


class _ReadOnly:
    def __init__(self,buffer):
        self._buffer=buffer

    @property
    def __cuda_array_interface__(self):
        view=dict(self._buffer.__cuda_array_interface__)
        view['data']=(view['data'][0],True)
        return view


class PersistentTapeFrame:
    def __init__(self,pair,inputs):
        f,b=pair.validate()
        if len(inputs)!=f['primal_inputs']:
            raise ValueError('persistent tape primal arity disagrees')
        self.pair=pair
        self.closed=False
        self.buffers=[]
        self._submissions=[]
        self._lock=threading.RLock()
        self._identities=(pair.forward.binding_digest,pair.backward.binding_digest)
        self._checked_status=_checked_status(pair.forward)
        self._bindings=[]
        for package,contract in zip((pair.forward,pair.backward),(f,b),strict=True):
            names=[f'arg{i}' for i in range(len(contract['inputs'])+len(contract['results']))]+(['status'] if self._checked_status else [])+['scratch']
            signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in names])
            self._bindings.append(generate_tensor_binding(package,signature))
        self._bindings[0]._bound=pair.forward.bind()
        native=self._bindings[0]._bound
        self.check,self.sync=native._check,native._sync
        self._driver=native._driver
        cuda=pair.forward.backend=='nvidia'
        P,S=ct.c_void_p,ct.c_size_t
        def bind(cu,hip,args):
            fn=getattr(self._driver,cu if cuda else hip)
            fn.argtypes,fn.restype=args,ct.c_int
            return fn
        self.alloc=bind('cuMemAlloc_v2','hipMalloc',[ct.POINTER(P),S])
        self.free=bind('cuMemFree_v2','hipFree',[P])
        self.copy=bind('cuMemcpyDtoD_v2','hipMemcpyDtoD',[P,P,S])
        self.context_type=P if cuda else ct.c_int
        self.current=bind('cuCtxGetCurrent','hipGetDevice',[ct.POINTER(self.context_type)])
        self.context=self.context_type()
        self.check(self.current(ct.byref(self.context)))
        try:
            self._status: tuple[_Buffer,...]=(_Buffer(self,(1,),'int64'),) if self._checked_status else ()
            self._inputs=tuple(_Buffer(self,_shape(t),_dtype(t)) for t in f['inputs'])
            self._outputs=tuple(_Buffer(self,_shape(t),_dtype(t)) for t in f['results'])
            self._bindings[0]._resident(*inputs,*self._outputs,*self._status,1)
            self.check(self.sync())
            for source,target in zip(inputs,self._inputs,strict=True):
                self.check(self.copy(target.pointer,P(source.__cuda_array_interface__['data'][0]),target.nbytes))
            self._bindings[0](*self._inputs,*self._outputs,*self._status,1)
            self._check_status()
            self._primal_count=f['primal_results']
            self.primals=tuple(_ReadOnly(v) for v in self._outputs[:self._primal_count])
            self.residuals=tuple(_ReadOnly(v) for v in self._outputs[self._primal_count:])
        except BaseException:
            self.close()
            raise

    def _ready(self, *, allow_retirement_failure=False):
        if self.closed:
            raise ValueError('persistent tape frame is closed')
        if getattr(self,'_retirement_poisoned',False) and not allow_retirement_failure:
            raise RuntimeError('failed asynchronous free quarantined the frame; device teardown is required')
        current=self.context_type()
        self.check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('persistent tape requires its owning device context')
        if self._identities!=(self.pair.forward.binding_digest,self.pair.backward.binding_digest):
            raise ValueError('persistent tape package changed')

    def _check_status(self):
        if not self._checked_status:
            return
        result=ct.c_int64()
        cuda=self.pair.forward.backend=='nvidia'
        copy=getattr(self._driver,'cuMemcpyDtoH_v2' if cuda else 'hipMemcpyDtoH')
        copy.argtypes,copy.restype=[ct.c_void_p,ct.c_void_p,ct.c_size_t],ct.c_int
        self.check(copy(ct.byref(result),self._status[0].pointer,ct.sizeof(result)))
        if result.value!=0:
            raise RuntimeError('persistent GPU product guard failed; outputs are unavailable')

    def backward(self,*cotangents):
        with self._lock:
            self._ready()
            if len(cotangents)!=self._primal_count:
                raise ValueError('persistent tape cotangent arity disagrees')
            start=len(self.buffers)
            try:
                outputs=tuple(_Buffer(self,v.shape,v.dtype) for v in self._inputs)
                self._bindings[1](*self._inputs,*cotangents,*self._outputs[self._primal_count:],*outputs,*self._status,1)
                self._check_status()
                return tuple(_ReadOnly(v) for v in outputs)
            except BaseException:
                self._release(start)
                raise

    def backward_async(self, stream, *cotangents, tracked=False):
        """Enqueue a distinct derivative generation on a caller-owned stream.

        The event retains the frame and cotangents. Outputs advertise their
        producer stream, so another native tensor binding can order consumers.
        Close still synchronizes before freeing externally visible generations.
        With tracked=True, outputs use stream-ordered pool storage and are exposed
        only by read(stream) scopes; retire(stream) orders frees after all readers.
        The frame's unrestricted primal/residual exports retain their close barrier.
        """
        with self._lock:
            self._ready()
            if self._checked_status:
                raise ValueError('checked GPU products require synchronous status consumption')
            if type(stream) is not int or not 0 < stream < (1 << 64):
                raise ValueError('persistent tape requires a non-null stream')
            if len(cotangents)!=self._primal_count:
                raise ValueError('persistent tape cotangent arity disagrees')
            if type(tracked) is not bool:
                raise ValueError('tracked ownership selection must be boolean')
            if tracked:
                cuda=self.pair.forward.backend=='nvidia'
                self.alloc_async=getattr(self._driver,'cuMemAllocAsync' if cuda else 'hipMallocAsync')
                self.free_async=getattr(self._driver,'cuMemFreeAsync' if cuda else 'hipFreeAsync')
                self.alloc_async.argtypes,self.alloc_async.restype=[ct.POINTER(ct.c_void_p),ct.c_size_t,ct.c_void_p],ct.c_int
                self.free_async.argtypes,self.free_async.restype=[ct.c_void_p,ct.c_void_p],ct.c_int
            start=len(self.buffers)
            try:
                outputs=tuple(_Buffer(self,v.shape,v.dtype,stream=stream if tracked else None) for v in self._inputs)
                submission=self._bindings[1].submit(stream,*self._inputs,*cotangents,
                    *self._outputs[self._primal_count:],*outputs,1)
                from .native_reader_retirement import TrackedDerivativeGeneration
                result: TrackedDerivativeGeneration | PersistentDerivativeSubmission
                if tracked:
                    result=TrackedDerivativeGeneration(self,submission,outputs,self._bindings[1]._bound)
                else:
                    result=PersistentDerivativeSubmission(self,submission,outputs,stream)
                self._submissions.append(result)
                return result
            except BaseException:
                # A launch may have succeeded before event recording failed.
                # Do not release any allocation without a completion proof.
                self._release(start)
                raise

    def poll(self):
        """Retire completed event owners without a device-wide wait."""
        with self._lock:
            self._ready()
            for submission in tuple(self._submissions):
                if submission.poll():
                    if submission in self._submissions:
                        self._submissions.remove(submission)
            return not self._submissions

    def _release(self,start):
        self.check(self.sync())
        while len(self.buffers)>start:
            buffer=self.buffers[-1]
            self.check(self.free(buffer.pointer))
            buffer.pointer=ct.c_void_p()
            self.buffers.pop()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready()
            for submission in tuple(self._submissions):
                submission.wait()
            self._submissions.clear()
            self._release(0)
            for binding in self._bindings:
                binding.close()
            self.closed=True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self,*exc):
        self.close()


class _ProducedReadOnly(_ReadOnly):
    def __init__(self, buffer, stream):
        super().__init__(buffer)
        self._stream=stream

    @property
    def __cuda_array_interface__(self):
        return {**super().__cuda_array_interface__, 'stream': self._stream}


class PersistentDerivativeSubmission:
    """An asynchronous derivative generation owned by its persistent frame."""
    def __init__(self, frame, submission, outputs, stream):
        self.frame, self.submission=frame,submission
        self._buffers=outputs
        self._released=False
        self.outputs=tuple(_ProducedReadOnly(v,stream) for v in outputs)

    def wait(self):
        with self.frame._lock:
            self.frame._ready()
            if self._released:
                raise ValueError('persistent derivative generation is released')
            self.submission.wait()
            return self.outputs

    def poll(self):
        with self.frame._lock:
            self.frame._ready()
            return self.submission.ticket.poll()

    def release(self):
        """Release this generation after all device readers have completed.

        Exported views can have downstream consumers outside this binding, so
        allocation release keeps a context completion barrier. Event polling
        alone retires submission owners, not arbitrary external readers.
        """
        with self.frame._lock:
            if self._released:
                return
            self.frame._ready()
            self.submission.wait()
            self.frame.check(self.frame.sync())
            for buffer in self._buffers:
                if buffer.pointer.value:
                    self.frame.check(self.frame.free(buffer.pointer))
                    buffer.pointer=ct.c_void_p()
                    self.frame.buffers.remove(buffer)
            self._released=True
            if self in self.frame._submissions:
                self.frame._submissions.remove(self)
