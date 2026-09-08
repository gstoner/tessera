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


def _input_status(package):
    if 'tessera.autodiff.input_status' not in package.arena_ir:
        return False
    if _attribute(package.arena_ir,'tessera.autodiff.input_status') != 'guard-v1' or not _checked_status(package):
        raise ValueError('unknown persistent tape input status contract')
    return True


def materialize_persistent_tape(source, *, compiler, llvm_bin, backend, chip, checked_status=False, gated_input=False):
    """Generate, bufferize and materialize both products from one fresh request."""
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    if backend not in ('nvidia','rocm'):
        raise ValueError('persistent tape requires a CUDA or HIP consumer')
    if type(gated_input) is not bool or (gated_input and not checked_status):
        raise ValueError('gated input requires checked status')
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
        gpu=_run(compiler,'--allow-unregistered-dialect','--tessera-native-tape-to-gpu=backend='+backend+(' status-buffer=true' if checked_status else '')+(' input-status=true' if gated_input and role=='backward' else ''),source=buffered)
        inputs=len(contract['inputs'])
        specs=tuple(TensorSpec(f'arg{i}',_dtype(t),shape,i>=inputs)
                    for i,(t,shape) in enumerate(zip(contract['inputs']+contract['results'],shapes,strict=True)))+((TensorSpec('dependency_status','int64',(1,),False),) if gated_input and role=='backward' else ())+((TensorSpec('status','int64',(1,),True),) if checked_status else ())+(IndexSpec('scratch',1,1),)
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
            if role=='forward' and _input_status(package):
                raise ValueError('forward capture cannot consume an upstream derivative status')
            manifest=read_tensor_contract(package)
            inputs=len(c['inputs'])
            expected=tuple(TensorSpec(f'arg{i}',_dtype(t),_shape(t),i>=inputs)
                           for i,t in enumerate(c['inputs']+c['results']))+((TensorSpec('dependency_status','int64',(1,),False),) if _input_status(package) else ())+((TensorSpec('status','int64',(1,),True),) if _checked_status(package) else ())+(IndexSpec('scratch',1,1),)
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

    def capture(self,*inputs,scoped=False,stream=None):
        return PersistentTapeFrame(self,inputs,scoped=scoped,stream=stream)


class _ReadOnly:
    def __init__(self,buffer):
        self._buffer=buffer

    @property
    def __cuda_array_interface__(self):
        view=dict(self._buffer.__cuda_array_interface__)
        view['data']=(view['data'][0],True)
        return view


class PersistentTapeFrame:
    def __init__(self,pair,inputs,*,scoped=False,stream=None):
        if type(scoped) is not bool or (scoped and (type(stream) is not int or not 0<stream<(1<<64))) or (not scoped and stream is not None):
            raise ValueError('scoped frame capture requires a non-null stream')
        self._scoped,self._capture_stream=scoped,stream
        self._retiring=False
        self._frame_owner=None
        f,b=pair.validate()
        if len(inputs)!=f['primal_inputs']:
            raise ValueError('persistent tape primal arity disagrees')
        self.pair=pair
        self.closed=False
        self.buffers: list[_Buffer]=[]
        self._submissions=[]
        self._lock=threading.RLock()
        self._identities=(pair.forward.binding_digest,pair.backward.binding_digest)
        self._checked_status=_checked_status(pair.forward)
        self._bindings=[]
        for package,contract in zip((pair.forward,pair.backward),(f,b),strict=True):
            names=[f'arg{i}' for i in range(len(contract['inputs'])+len(contract['results']))]+(['dependency_status'] if _input_status(package) else [])+(['status'] if self._checked_status else [])+['scratch']
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
        if scoped:
            self.alloc_async=bind('cuMemAllocAsync','hipMallocAsync',[ct.POINTER(P),S,P])
            self.free_async=bind('cuMemFreeAsync','hipFreeAsync',[P,P])
            self.alloc=lambda pointer,size: self.alloc_async(pointer,size,P(stream))
        self.copy=bind('cuMemcpyDtoD_v2','hipMemcpyDtoD',[P,P,S])
        self.context_type=P if cuda else ct.c_int
        self.current=bind('cuCtxGetCurrent','hipGetDevice',[ct.POINTER(self.context_type)])
        self.context=self.context_type()
        self.check(self.current(ct.byref(self.context)))
        try:
            self._status: tuple[_Buffer,...]=(_Buffer(self,(1,),'int64'),) if self._checked_status else ()
            self._inputs=tuple(_Buffer(self,_shape(t),_dtype(t)) for t in f['inputs'])
            self._outputs=tuple(_Buffer(self,_shape(t),_dtype(t)) for t in f['results'])
            if scoped:
                self.check(self.sync())
            self._bindings[0]._resident(*inputs,*self._outputs,*self._status,1)
            self.check(self.sync())
            for source,target in zip(inputs,self._inputs,strict=True):
                self.check(self.copy(target.pointer,P(source.__cuda_array_interface__['data'][0]),target.nbytes))
            self._bindings[0](*self._inputs,*self._outputs,*self._status,1)
            self._check_status()
            self._dependency_status: tuple[_Buffer,...]=()
            if _input_status(pair.backward):
                self._dependency_status=(_Buffer(self,(1,),'int64'),)
                if scoped:
                    self.check(self.sync())
                self.check(self.copy(self._dependency_status[0].pointer,next(iter(self._status)).pointer,8))
            self._primal_count=f['primal_results']
            if scoped:
                from types import SimpleNamespace
                from .native_reader_retirement import TrackedDerivativeGeneration, _record
                capture=_record(native,stream,(self,),[])
                self._frame_owner=TrackedDerivativeGeneration(self,SimpleNamespace(ticket=capture),tuple(self.buffers),native)
                self._frame_owner._reader_buffers=self._outputs
            else:
                self.primals=tuple(_ReadOnly(v) for v in self._outputs[:self._primal_count])
                self.residuals=tuple(_ReadOnly(v) for v in self._outputs[self._primal_count:])
        except BaseException:
            # Failed capture may already have queued copies/allocations; retain
            # the existing conservative synchronous recovery boundary.
            self._scoped=False
            self.close()
            raise

    def _ready(self, *, allow_retirement_failure=False):
        if self.closed or (getattr(self,"_retiring",False) and not allow_retirement_failure):
            raise ValueError('persistent tape frame is closed or retiring')
        if getattr(self,'_retirement_poisoned',False) and not allow_retirement_failure:
            raise RuntimeError('failed asynchronous free quarantined the frame; device teardown is required')
        current=self.context_type()
        self.check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('persistent tape requires its owning device context')
        if self._identities!=(self.pair.forward.binding_digest,self.pair.backward.binding_digest):
            raise ValueError('persistent tape package changed')

    def _check_status(self, status=None):
        if not self._checked_status:
            return
        result=ct.c_int64()
        cuda=self.pair.forward.backend=='nvidia'
        copy=getattr(self._driver,'cuMemcpyDtoH_v2' if cuda else 'hipMemcpyDtoH')
        copy.argtypes,copy.restype=[ct.c_void_p,ct.c_void_p,ct.c_size_t],ct.c_int
        self.check(copy(ct.byref(result),(self._status[0] if status is None else status).pointer,ct.sizeof(result)))
        if result.value!=0:
            raise RuntimeError('persistent GPU product guard failed; outputs are unavailable')

    def backward(self,*cotangents):
        with self._lock:
            self._ready()
            if getattr(self,"_scoped",False):
                raise ValueError('scoped frame derivatives require tracked asynchronous ownership')
            if len(cotangents)!=self._primal_count:
                raise ValueError('persistent tape cotangent arity disagrees')
            start=len(self.buffers)
            try:
                outputs=tuple(_Buffer(self,v.shape,v.dtype) for v in self._inputs)
                self._bindings[1](*self._inputs,*cotangents,*self._outputs[self._primal_count:],*outputs,*self._dependency_status,*self._status,1)
                self._check_status()
                return tuple(_ReadOnly(v) for v in outputs)
            except BaseException:
                self._release(start)
                raise

    def backward_async(self, stream, *cotangents, tracked=False, _dependency=None):
        """Enqueue a distinct derivative generation on a caller-owned stream.

        The event retains the frame and cotangents. Outputs advertise their
        producer stream, so another native tensor binding can order consumers.
        Close still synchronizes before freeing externally visible generations.
        With tracked=True, outputs use stream-ordered pool storage and are exposed
        only by read(stream) scopes; retire(stream) orders frees after all readers.
        Checked tracked products expose only gated or explicitly checked readers;
        untracked checked products require a host status check before exposure.
        The frame's unrestricted primal/residual exports retain their close barrier.
        """
        with self._lock:
            self._ready()
            if type(stream) is not int or not 0 < stream < (1 << 64):
                raise ValueError('persistent tape requires a non-null stream')
            if len(cotangents)!=self._primal_count:
                raise ValueError('persistent tape cotangent arity disagrees')
            if getattr(self,"_scoped",False) and not tracked:
                raise ValueError('scoped frame derivatives require tracked ownership')
            if type(tracked) is not bool:
                raise ValueError('tracked ownership selection must be boolean')
            if tracked:
                cuda=self.pair.forward.backend=='nvidia'
                self.alloc_async=getattr(self._driver,'cuMemAllocAsync' if cuda else 'hipMallocAsync')
                self.free_async=getattr(self._driver,'cuMemFreeAsync' if cuda else 'hipFreeAsync')
                self.alloc_async.argtypes,self.alloc_async.restype=[ct.POINTER(ct.c_void_p),ct.c_size_t,ct.c_void_p],ct.c_int
                self.free_async.argtypes,self.free_async.restype=[ct.c_void_p,ct.c_void_p],ct.c_int
            dependency = self._dependency_status
            if _dependency is not None:
                from .native_reader_retirement import CheckedTrackedDerivativeGeneration
                if not isinstance(_dependency, (CheckedDerivativeSubmission, CheckedTrackedDerivativeGeneration)) or not _input_status(self.pair.backward):
                    raise ValueError('device reader requires a compiler-gated checked product')
                if _dependency._released:
                    raise ValueError('upstream derivative generation is released')
                dependency=(_dependency._status_buffer,)
                _dependency.submission.ticket.wait_on(stream)
            start=len(self.buffers)
            try:
                outputs=tuple(_Buffer(self,v.shape,v.dtype,stream=stream if tracked else None) for v in self._inputs)
                status: tuple[_Buffer, ...] = (_Buffer(self,(1,),'int64',stream=stream if tracked else None),) if self._checked_status else ()
                submission=self._bindings[1].submit(stream,*self._inputs,*cotangents,
                    *self._outputs[self._primal_count:],*outputs,*dependency,*status,1)
                from .native_reader_retirement import TrackedDerivativeGeneration
                result: TrackedDerivativeGeneration | PersistentDerivativeSubmission
                if self._checked_status and tracked:
                    from .native_reader_retirement import CheckedTrackedDerivativeGeneration
                    result=CheckedTrackedDerivativeGeneration(self,submission,outputs,status[0],self._bindings[1]._bound)
                elif self._checked_status:
                    result=CheckedDerivativeSubmission(self,submission,outputs,stream,status[0])
                elif tracked:
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

    def read(self, stream):
        """Borrow captured primals/residuals only for this declared reader stream."""
        with self._lock:
            self._ready()
            if not self._scoped or self._frame_owner is None:
                raise ValueError('scoped capture is required for frame reader tracking')
            return self._frame_owner.read(stream)

    def retire(self, stream):
        """Enqueue complete scoped-frame storage retirement after all readers."""
        from .native_reader_retirement import TrackedDerivativeGeneration
        with self._lock:
            self._ready()
            if not self._scoped or self._frame_owner is None:
                raise ValueError('unrestricted frame exports require synchronous close')
            generations=tuple(self._submissions)
            if self._frame_owner._active or any(not isinstance(g,TrackedDerivativeGeneration) or g._active for g in generations):
                raise ValueError('frame retirement requires closed scoped readers')
            # Each generation retains every internal frame read until its own
            # completion; retire events additionally cover its external readers.
            for generation in generations:
                if not generation.retiring:
                    generation.retire(stream)
                self._frame_owner._readers.extend(generation._retirements)
            try:
                self._frame_owner.retire(stream)
            finally:
                self._retiring=self._frame_owner.retiring
        return self

    def poll_retired(self):
        """Query storage completion and release idle modules without a context wait."""
        with self._lock:
            if self.closed:
                return True
            self._ready(allow_retirement_failure=True)
            if not self._retiring:
                return False
            if self._frame_owner is None:
                raise RuntimeError('scoped frame has no retirement owner')
            for generation in tuple(self._submissions):
                if not generation.poll():
                    return False
            if not self._frame_owner.poll():
                return False
            if not all(binding.close_if_complete() for binding in self._bindings):
                return False
            self.closed=True
            return True

    def close(self):
        with self._lock:
            if self.closed:
                return
            if getattr(self,"_scoped",False):
                if not self._retiring:
                    self.retire(self._capture_stream)
                for generation in tuple(self._submissions):
                    generation.wait()
                if self._frame_owner is None:
                    raise RuntimeError('scoped frame has no retirement owner')
                self._frame_owner.wait()
                if not self.poll_retired():
                    raise RuntimeError('scoped frame completion remained pending after wait')
                return
            self._ready()
            for submission in tuple(self._submissions):
                if isinstance(submission, CheckedDerivativeSubmission):
                    submission.submission.wait()
                else:
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
        self._views=tuple(_ProducedReadOnly(v,stream) for v in outputs)

    @property
    def outputs(self):
        return self._views

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


class CheckedDerivativeSubmission(PersistentDerivativeSubmission):
    """Asynchronous execution with host-checked success before pointer exposure.

    Each generation has its own status allocation. Completion alone is never
    success, and a failure is cached until release. Unrestricted successful
    exports retain the conservative context barrier on release.
    """
    def __init__(self, frame, submission, outputs, stream, status):
        self.frame, self.submission = frame, submission
        self._buffers = (*outputs, status)
        self._views = tuple(_ProducedReadOnly(v, stream) for v in outputs)
        self._status_buffer = status
        self._released = False
        self._verified = False
        self._failure = None

    @property
    def outputs(self):
        with self.frame._lock:
            self.frame._ready()
            if self._released:
                raise ValueError('persistent derivative generation is released')
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if not self._verified:
                raise ValueError('checked derivative results require successful wait or poll')
            return self._views

    def _verify(self):
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if not self._verified:
            try:
                self.frame._check_status(self._status_buffer)
            except RuntimeError as error:
                self._failure = str(error)
                raise
            self._verified = True

    def wait(self):
        with self.frame._lock:
            self.frame._ready()
            if self._released:
                raise ValueError('persistent derivative generation is released')
            self.submission.wait()
            self._verify()
            return self.outputs

    def poll(self):
        with self.frame._lock:
            self.frame._ready()
            if self._released:
                return True
            if not self.submission.ticket.poll():
                return False
            self._verify()
            return True


    def backward_into(self, frame, stream):
        """Order a compiler-gated downstream reader without host status readback."""
        from contextlib import ExitStack
        if not isinstance(frame, PersistentTapeFrame):
            raise TypeError('device-gated derivative reader requires a persistent frame')
        # Consistent lock order permits reciprocal frame composition safely.
        with ExitStack() as stack:
            for owner in sorted({self.frame, frame}, key=id):
                stack.enter_context(owner._lock)
                owner._ready()
            if self._released:
                raise ValueError('upstream derivative generation is released')
            if (self.frame.pair.forward.backend,self.frame.pair.forward.chip) != (frame.pair.forward.backend,frame.pair.forward.chip):
                raise ValueError('device-gated reader requires the same owning target')
            if not _input_status(frame.pair.backward):
                raise ValueError('device reader requires a compiler-gated checked product')
            return frame.backward_async(stream,*self._views,_dependency=self)
