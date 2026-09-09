"""Explicit full-tensor state effects for the native CPU source-CFG consumer.

Input aliases are part of serialized IR. Native execution uses snapshots and
separate result storage; declared state is copied back only after completion.
Callers must own these arrays exclusively throughout an invocation. This does
not implement concurrent mutation, overlapping writes or aliased return objects.
"""
import hashlib
import json
import re
import threading
import numpy as np
from .native_persistent_tape import _attribute
from .source_control_flow import to_native_source_ir
from .trace import trace


def _aliases(arrays, *, allow_partial=False):
    groups: list[list[int]]=[]
    for index,array in enumerate(arrays):
        if type(array) is not np.ndarray or array.dtype not in (np.dtype('float32'),np.dtype('float64')) or not array.size or not array.ndim:
            raise ValueError('native source state requires nonempty fp32/fp64 arrays')
        # A snapshot does not preserve overlapping logical elements. Reject
        # broadcast and self-overlapping views before accepting state effects.
        stride_extent=array.itemsize
        for stride,extent in sorted(zip(map(abs,array.strides),array.shape,strict=True)):
            if not allow_partial and extent>1 and stride<stride_extent:
                raise ValueError('self-overlapping state views are unsupported')
            stride_extent+=stride*(extent-1)
        group=None
        for prior,other in enumerate(arrays[:index]):
            if np.shares_memory(array,other,max_work=1000):
                if (array.__array_interface__['data'][0]!=other.__array_interface__['data'][0]
                        or array.shape!=other.shape or array.strides!=other.strides or array.dtype!=other.dtype):
                    if allow_partial:continue
                    raise ValueError('partial or differently shaped state aliases are unsupported')
                group=next(row for row in groups if prior in row)
                break
        if group is None:groups.append([index])
        else:group.append(index)
    return groups


def _output(type_name):
    match=re.fullmatch(r'tensor<((?:[0-9]+x)*)f(32|64)>',type_name)
    if match is None:raise ValueError('state outputs require static floating tensors')
    shape=tuple(int(n) for n in match[1].split('x') if n)
    return np.empty(shape,dtype='float'+match[2])


class NativeSourceStateProgram:
    def __init__(self,native_ir):
        from tessera import _jit_boundary as jit
        self._ir=native_ir
        self._digest=hashlib.sha256(native_ir.encode()).hexdigest()
        self._lock=threading.RLock()
        self._handle: int | None=jit.compile_module(native_ir)

    @property
    def native_ir(self):
        return self._ir

    def run(self,*arrays):
        from tessera import _jit_boundary as jit
        with self._lock:
            if self._handle is None:raise ValueError('native source state program is closed')
            if hashlib.sha256(self._ir.encode()).hexdigest()!=self._digest:
                raise ValueError('native source state artifact changed')
            contract=json.loads(_attribute(self._ir,'tessera.source_state'))
            if contract['schema']!=1 or len(arrays)!=len(contract['arguments']):
                raise ValueError('state argument contract disagrees')
            if _aliases(arrays,allow_partial=not contract['groups'])!=contract['aliases']:
                raise ValueError('state alias topology disagrees with compiled IR')
            for array,spec in zip(arrays,contract['arguments'],strict=True):
                if list(array.shape)!=spec['shape'] or array.dtype!=np.dtype({'f32':'float32','f64':'float64'}[spec['dtype']]):
                    raise ValueError('state shape or dtype disagrees with compiled IR')
            for group in contract['groups']:
                if any(not arrays[index].flags.writeable for index in group):
                    raise ValueError('declared state alias is read-only')
            snapshots=[array.copy() for array in arrays]
            outputs=[_output(type_name) for type_name in contract['outputs']]
            jit.invoke(self._handle,'source_program',snapshots,outputs)
            count=contract['result_count']
            for group,result in zip(contract['groups'],outputs[count:count+len(contract['groups'])],strict=True):
                np.copyto(arrays[group[0]],result,casting='no')
            if contract.get('error_specs'):
                code=float(outputs[-1].item())
                import builtins
                allowed={'Exception','ValueError','RuntimeError','AssertionError','TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError'}
                table=contract.get('error_table',())
                if code != 0:
                    if not code.is_integer() or not 1<=code<=len(table):raise RuntimeError('invalid native source exception status')
                    kind,payload=table[int(code)-1]
                    if kind not in allowed or not isinstance(payload,list) or any(type(v) is not str for v in payload):
                        raise RuntimeError('invalid native source exception payload')
                    raise getattr(builtins,kind)(*payload)
            values=tuple(outputs[:count])
            return values[0] if len(values)==1 else values

    def close(self):
        from tessera import _jit_boundary as jit
        with self._lock:
            if self._handle is not None:
                jit.destroy(self._handle)
                self._handle=None

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()


def compile_source_state(fn,*arrays,mutable,max_steps=None,error_specs=(),object_fields=()):
    """Compile declared full-slice writes, preserving exact input alias groups."""
    aliases=_aliases(arrays,allow_partial=not mutable)
    if not isinstance(mutable,tuple) or any(type(i) is not int or not 0<=i<len(arrays) for i in mutable) or len(set(mutable))!=len(mutable):
        raise ValueError('mutable must name distinct input argument indices')
    groups=[]
    for group in aliases:
        targets=[i for i in mutable if i in group]
        if targets:
            primary=targets[0]
            if any(not arrays[index].flags.writeable for index in group):raise ValueError('declared state alias is read-only')
            groups.append((primary,*(i for i in group if i!=primary)))
    traced=trace(fn,*arrays,source_control_flow=True,max_steps=max_steps,source_state_groups=tuple(groups),source_error_specs=error_specs,source_object_fields=object_fields)
    native=to_native_source_ir(traced)
    contract=json.loads(_attribute(native,'tessera.source_state'))
    old=json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    contract['aliases']=aliases
    new=json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    if native.count(old)!=1:raise ValueError('source state contract is not unique')
    return NativeSourceStateProgram(native.replace(old,new,1))


class NativeSourceJit:
    """Bounded lazy native source specialization with explicit module lifetime."""

    def __init__(self,fn,*,mutable=(),error_specs=(),max_steps=None,object_fields=()):
        import inspect
        from collections import OrderedDict
        from functools import update_wrapper
        self.fn=fn
        self.signature=inspect.signature(fn)
        self.mutable,self.error_specs,self.max_steps=mutable,error_specs,max_steps
        if not isinstance(object_fields,tuple) or any(not isinstance(row,tuple) or len(row)!=2 or type(row[0]) is not int or not 0<=row[0]<len(self.signature.parameters) or not isinstance(row[1],tuple) or not row[1] or any(type(f) is not str or not f.isidentifier() or f.startswith('_') for f in row[1]) or len(set(row[1]))!=len(row[1]) for row in object_fields) or len({row[0] for row in object_fields})!=len(object_fields):
            raise ValueError('object fields require unique parameter indices and explicit field names')
        self.object_fields=object_fields
        self._programs=OrderedDict()
        self._lock=threading.RLock()
        self._closed=False
        update_wrapper(self,fn)

    def __call__(self,*args,**kwargs):
        with self._lock:
            if self._closed:raise ValueError('native source JIT is closed')
            bound=self.signature.bind(*args,**kwargs)
            bound.apply_defaults()
            from types import SimpleNamespace
            arrays_list: list[np.ndarray]=[]
            mutable: list[int]=[]
            fields=dict(self.object_fields)
            for index,value in enumerate(bound.arguments.values()):
                if index in fields:
                    if type(value) not in (dict,SimpleNamespace):raise ValueError('source objects require exact dict or SimpleNamespace storage; custom accessors are unsupported')
                    data=value if type(value) is dict else vars(value)
                    children=[data[field] for field in fields[index]]
                else:children=[value]
                if index in self.mutable:mutable.extend(range(len(arrays_list),len(arrays_list)+len(children)))
                arrays_list.extend(children)
            arrays=tuple(arrays_list)
            aliases=_aliases(arrays,allow_partial=not mutable)
            key=(tuple((a.shape,a.dtype.str) for a in arrays),tuple(map(tuple,aliases)))
            program=self._programs.get(key)
            if program is None:
                # Evict before compilation to bound live native module ownership.
                if len(self._programs)>=4:
                    _,old=self._programs.popitem(last=False)
                    old.close()
                program=compile_source_state(self.fn,*arrays,mutable=tuple(mutable),
                                             max_steps=self.max_steps,error_specs=self.error_specs,object_fields=self.object_fields)
                self._programs[key]=program
            self._programs.move_to_end(key)
            return program.run(*arrays)

    def vjp(self,*arrays,cotangents):
        """Execute compiler-exported CPU forward/backward products for pure source."""
        from tessera import _jit_boundary as jit
        from .scheduled_matmul import find_tessera_opt, run_tessera_opt
        from .native_persistent_tape import _shape, _dtype
        with self._lock:
            if self._closed:raise ValueError('native source JIT is closed')
            if self.mutable or self.object_fields or self.error_specs:
                raise ValueError('source VJP currently requires pure tensor inputs and no exception transport')
            _aliases(arrays,allow_partial=True)
            traced=trace(self.fn,*arrays,source_control_flow=True,max_steps=self.max_steps)
            count=len(traced.outputs)
            if not isinstance(cotangents,tuple) or len(cotangents)!=count:
                raise ValueError('source VJP requires one cotangent per result')
            source=to_native_source_ir(traced,autodiff='reverse')
            compiler=find_tessera_opt()
            if compiler is None:raise ValueError('source VJP requires the native compiler')
            handles=[]
            contracts=[]
            def allocate(types):
                names={'fp32':'float32','fp64':'float64','int64':'int64','int8':'int8'}
                return [np.zeros(_shape(t),dtype=names[_dtype(t)]) for t in types]
            try:
                for role in ('forward','backward'):
                    exported=run_tessera_opt(compiler,source,'--tessera-autodiff-paired=box-product-scalars=true normalize-data-while=true export-product='+role)
                    contracts.append(json.loads(_attribute(exported,'tessera.autodiff.product_abi')))
                    handles.append(jit.compile_module(exported))
                outputs=allocate(contracts[0]['results'])
                for cotangent,primal in zip(cotangents,outputs[:count],strict=True):
                    if type(cotangent) is not np.ndarray or cotangent.shape!=primal.shape or cotangent.dtype!=primal.dtype:
                        raise ValueError('source VJP cotangent shape or dtype disagrees')
                snapshots=[a.copy() for a in arrays]
                jit.invoke(handles[0],contracts[0]['entry'],snapshots,outputs)
                derivatives=allocate(contracts[1]['results'])
                jit.invoke(handles[1],contracts[1]['entry'],[*snapshots,*[c.copy() for c in cotangents],*outputs[count:]],derivatives)
                return tuple(outputs[:count]),tuple(derivatives)
            finally:
                for handle in handles:jit.destroy(handle)

    def close(self):
        with self._lock:
            for program in self._programs.values():program.close()
            self._programs.clear()
            self._closed=True

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()


def materialize_source_state(native_ir, *, compiler, llvm_bin, backend, chip, capacity):
    """Produce device-resident next-state results from serialized source IR.

    Results are immutable owned generations. This does not mutate caller device
    storage in place; callers retain the returned frame until all readers finish.
    """
    from pathlib import Path
    from .native_gpu_storage import _run, build_native_gpu_storage
    from .native_public_result import _prepare, NativePublicResult
    if type(capacity) is not int or not 1<=capacity<=1024:
        raise ValueError('source state requires a capacity from one through 1024')
    contract=json.loads(_attribute(native_ir,'tessera.source_state'))
    if contract.get('schema')!=1 or contract.get('error_specs'):
        raise ValueError('GPU source state currently requires an exception-free state contract')
    if len(contract.get('arguments',()))!=1 or contract.get('groups')!=[[0]]:
        raise ValueError('GPU source state requires one declared state input until multi-input alias projection is available')
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    native=_run(compiler,'--tessera-to-linalg',source=native_ir)
    buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
        '--convert-linalg-to-loops','--canonicalize',source=native)
    gpu,_,_=_prepare(buffered,compiler,backend,capacity)
    package=build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip)
    return NativePublicResult(buffered,compiler,package,capacity)


class OwnedSourceGPUState:
    """Exclusive synchronous state storage; borrowed readers block mutation.

    Construction executes the initial step into privately owned state. step()
    computes into a fresh result frame, then copies its checked state into this
    owner's fixed allocation. Returned result frames are independently owned.
    """
    def __init__(self,program,value):
        contract=json.loads(_attribute(program.source,'tessera.source_state'))
        if len(contract.get('arguments',()))!=1 or contract.get('groups')!=[[0]] or contract.get('error_specs'):
            raise ValueError('owned GPU mutation requires one exception-free state input')
        self._program=program
        self._lock=threading.RLock()
        self._readers=0
        self._closed=False
        self._poisoned=False
        self._quarantine=[]
        self._frame=program.run(value)

    def _ready(self):
        if self._closed or self._poisoned:raise ValueError('owned GPU state is closed or poisoned')
        self._frame._ready()

    def read(self):
        from contextlib import contextmanager
        owner=self
        @contextmanager
        def scope():
            with owner._lock:
                owner._ready()
                owner._readers+=1
                active=[True]
                class View:
                    @property
                    def __cuda_array_interface__(self):
                        with owner._lock:
                            owner._ready()
                            if not active[0]:raise ValueError('GPU state reader scope ended')
                            return owner._frame.results[-1].__cuda_array_interface__
            try:yield View()
            finally:
                with owner._lock:
                    active[0]=False
                    try:owner._frame.check(owner._frame.sync())
                    except BaseException:
                        owner._poisoned=True
                        raise
                    finally:owner._readers-=1
        return scope()

    def step(self):
        import ctypes as ct
        with self._lock:
            self._ready()
            if self._readers:raise ValueError('GPU state mutation requires all reader scopes to end')
            result=self._program.run(self._frame.results[-1])
            try:
                destination=self._frame.results[-1]
                source=result.results[-1]
                dest=destination.__cuda_array_interface__
                src=source.__cuda_array_interface__
                if (dest['shape'],dest['typestr'])!=(src['shape'],src['typestr']):
                    raise ValueError('GPU next-state ABI changed')
                nbytes=int(np.prod(dest['shape']))*np.dtype(dest['typestr']).itemsize
                native=self._frame.binding._bound
                cuda=self._program.package.backend=='nvidia'
                copy=getattr(native._driver,'cuMemcpyDtoD_v2' if cuda else 'hipMemcpy')
                copy.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t]+([] if cuda else [ct.c_int])
                copy.restype=ct.c_int
                self._frame.check(copy(destination.pointer,source.pointer,nbytes,*(() if cuda else (3,))))
                self._frame.check(self._frame.sync())
                return result
            except BaseException:
                self._poisoned=True
                self._quarantine.append(result)
                raise

    def close(self):
        with self._lock:
            if self._closed:return
            if self._readers:raise ValueError('GPU state has active readers')
            for frame in self._quarantine:frame.close()
            self._quarantine.clear()
            self._frame.close()
            self._closed=True

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()
