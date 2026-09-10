"""Explicit full-tensor state effects for the native CPU source-CFG consumer.

Input aliases are part of serialized IR. Native execution uses snapshots and
separate result storage; declared state is copied back only after completion.
Callers must own these arrays exclusively throughout an invocation. This does
not implement concurrent mutation or aliased return objects. Bounded mapped
writes share an explicit containing root and execute in source order.
"""
import hashlib
import json
import re
import threading
import numpy as np
from .native_persistent_tape import _attribute
from .source_control_flow import to_native_source_ir
from .trace import trace


def _aliases(arrays, *, allow_partial=False, mutable=None):
    groups: list[list[int]]=[]
    writes=set(range(len(arrays))) if mutable is None and not allow_partial else set(mutable or ())
    for array in arrays:
        if type(array) is not np.ndarray or array.dtype not in (np.dtype('float32'),np.dtype('float64'),np.dtype('int64')) or not array.size or not array.ndim:
            raise ValueError('native source state requires nonempty fp32/fp64 arrays')
    def identity(array):
        return (array.__array_interface__['data'][0],array.shape,array.strides,array.dtype)
    if any(arrays[i].dtype==np.dtype('int64') for i in writes):
        raise ValueError('mutable source state requires floating tensors')
    written={identity(arrays[i]) for i in writes}
    writes.update(i for i,a in enumerate(arrays) if identity(a) in written)
    for index,array in enumerate(arrays):
        # A snapshot does not preserve overlapping logical elements. Reject
        # broadcast and self-overlapping views before accepting state effects.
        stride_extent=array.itemsize
        for stride,extent in sorted(zip(map(abs,array.strides),array.shape,strict=True)):
            if index in writes and extent>1 and stride<stride_extent:
                raise ValueError('self-overlapping state views are unsupported')
            stride_extent+=stride*(extent-1)
        group=None
        for prior,other in enumerate(arrays[:index]):
            if np.shares_memory(array,other,max_work=1000):
                if (array.__array_interface__['data'][0]!=other.__array_interface__['data'][0]
                        or array.shape!=other.shape or array.strides!=other.strides or array.dtype!=other.dtype):
                    if index not in writes and prior not in writes:continue
                    raise ValueError('partial or differently shaped state aliases are unsupported')
                group=next(row for row in groups if prior in row)
                break
        if group is None:groups.append([index])
        else:group.append(index)
    return groups



def _state_layout(arrays, mutable):
    """Project injective bounded views onto an explicit contiguous SSA root."""
    aliases=_aliases(arrays,allow_partial=True)
    for i in mutable:_aliases((arrays[i],))
    components: list[list[int]]=[]
    for i,array in enumerate(arrays):
        connected=[g for g in components if any(np.shares_memory(array,arrays[j],max_work=1000) for j in g)]
        merged=[i]
        for g in connected:
            merged.extend(g)
            components.remove(g)
        components.append(sorted(merged))
    groups=[]
    views: list[tuple]=[]
    for members in sorted(components):
        targets=[i for i in mutable if i in members]
        if not targets:continue
        exact=all(arrays[i].shape==arrays[targets[0]].shape and arrays[i].strides==arrays[targets[0]].strides and arrays[i].dtype==arrays[targets[0]].dtype and arrays[i].ctypes.data==arrays[targets[0]].ctypes.data for i in members)
        root=targets[0]
        if not exact:
            if any(a.ndim!=arrays[root].ndim or any(stride%a.itemsize for stride in a.strides) or a.dtype!=arrays[root].dtype for a in (arrays[i] for i in members)):
                raise ValueError('partial writable aliases require same-rank element-aligned views')
            def bounds(a):
                low=sum(min(0,(n-1)*stride) for n,stride in zip(a.shape,a.strides,strict=True))
                high=sum(max(0,(n-1)*stride) for n,stride in zip(a.shape,a.strides,strict=True))
                return a.ctypes.data+low,a.ctypes.data+high+a.itemsize
            candidates=[i for i in members if arrays[i].flags.c_contiguous and all(arrays[i].ctypes.data<=bounds(arrays[j])[0] and bounds(arrays[j])[1]<=arrays[i].ctypes.data+arrays[i].nbytes for j in members)]
            if not candidates:raise ValueError('partial writable aliases require an explicit containing input')
            root=candidates[0]
            for i in members:
                offset=arrays[i].ctypes.data-arrays[root].ctypes.data
                if offset%arrays[root].itemsize:raise ValueError('partial writable aliases require element-aligned offsets')
                if arrays[i].ndim==1 and arrays[i].strides[0]>0:
                    views.append((i,offset//arrays[root].itemsize,arrays[i].size,arrays[i].strides[0]//arrays[i].itemsize))
                else:
                    start=tuple(int(v) for v in np.unravel_index(offset//arrays[root].itemsize,arrays[root].shape))
                    steps=tuple(stride//root_stride for stride,root_stride in zip(arrays[i].strides,arrays[root].strides,strict=True))
                    if all(stride>0 and stride%root_stride==0 for stride,root_stride in zip(arrays[i].strides,arrays[root].strides,strict=True)) and all(o+(n-1)*st<r for o,n,st,r in zip(start,arrays[i].shape,steps,arrays[root].shape,strict=True)):
                        views.append((i,'rect',start,tuple(arrays[i].shape),steps))
                        continue
                    if arrays[i].size>256:raise ValueError('mapped state views require at most 256 elements')
                    coordinates=[]
                    for index in np.ndindex(arrays[i].shape):
                        linear=(offset+sum(v*stride for v,stride in zip(index,arrays[i].strides,strict=True)))//arrays[i].itemsize
                        coordinates.append(tuple(int(v) for v in np.unravel_index(linear,arrays[root].shape)))
                    views.append((i,tuple(coordinates),tuple(arrays[i].shape)))
        if any(not arrays[i].flags.writeable for i in members):raise ValueError('declared state alias is read-only')
        groups.append((root,*(i for i in members if i!=root)))
    return aliases,tuple(groups),tuple(views)


def _object_storage(value, fields):
    """Project ordinary instance storage without running Python accessors."""
    from types import SimpleNamespace, GetSetDescriptorType
    cls=type(value)
    if cls is dict:return value
    if cls is SimpleNamespace:return vars(value)
    import inspect
    if type(cls) is not type or inspect.getattr_static(cls,'__getattribute__') is not object.__getattribute__ or any('__getattr__' in base.__dict__ for base in cls.__mro__):
        raise ValueError('source custom accessors are unsupported')
    for field in (*fields,'__dict__'):
        member=inspect.getattr_static(cls,field,None)
        if member is not None and hasattr(type(member),'__get__'):
            if field!='__dict__' or type(member) is not GetSetDescriptorType:
                raise ValueError('source custom accessors are unsupported')
    try:data=object.__getattribute__(value,'__dict__')
    except AttributeError as exc:raise ValueError('source objects require plain instance dictionary storage') from exc
    if type(data) is not dict:raise ValueError('source objects require plain instance dictionary storage')
    return data


def _output(type_name):
    match=re.fullmatch(r'tensor<((?:[0-9]+x)*)f(32|64)>',type_name)
    if match is None:raise ValueError('state outputs require static floating tensors')
    shape=tuple(int(n) for n in match[1].split('x') if n)
    return np.empty(shape,dtype='float'+match[2])


def bind_source_exception_types(contract, supplied=None):
    """Resolve declared class names only through explicitly owned host bindings."""
    allowed={'Exception','ValueError','RuntimeError','AssertionError','TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError'}
    bindings=dict(supplied or {})
    if any(type(name) is not str or name in allowed or not isinstance(kind,type) or not issubclass(kind,Exception) for name,kind in bindings.items()):
        raise ValueError('invalid explicit source exception class binding')
    if 'exception_heap' in contract:
        from .source_exception_heap import validate_heap
        validate_heap(contract['exception_heap'],bindings)
        return bindings
    def visit(edge,depth=0):
        if depth>32 or not isinstance(edge,(list,tuple)) or len(edge)<2:
            raise ValueError('invalid source exception graph')
        name=edge[0]
        if type(name) is not str or name not in allowed and name not in bindings:
            raise ValueError('source exception class requires an explicit host binding')
        if len(edge)>2 and edge[2] is not None:
            cause,args=edge[2]
            if cause=='edge':visit(args,depth+1)
            # The table also retains pre-elaboration declaration entries.
            # A named cause is resolved to an edge by the source producer;
            # "binding" denotes a local reference, never a host class name.
            elif cause not in ('suppress','binding'):visit((cause,args),depth+1)
        if len(edge)>4 and edge[4] is not None:visit(edge[4],depth+1)
    for edge in contract.get('error_table',()):visit(edge)
    return bindings


def decode_source_exception(contract,outputs,*,exception_types=None):
    """Decode checked completion data; logical source locations are not Python frames."""
    import builtins
    sites=contract.get('error_payload_sites',())
    if not isinstance(sites,(list,tuple)) or len(sites)>32 or any(type(site) is not str for site in sites) or len(set(sites))!=len(sites):raise RuntimeError('invalid source exception payload slots')
    code=float(outputs[-((2 if contract.get('error_dynamic') else 1)+len(sites))].item())
    if code==0:return None
    if 'exception_heap' in contract:
        from .source_exception_heap import decode_heap
        if not code.is_integer():raise RuntimeError('invalid native source exception status')
        bindings=bind_source_exception_types(contract,exception_types)
        return decode_heap(contract['exception_heap'],int(code),contract,outputs,bindings)
    table=contract.get('error_table',())
    if not code.is_integer() or not 1<=code<=len(table):raise RuntimeError('invalid native source exception status')
    allowed={'Exception','ValueError','RuntimeError','AssertionError','TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError'}
    bindings=dict(exception_types or {})
    if any(type(name) is not str or name in allowed or not isinstance(kind,type) or not issubclass(kind,Exception) for name,kind in bindings.items()):
        raise ValueError('invalid explicit source exception class binding')
    # Validate the whole selected graph before running any user constructor.
    # A valid root must not execute host effects before a malformed child or
    # location is rejected. This is the bounded wire graph, not a Python heap.
    pending=[(table[int(code)-1],0)]
    nodes=0
    while pending:
        edge,depth=pending.pop()
        nodes+=1
        if depth>32 or nodes>4096 or not isinstance(edge,(list,tuple)) or not 2<=len(edge)<=6:
            raise RuntimeError('invalid native source exception graph')
        kind,payload,*extra=edge
        if type(kind) is not str or kind not in allowed and kind not in bindings or not isinstance(payload,(list,tuple)) or any(type(v) is not str for v in payload):
            raise RuntimeError('invalid native source exception payload')
        if len(payload)==2 and payload[0]=='@tensor':
            if not contract.get('error_dynamic') or sites and payload[1] not in sites:
                raise RuntimeError('missing source exception payload slot')
        if extra and extra[0] is not None:
            cause=extra[0]
            if not isinstance(cause,(list,tuple)) or len(cause)!=2 or type(cause[0]) is not str:
                raise RuntimeError('invalid native source exception cause')
            if cause[0]=='edge':pending.append((cause[1],depth+1))
            elif cause[0]!='suppress':pending.append((cause,depth+1))
        if len(extra)>1:
            location=extra[1]
            if not isinstance(location,(list,tuple)) or len(location)!=2 or type(location[0]) is not str or type(location[1]) is not int or location[1]<1:
                raise RuntimeError('invalid source exception location')
        if len(extra)>2 and extra[2] is not None:pending.append((extra[2],depth+1))
        if len(extra)>3 and type(extra[3]) is not str:
            raise RuntimeError('invalid source exception occurrence')
    memo: dict[str,Exception]={}
    def build(edge,depth=0):
        if depth>32 or not isinstance(edge,(list,tuple)) or len(edge)<2:raise RuntimeError('invalid native source exception graph')
        key=json.dumps(edge,sort_keys=True)
        if key in memo:return memo[key]
        kind,payload,*extra=edge
        if kind not in allowed and kind not in bindings or not isinstance(payload,(list,tuple)) or any(type(v) is not str for v in payload):raise RuntimeError('invalid native source exception payload')
        args=payload
        if len(payload)==2 and payload[0]=='@tensor' and contract.get('error_dynamic'):
            if sites and payload[1] not in sites:raise RuntimeError('missing source exception payload slot')
            index=sites.index(payload[1])-len(sites) if sites else -1
            args=(outputs[index].copy(),)
        error=(bindings[kind] if kind in bindings else getattr(builtins,kind))(*args);memo[key]=error
        if extra and extra[0] is not None:
            cause_kind,cause_args=extra[0]
            error.__suppress_context__=True
            if cause_kind=='edge':error.__cause__=build(cause_args,depth+1)
            elif cause_kind!='suppress':error.__cause__=build((cause_kind,cause_args),depth+1)
        if len(extra)>1:
            file,line=extra[1]
            if type(file) is not str or type(line) is not int or line<1:raise RuntimeError('invalid source exception location')
            error.add_note(f'Native source raise at {file}:{line}; no Python frame executed there')
        if len(extra)>2 and extra[2] is not None:error.__context__=build(extra[2],depth+1)
        return error
    return build(table[int(code)-1])


class NativeSourceStateProgram:
    def __init__(self,native_ir,*,exception_types=None):
        from tessera import _jit_boundary as jit
        self._exception_types=dict(exception_types or {})
        self._ir=native_ir
        self._digest=hashlib.sha256(native_ir.encode()).hexdigest()
        self._lock=threading.RLock()
        self._result_abi=None
        contract=json.loads(_attribute(native_ir,'tessera.source_state'))
        if any('?' in ty for ty in contract['outputs']):
            from .scheduled_matmul import find_tessera_opt
            from .native_gpu_storage import _run
            from pathlib import Path
            import math
            compiler=find_tessera_opt()
            if compiler is None:raise ValueError('dynamic CPU results require the native compiler')
            capacity=max(math.prod(arg['shape']) for arg in contract['arguments'])
            if not 1<=capacity<=1024:raise ValueError('dynamic CPU result capacity must be at most 1024 elements')
            lowered=_run(compiler,'--tessera-to-linalg',source=native_ir)
            # Use the configured LLVM companion; it must match the compiler.
            import os
            llvm=Path(os.environ.get('TESSERA_LLVM_BIN','/usr/lib/llvm-23/bin'))
            buffered=_run(llvm/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
                '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map copy-before-write=true',
                '--convert-linalg-to-loops','--canonicalize',source=lowered)
            native_ir=_run(compiler,'--tessera-native-tape-to-gpu=host-results-only=true status-buffer=true public-result-capacity='+str(capacity),source=buffered)
            self._result_abi=json.loads(_attribute(native_ir,'tessera.native_result_abi'))
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
            aliases,groups,views=_state_layout(arrays,tuple(contract.get('mutable',tuple(g[0] for g in contract['groups']))))
            if aliases!=contract['aliases'] or list(map(list,groups))!=contract['groups'] or json.loads(json.dumps(views))!=[v+[1] if len(v)==3 and isinstance(v[1],int) else v for v in contract.get('state_views',[])]:
                raise ValueError('state alias topology disagrees with compiled IR')
            for array,spec in zip(arrays,contract['arguments'],strict=True):
                if list(array.shape)!=spec['shape'] or array.dtype!=np.dtype({'f32':'float32','f64':'float64','i64':'int64'}[spec['dtype']]):
                    raise ValueError('state shape or dtype disagrees with compiled IR')
            for group in contract['groups']:
                if any(not arrays[index].flags.writeable for index in group):
                    raise ValueError('declared state alias is read-only')
            snapshots=[array.copy() for array in arrays]
            if self._result_abi is None:
                outputs=[_output(type_name) for type_name in contract['outputs']]
                jit.invoke(self._handle,'source_program',snapshots,outputs)
            else:
                names={'f32':'float32','f64':'float64','i64':'int64','i8':'int8'}
                physical=[np.empty(row['shape'],names[row['storage']]) for row in self._result_abi['arguments'] if row['writable']]
                status=np.full(1,-1,np.int64)
                jit.invoke(self._handle,'source_program',[*snapshots,*physical,status],[])
                if status.item()!=0:raise RuntimeError('dynamic CPU result guard failed; output is unavailable')
                arguments=[*snapshots,*physical]
                outputs=[]
                for row in self._result_abi['results']:
                    shape=tuple(int(n) for n in arguments[row['shape']][:row.get('rank',1)])
                    if any(n<0 or n>row['capacity'] for n in shape) or np.prod(shape)>row['capacity']:
                        raise RuntimeError('dynamic CPU result shape exceeds capacity')
                    outputs.append(arguments[row['data']][:int(np.prod(shape))].reshape(shape).copy())
            count=contract['result_count']
            for group,result in zip(contract['groups'],outputs[count:count+len(contract['groups'])],strict=True):
                np.copyto(arrays[group[0]],result,casting='no')
            if contract.get('error_specs'):
                error=decode_source_exception(contract,outputs,exception_types=self._exception_types)
                if error is not None:raise error
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
    if not isinstance(mutable,tuple) or any(type(i) is not int or not 0<=i<len(arrays) for i in mutable) or len(set(mutable))!=len(mutable):
        raise ValueError('mutable must name distinct input argument indices')
    aliases,groups,views=_state_layout(arrays,mutable)
    traced=trace(fn,*arrays,source_control_flow=True,max_steps=max_steps,source_state_groups=groups,source_state_views=views,source_error_specs=error_specs,source_object_fields=object_fields)
    native=to_native_source_ir(traced)
    contract=json.loads(_attribute(native,'tessera.source_state'))
    old=json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    contract['aliases']=aliases
    contract['mutable']=mutable
    new=json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    if native.count(old)!=1:raise ValueError('source state contract is not unique')
    return NativeSourceStateProgram(native.replace(old,new,1),exception_types=traced.source_exception_types)


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

    def _inputs(self,args,kwargs):
        bound=self.signature.bind(*args,**kwargs)
        bound.apply_defaults()
        arrays_list: list[np.ndarray]=[]
        mutable: list[int]=[]
        fields=dict(self.object_fields)
        for index,value in enumerate(bound.arguments.values()):
            if index in fields:
                data=_object_storage(value,fields[index])
                children=[data[field] for field in fields[index]]
            else:
                if type(value) is int:
                    if not -(1<<63)<=value<(1<<63):raise ValueError('source integer argument exceeds int64')
                    value=np.array([value],np.int64)
                children=[value]
            if index in self.mutable:mutable.extend(range(len(arrays_list),len(arrays_list)+len(children)))
            arrays_list.extend(children)
        arrays=tuple(arrays_list)
        return arrays,tuple(mutable)

    def __call__(self,*args,**kwargs):
        with self._lock:
            if self._closed:raise ValueError('native source JIT is closed')
            arrays,mutable=self._inputs(args,kwargs)
            aliases,groups,views=_state_layout(arrays,tuple(mutable))
            key=(tuple((a.shape,a.dtype.str) for a in arrays),tuple(map(tuple,aliases)),groups,views)
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
        """Differentiate functional results and declared next-state outputs without copyback."""
        from tessera import _jit_boundary as jit
        from .scheduled_matmul import find_tessera_opt, run_tessera_opt
        with self._lock:
            if self._closed:raise ValueError('native source JIT is closed')
            arrays,mutable=self._inputs(arrays,{})
            aliases,groups,views=_state_layout(arrays,mutable)
            traced=trace(self.fn,*arrays,source_control_flow=True,max_steps=self.max_steps,source_state_groups=groups,source_state_views=views,source_object_fields=self.object_fields,source_error_specs=self.error_specs)
            count=len(traced.outputs)
            public_count=len(self.error_specs)+len(groups) if self.error_specs else count
            if not isinstance(cotangents,tuple) or len(cotangents)!=public_count:
                raise ValueError('source VJP requires one cotangent per result')
            source=to_native_source_ir(traced,autodiff='reverse')
            state_contract=json.loads(_attribute(source,'tessera.source_state'))
            compiler=find_tessera_opt()
            if compiler is None:raise ValueError('source VJP requires the native compiler')
            handles=[]
            contracts=[]
            def allocate(types):
                import math
                names={'f32':'float32','f64':'float64','i64':'int64','i8':'int8'}
                values=[]
                for kind in types:
                    match=re.fullmatch(r'tensor<((?:[1-9][0-9]*x)*)(f32|f64|i8|i64)>',kind)
                    if match is None:raise ValueError('source CPU product requires static tensor slots')
                    shape=tuple(int(n) for n in match[1].split('x') if n)
                    if math.prod(shape)>16777216:raise ValueError('source CPU product slot exceeds 16M elements')
                    values.append(np.zeros(shape,dtype=names[match[2]]))
                return values
            try:
                for role in ('forward','backward'):
                    exported=run_tessera_opt(compiler,source,'--tessera-autodiff-paired=box-product-scalars=true normalize-data-while=true export-product='+role)
                    contracts.append(json.loads(_attribute(exported,'tessera.autodiff.product_abi')))
                    handles.append(jit.compile_module(exported))
                outputs=allocate(contracts[0]['results'])
                for cotangent,primal in zip(cotangents,outputs[:public_count],strict=True):
                    if type(cotangent) is not np.ndarray or cotangent.shape!=primal.shape or cotangent.dtype!=primal.dtype:
                        raise ValueError('source VJP cotangent shape or dtype disagrees')
                snapshots=[a.copy() for a in arrays]
                jit.invoke(handles[0],contracts[0]['entry'],snapshots,outputs)
                if self.error_specs:
                    error=decode_source_exception(state_contract,outputs[:count],exception_types=traced.source_exception_types)
                    if error is not None:raise error
                seeds=[c.copy() for c in cotangents]+[np.zeros_like(o) for o in outputs[public_count:count]]
                derivatives=allocate(contracts[1]['results'])
                jit.invoke(handles[1],contracts[1]['entry'],[*snapshots,*seeds,*outputs[count:]],derivatives)
                return tuple(outputs[:public_count]),tuple(derivatives)
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
    if contract.get('schema')!=1:
        raise ValueError('GPU source state requires schema one')
    if len(contract.get('arguments',()))!=1 or contract.get('groups')!=[[0]]:
        raise ValueError('GPU source state requires one declared state input until multi-input alias projection is available')
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    native=_run(compiler,'--tessera-to-linalg',source=native_ir)
    buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops copy-before-write=true',
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
        self._pending: list | None=None
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
                if owner._pending is not None:raise ValueError('GPU state update is pending')
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
            if self._pending is not None:raise ValueError('GPU state update is pending')
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

    def submit_step(self,stream):
        """Submit computation; poll_step validates it before asynchronous copyback."""
        with self._lock:
            self._ready()
            if self._readers:raise ValueError('GPU state mutation requires all reader scopes to end')
            if self._pending is not None:raise ValueError('GPU state update is pending')
            result=self._program.submit(stream,self._frame.results[-1])
            self._pending=[result,stream,None]

    def _event_function(self,cu,hip,args):
        import ctypes as ct
        native=self._frame.binding._bound
        fn=getattr(native._driver,cu if self._program.package.backend=='nvidia' else hip)
        fn.argtypes,fn.restype=args,ct.c_int
        return fn

    def poll_step(self):
        """Return the owned result after copyback completes, or None while pending."""
        import ctypes as ct
        with self._lock:
            self._ready()
            if self._pending is None:raise ValueError('GPU state has no pending update')
            result,stream,event=self._pending
            try:
                if event is None:
                    if not result.poll():return None
                    src=result.results[-1].__cuda_array_interface__
                    dest=self._frame.results[-1].__cuda_array_interface__
                    if (src['shape'],src['typestr'])!=(dest['shape'],dest['typestr']):raise ValueError('GPU next-state ABI changed')
                    event=ct.c_void_p()
                    create=self._event_function('cuEventCreate','hipEventCreateWithFlags',[ct.POINTER(ct.c_void_p),ct.c_uint])
                    self._frame.check(create(ct.byref(event),2))
                    self._pending[2]=event
                    copy=self._event_function('cuMemcpyDtoDAsync_v2','hipMemcpyDtoDAsync',[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p])
                    self._frame.check(copy(dest['data'][0],src['data'][0],int(np.prod(src['shape']))*np.dtype(src['typestr']).itemsize,stream))
                    record=self._event_function('cuEventRecord','hipEventRecord',[ct.c_void_p,ct.c_void_p])
                    self._frame.check(record(event,stream))
                query=self._event_function('cuEventQuery','hipEventQuery',[ct.c_void_p])
                code=query(event)
                if code==600:return None
                self._frame.check(code)
                destroy=self._event_function('cuEventDestroy_v2','hipEventDestroy',[ct.c_void_p])
                self._frame.check(destroy(event))
                self._pending=None
                return result
            except BaseException:
                self._poisoned=True
                raise

    def close(self):
        with self._lock:
            if self._closed:return
            if self._readers:raise ValueError('GPU state has active readers')
            if self._pending is not None:
                import ctypes as ct
                result,_,event=self._pending
                self._frame.check(self._frame.sync())
                if event is not None:
                    destroy=self._event_function('cuEventDestroy_v2','hipEventDestroy',[ct.c_void_p])
                    self._frame.check(destroy(event))
                    self._pending[2]=None
                result.close()
                self._pending=None
            for frame in self._quarantine:frame.close()
            self._quarantine.clear()
            self._frame.close()
            self._closed=True

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()
