"""Checked binding of compiler-owned gate/up and down-projection SSA plans."""
from __future__ import annotations
import base64
import ctypes as C
from dataclasses import dataclass,replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from .quantized_binding_cache import memoized_validation,memoized_binding
from .native_scaled_program import PreparedScaledProgram,_BoundScaledABI,_Buffer,_StorageSpec,_status
from .rocm_nvfp4_expansion_program import _MultiOutputStep
from .rocm_quantized_product_program import NativeQuantizedProductProgram,PreparedQuantizedProductProgram,_frame_shape,package_quantized_product_graph

_SCHEMA='tessera.native.quantized_product.v4'
_BOUNDED_SCHEMA='tessera.native.quantized_product.v5'
_PACKAGE='tessera.rocm.quantized_product_package.v4'
_BOUNDED_PACKAGE='tessera.rocm.quantized_product_package.v5'
_PLAN_KEYS={'schema','source_graph_ir','member_graphs','buffers','steps','output','subprograms'}
_BOUNDED_KEYS=_PLAN_KEYS|{'active_m','m_bound','original_graph_ir','row_frame'}
#: Original activation, gate quantizer pair, gate/up parent, activation, down
#: quantizer pair and returned output vary with M; both weight expansions and
#: their products' B operands never do.
_ROW_SLOTS=(7,8,11,12,13,14,17)
_ROW_STEPS=[0,2,3,4,6]


def _components(plan):
    if (not isinstance(plan,dict) or plan.get('schema') not in (_SCHEMA,_BOUNDED_SCHEMA)
            or set(plan)!=(_BOUNDED_KEYS if plan['schema']==_BOUNDED_SCHEMA else _PLAN_KEYS)
            or not isinstance(plan['subprograms'],list) or len(plan['subprograms'])!=2
            or not all(isinstance(v,str) for v in plan['subprograms'])):
        raise ValueError('native quantized MLP component encoding differs')
    try:
        components=tuple(json.loads(base64.b64decode(v,validate=True)) for v in plan['subprograms'])
    except (ValueError,TypeError) as exc:
        raise ValueError('native quantized MLP component encoding differs') from exc
    if ([c.get('schema') if isinstance(c,dict) else None for c in components]
            !=['tessera.native.quantized_product.v3','tessera.native.quantized_product.v2']):
        raise ValueError('native quantized MLP component schemas differ')
    return components


def _frame(plan):
    gate,down=_components(plan)
    gr,_=_frame_shape(gate);dr,shape=_frame_shape(down)
    if (not isinstance(plan['source_graph_ir'],str) or not plan['source_graph_ir']
            or type(plan['output']) is not int or plan['output']!=17
            or not isinstance(plan['steps'],list) or len(plan['steps'])!=7
            or not isinstance(plan['buffers'],list) or len(plan['buffers'])!=18
            or plan['member_graphs']!=gate['member_graphs']+down['member_graphs']):
        raise ValueError('native quantized MLP frame differs')
    steps=plan['steps']
    for i,step in enumerate(steps):
        if (not isinstance(step,dict) or set(step)!={'step','operation','member','inputs','outputs'}
                or type(step['step']) is not int or step['step']!=i
                or not isinstance(step['member'],str) or not step['member']
                or not isinstance(step['inputs'],list) or not isinstance(step['outputs'],list)
                or any(type(v) is not int for v in step['inputs']+step['outputs'])):
            raise ValueError('native quantized MLP step differs')
    if len(steps[0]['inputs'])!=1 or len(steps[1]['inputs'])!=3 or len(steps[5]['inputs'])!=3:
        raise ValueError('native quantized MLP root roles differ')
    roles=steps[0]['inputs']+steps[1]['inputs']+steps[5]['inputs']
    if sorted(roles)!=list(range(7)) or len({s['member'] for s in steps})!=7:
        raise ValueError('native quantized MLP original arguments differ')
    gm=dict(zip(gr,roles[:4],strict=True));gm.update({i:i+3 for i in range(4,10)})
    dm=dict(zip(dr,[12,*roles[4:]],strict=True));dm.update({i:i+9 for i in range(4,9)})
    rows: list[dict[str,Any] | None]=[None]*18
    for component,mapping,offset in ((gate,gm,0),(down,dm,4)):
        for local in component['buffers']:
            slot=mapping[local['id']]
            row=dict(local,id=slot,
                first_write=local['first_write']+offset if local['first_write']>=0 else -1,
                last_read=local['last_read']+offset,
                ownership='readonly_input' if slot<7 else 'returned_output' if slot==17 else 'private_scratch')
            previous=rows[slot]
            if previous is not None:
                if any(previous[key]!=row[key] for key in ('shape','storage','bytes')):
                    raise ValueError('native quantized MLP activation/quantizer shared interface differs')
                row['first_write']=previous['first_write']
                row['last_read']=max(previous['last_read'],row['last_read'])
            rows[slot]=row
        for local in component['steps']:
            actual=steps[local['step']+offset]
            if (actual['operation']!=local['operation']
                    or actual['inputs']!=[mapping[i] for i in local['inputs']]
                    or actual['outputs']!=[mapping[i] for i in local['outputs']]):
                raise ValueError('native quantized MLP private edges differ')
    for actual,expected in zip(plan['buffers'],rows,strict=True):
        if (not isinstance(actual,dict) or actual!=expected
                or any(type(actual.get(k)) is not int for k in ('id','bytes','first_write','last_read'))
                or any(type(d) is not int for d in actual['shape'])):
            raise ValueError('native quantized MLP buffer storage/lifetime differs')
    if plan['schema']==_BOUNDED_SCHEMA:
        _row_frame(plan,roles)
    return roles,shape,(gm,dm)


def _row_frame(plan,roles):
    """Re-derive the exported v5 row frame; return (active, capacity, slots, strides)."""
    active,bound=plan['active_m'],plan['m_bound']
    if (type(active) is not int or type(bound) is not int or not 0<active<=bound
            or not isinstance(plan['original_graph_ir'],str) or not plan['original_graph_ir']):
        raise ValueError('native quantized MLP bounded rows differ')
    slots=sorted((roles[0],*_ROW_SLOTS))
    rows=plan['buffers']
    strides=[]
    for slot in slots:
        row=rows[slot]
        if row['shape'][0]!=bound or row['bytes']%bound:
            raise ValueError('native quantized MLP row slot capacity differs')
        strides.append(row['bytes']//bound)
    if plan['row_frame']!={'slots':slots,'row_strides_bytes':strides,'row_steps':_ROW_STEPS}:
        raise ValueError('native quantized MLP exported row frame differs')
    return active,bound,tuple(slots),tuple(strides)


@dataclass(frozen=True)
class NativeQuantizedMLPProgram:
    program_json: str
    graph_ir: str
    gate: NativeQuantizedProductProgram
    down: NativeQuantizedProductProgram
    binding_digest: str

    @property
    def native_stages(self):
        return self.gate.native_stages+self.down.native_stages

    @property
    def bounded(self):
        return json.loads(self.program_json)['schema']==_BOUNDED_SCHEMA

    @property
    def schedule_irs(self):
        return self.gate.schedule_irs+self.down.schedule_irs

    def _identity(self):
        return hashlib.sha256(json.dumps([self.program_json,self.graph_ir,
            self.gate.binding_digest,self.down.binding_digest],separators=(',',':')).encode()).hexdigest()

    @memoized_validation
    def validate(self):
        self.gate.validate();self.down.validate()
        plan=json.loads(self.program_json)
        _frame(plan)
        components=_components(plan)
        if (plan['source_graph_ir']!=self.graph_ir or self.binding_digest!=self._identity()
                or json.loads(self.gate.program_json)!=components[0]
                or json.loads(self.down.program_json)!=components[1]):
            raise ValueError('native quantized MLP member binding differs')
        # A bounded program launches both products below capacity, so both
        # images must guard arbitrary M; a static program must not claim it.
        bounded=plan['schema']==_BOUNDED_SCHEMA
        if self.gate.arbitrary_m is not bounded or self.down.arbitrary_m is not bounded:
            raise ValueError('native quantized MLP product row policy differs from its frame')

    def to_manifest(self):
        self.validate()
        bounded=self.bounded
        # A bounded record carries the v5 plan (row frame, capacity and the
        # original request) and both arbitrary-M products; the two schemas are
        # bound to each other on replay.
        return {'schema':_BOUNDED_PACKAGE if bounded else _PACKAGE,
                'program_json':self.program_json,'graph_ir':self.graph_ir,
                'gate':self.gate.to_manifest(_bounded_member=bounded),
                'down':self.down.to_manifest(_bounded_member=bounded),'binding_digest':self.binding_digest}

    @classmethod
    def from_manifest(cls,record):
        if (not isinstance(record,dict) or set(record)!={'schema','program_json','graph_ir','gate','down','binding_digest'}
                or record['schema'] not in (_PACKAGE,_BOUNDED_PACKAGE)
                or any(not isinstance(record[k],str) for k in ('program_json','graph_ir','binding_digest'))):
            raise ValueError('native quantized MLP portable record differs')
        bounded=record['schema']==_BOUNDED_PACKAGE
        try:
            plan_schema=json.loads(record['program_json']).get('schema')
        except (ValueError,AttributeError) as exc:
            raise ValueError('native quantized MLP portable record differs') from exc
        if plan_schema!=(_BOUNDED_SCHEMA if bounded else _SCHEMA):
            raise ValueError('native quantized MLP portable schema/frame differs')
        # Exact component schemas prevent recursive full-program manifests.
        if (not isinstance(record['gate'],dict) or not isinstance(record['down'],dict)
                or record['gate'].get('schema')!='tessera.rocm.quantized_product_package.v3'
                or record['down'].get('schema')!='tessera.rocm.quantized_product_package.v2'):
            raise ValueError('native quantized MLP portable component schemas differ')
        program=cls(record['program_json'],record['graph_ir'],
            NativeQuantizedProductProgram.from_manifest(record['gate'],_bounded_member=bounded),
            NativeQuantizedProductProgram.from_manifest(record['down'],_bounded_member=bounded),
            record['binding_digest'])
        # validate() re-derives the row frame and requires both products'
        # arbitrary-M policy to match the plan.
        program.validate()
        return program

    def prepare(self,inputs,*,runtime_library):
        return PreparedQuantizedMLPProgram(self,inputs,runtime_library=runtime_library)


def package_quantized_mlp_plan(plan,tool):
    from .rocm_quantized_product_program import _export
    _frame(plan)
    bounded=plan['schema']==_BOUNDED_SCHEMA
    # A bounded plan replays from the original request: the capacity source
    # alone carries no bound and would export the static v4 frame.
    if _export(tool,plan['original_graph_ir' if bounded else 'source_graph_ir'])!=plan:
        raise ValueError('native quantized MLP normalized source replay differs')
    gate,down=_components(plan)
    program=NativeQuantizedMLPProgram(json.dumps(plan,sort_keys=True),plan['source_graph_ir'],
        package_quantized_product_graph(gate['source_graph_ir'],arbitrary_m=bounded),
        package_quantized_product_graph(down['source_graph_ir'],arbitrary_m=bounded),'')
    program=replace(program,binding_digest=program._identity())
    program.validate()
    return program


class _RowStep(C.Structure):
    _fields_=[('step',C.c_uint32),('scalar',C.c_uint32),('scalar_per_row',C.c_int64),
              ('grid_axis',C.c_uint32),('grid_tile',C.c_uint32),('grid_per_row',C.c_int64)]


class _RowFrame(C.Structure):
    _fields_=[('capacity_m',C.c_int64),('slot_count',C.c_uint32),('step_count',C.c_uint32),
              ('slots',C.POINTER(C.c_uint32)),('row_bytes',C.POINTER(C.c_uint64)),
              ('steps',C.POINTER(_RowStep))]


def _row_step(index,step,capacity,macro_m):
    """Scale the one row-dependent launch field each member ABI exposes."""
    if step.scalar_count==3:
        # Scaled product: M scalar, macro-tile rows on grid y.
        if step.scalars[0]!=capacity or macro_m<=0:
            raise ValueError('native quantized MLP product M differs from its capacity')
        return _RowStep(index,0,1,1,macro_m,1)
    # Quantizer Groups / SiLU-mul count: a flat element count over 256 lanes.
    if step.scalar_count!=1 or step.scalars[0]%capacity or step.geometry[3]!=256:
        raise ValueError('native quantized MLP flat member count differs from its capacity')
    per_row=step.scalars[0]//capacity
    return _RowStep(index,0,per_row,0,256,per_row)


class PreparedQuantizedMLPProgram(PreparedScaledProgram):
    package: NativeQuantizedMLPProgram
    _prepare_symbol='tessera_rocm_program_prepare_multi_output'
    _step_type=_MultiOutputStep

    def __init__(self,package,inputs,*,runtime_library):
        from tessera import runtime as rt
        if rt._rocm_live_arch()!='gfx1201':
            raise RuntimeError('native quantized MLP requires owning gfx1201')
        plan=json.loads(package.program_json)
        self._bounded=plan['schema']==_BOUNDED_SCHEMA
        if self._bounded:
            roles,_,_=_frame(plan)
            self._active_m,self._capacity_m,self._row_slots,self._row_bytes=_row_frame(plan,roles)
            self._x_slot=roles[0]
            self._prepare_symbol='tessera_rocm_program_prepare_multi_output_row_frame'
        super().__init__(package,inputs,runtime_library=runtime_library)

    def _prepare_tail(self):
        if not self._bounded:
            return (),()
        if not hasattr(self,'_frame_struct'):
            steps=self._binding.steps
            stages=self.package.native_stages
            entries=[_row_step(i,steps[i],self._capacity_m,
                               stages[i].descriptor.provenance.get('macro_tile',(0,))[0])
                     for i in _ROW_STEPS]
            self._frame_arrays=((C.c_uint32*len(self._row_slots))(*self._row_slots),
                                (C.c_uint64*len(self._row_bytes))(*self._row_bytes),
                                (_RowStep*len(entries))(*entries))
            slots,strides,rows=self._frame_arrays
            self._frame_struct=_RowFrame(self._capacity_m,len(self._row_slots),len(entries),
                C.cast(slots,C.POINTER(C.c_uint32)),C.cast(strides,C.POINTER(C.c_uint64)),
                C.cast(rows,C.POINTER(_RowStep)))
        return (C.POINTER(_RowFrame),C.c_int64),(C.byref(self._frame_struct),self._active_m)

    def _storage(self,active_m):
        if not self._bounded:
            return self._binding.storage
        result=list(self._binding.storage)
        for slot,stride in zip(self._row_slots,self._row_bytes,strict=True):
            spec=result[slot]
            result[slot]=_StorageSpec((active_m,*spec.shape[1:]),spec.storage,stride*active_m,spec.numpy_dtype)
        return tuple(result)

    def _input_m(self,activation):
        shape=np.shape(activation)
        if len(shape)!=2 or not 0<shape[0]<=self._capacity_m:
            raise ValueError('native quantized MLP active M exceeds its capacity')
        return shape[0]

    @property
    def active_m(self):
        return self._active_m if self._bounded else self._binding.storage[17].shape[0]

    @memoized_binding
    def _bind_abi(self,package):
        package.validate()
        plan=json.loads(package.program_json);roles,_,maps=_frame(plan)
        binder=PreparedQuantizedProductProgram.__new__(PreparedQuantizedProductProgram)
        bindings=(binder._bind_abi(package.gate),binder._bind_abi(package.down))
        steps=(_MultiOutputStep*7)();storage: list[_StorageSpec | None]=[None]*18
        for binding,mapping,offset in zip(bindings,maps,(0,4),strict=True):
            for local,contract in enumerate(binding.storage):
                storage[mapping[local]]=contract
            for local,step in enumerate(binding.steps):
                steps[local+offset]=step
                copied=steps[local+offset]
                copied.inputs[:copied.input_count]=[mapping[i] for i in step.inputs[:step.input_count]]
                copied.outputs[:copied.output_count]=[mapping[i] for i in step.outputs[:step.output_count]]
        ownership={'readonly_input':0,'private_scratch':1,'returned_output':2}
        buffers=(_Buffer*18)(*[_Buffer(b['bytes'],math.prod(b['shape']),b['first_write'],b['last_read'],ownership[b['ownership']],0) for b in plan['buffers']])
        contracts=tuple(s for s in storage if s is not None)
        if len(contracts)!=18:
            raise ValueError('native quantized MLP incomplete storage binding')
        return _BoundScaledABI(buffers,steps,bindings[0].blobs+bindings[1].blobs,
            bindings[0].entries+bindings[1].entries,len(roles),contracts,(17,),())

    def _inputs(self,inputs):
        from .nvfp4_expansion import checked_expansion_inputs,nvfp4_expansion_policy
        from .rocm_nvfp4_expand_native import _contract
        roles,_,_=_frame(json.loads(self.package.program_json))
        if len(inputs)!=7:
            raise ValueError('native quantized MLP requires seven roots')
        for component,slots in ((self.package.gate,roles[1:4]),(self.package.down,roles[4:])):
            if component.expansion is None:
                raise ValueError('native quantized MLP requires both expansions')
            _,_,_,offsets,_,_=_contract(component.expansion.native.tile_ir,component.expansion.native.target_ir)
            checked_expansion_inputs(*(inputs[i] for i in slots),row_offsets=offsets,numeric_policy=nvfp4_expansion_policy(),_scale_bounds=self._nvfp4_scale_bounds())
        if not self._bounded:
            return super()._inputs(inputs)
        active_m=self._input_m(inputs[roles[0]])
        arrays=self._inputs_for_storage(inputs,self._storage(active_m))
        self._active_m=active_m
        return arrays

    def _update_rows(self,active_m,slots,arrays):
        update=self.lib.tessera_rocm_program_update_rows
        update.argtypes=[C.c_uint64,C.c_int64,C.c_uint32,C.POINTER(C.c_uint32),
                         C.POINTER(C.c_void_p),C.POINTER(C.c_uint64)]
        update.restype=C.c_int
        count=len(slots)
        _status(update(self.handle,active_m,count,(C.c_uint32*count)(*slots),
                       (C.c_void_p*count)(*[a.ctypes.data for a in arrays]),
                       (C.c_uint64*count)(*[a.nbytes for a in arrays])))
        self._active_m=active_m

    def update(self,inputs):
        if not self._bounded:
            return super().update(inputs)
        # Validates against the requested M before any native state changes.
        previous=self._active_m
        try:
            arrays,_,_=self._inputs(inputs)
            self._update_rows(self._active_m,tuple(range(7)),arrays)
        except Exception:
            self._active_m=previous
            raise

    def update_activations(self,activation):
        """Replace only the original activation; expanded weights stay resident."""
        from .native_scaled_program import pack_host_view
        if not self._bounded:
            raise ValueError('static native quantized MLP has no bounded row frame')
        active_m=self._input_m(activation)
        contract=self._storage(active_m)[self._x_slot]
        array=np.asarray(activation)
        if array.dtype!=contract.numpy_dtype or array.shape!=contract.shape or array.nbytes!=contract.bytes:
            raise ValueError('native quantized MLP activation storage/shape differs')
        self._update_rows(active_m,(self._x_slot,),(pack_host_view(self.lib,array),))

    def rows_info(self):
        if not self._bounded:
            raise ValueError('static native quantized MLP has no bounded row frame')
        info=self.lib.tessera_rocm_program_rows_info
        info.argtypes=[C.c_uint64,C.POINTER(C.c_int64),C.POINTER(C.c_int64),
                       C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
        info.restype=C.c_int
        capacity,active,size,count=C.c_int64(),C.c_int64(),C.c_uint64(),C.c_uint64()
        _status(info(self.handle,C.byref(capacity),C.byref(active),C.byref(size),C.byref(count)))
        return capacity.value,active.value,size.value,count.value

    def read(self,generation,*,batched=True):
        if not self._bounded:
            return super().read(generation,batched=batched)
        contract=self._storage(self._active_m)[17]
        output=np.empty(contract.shape,dtype=contract.numpy_dtype)
        _status(self.lib.tessera_rocm_program_read(
            self.handle,17,generation,C.c_void_p(output.ctypes.data),output.nbytes))
        return (output,)

    def invoke_range(self,first,count,repeats=1,*,timed=True):
        if (type(first) is not int or type(count) is not int or first<0 or count<=0
                or first+count>7 or type(repeats) is not int or not 1<=repeats<=1048576):
            raise ValueError('native quantized MLP invocation range differs')
        invoke=self.lib.tessera_rocm_program_invoke_range
        invoke.argtypes=[C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_float)]
        invoke.restype=C.c_int
        generation,elapsed=C.c_uint64(),C.c_float()
        _status(invoke(self.handle,first,count,repeats,C.byref(generation),C.byref(elapsed) if timed else None))
        return generation.value,float(elapsed.value) if timed else None
