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
#: v5 at capacity K plus the contraction frame (GFX1201-BOUNDED-QUANTIZED-K-20261028).
_K_BOUNDED_SCHEMA='tessera.native.quantized_product.v6'
_PACKAGE='tessera.rocm.quantized_product_package.v4'
_BOUNDED_PACKAGE='tessera.rocm.quantized_product_package.v5'
_K_BOUNDED_PACKAGE='tessera.rocm.quantized_product_package.v6'
_PLAN_KEYS={'schema','source_graph_ir','member_graphs','buffers','steps','output','subprograms'}
_BOUNDED_KEYS=_PLAN_KEYS|{'active_m','m_bound','original_graph_ir','row_frame'}
_K_BOUNDED_KEYS=_BOUNDED_KEYS|{'active_k','k_bound','k_frame'}
_SCHEMA_KEYS={_SCHEMA:_PLAN_KEYS,_BOUNDED_SCHEMA:_BOUNDED_KEYS,_K_BOUNDED_SCHEMA:_K_BOUNDED_KEYS}
_PACKAGES={_SCHEMA:_PACKAGE,_BOUNDED_SCHEMA:_BOUNDED_PACKAGE,_K_BOUNDED_SCHEMA:_K_BOUNDED_PACKAGE}
#: Original activation, gate quantizer pair, gate/up parent, activation, down
#: quantizer pair and returned output vary with M; both weight expansions and
#: their products' B operands never do.
_ROW_SLOTS=(7,8,11,12,13,14,17)
_ROW_STEPS=[0,2,3,4,6]
#: The gate contraction reaches the original activation, the gate quantizer
#: pair, both gate weight planes and their expansion; the gate product
#: contracts it away. (slot role, axis, K divisor); roles are original
#: argument positions resolved against the plan.
_K_STEPS=[0,1,2]
_K_ROLE_AXES=((0,1,1),(1,1,2),(2,1,16))       # activation, packed, block scales
_K_SLOT_AXES=((7,1,1),(8,1,16),(9,1,1),(10,0,16))


def _k_axes(roles):
    axes={roles[role]:(axis,divisor) for role,axis,divisor in _K_ROLE_AXES}
    axes.update({slot:(axis,divisor) for slot,axis,divisor in _K_SLOT_AXES})
    return axes


def _components(plan):
    if (not isinstance(plan,dict) or plan.get('schema') not in _SCHEMA_KEYS
            or set(plan)!=_SCHEMA_KEYS[plan['schema']]
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
    if plan['schema']!=_SCHEMA:
        _row_frame(plan,roles)
    if plan['schema']==_K_BOUNDED_SCHEMA:
        _k_frame(plan,roles)
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


def _k_frame(plan,roles):
    """Re-derive the exported v6 contraction frame; return (active, capacity, axes)."""
    active,bound=plan['active_k'],plan['k_bound']
    if (type(active) is not int or type(bound) is not int or not 0<active<=bound
            or active%16 or bound%16):
        raise ValueError('native quantized MLP bounded K differs')
    axes=_k_axes(roles)
    for slot,(axis,divisor) in axes.items():
        if plan['buffers'][slot]['shape'][axis]!=bound//divisor:
            raise ValueError('native quantized MLP contraction slot capacity differs')
    if plan['k_frame']!={'slots':sorted(axes),'k_steps':_K_STEPS}:
        raise ValueError('native quantized MLP exported contraction frame differs')
    return active,bound,axes


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
        return json.loads(self.program_json)['schema'] in (_BOUNDED_SCHEMA,_K_BOUNDED_SCHEMA)

    @property
    def k_bounded(self):
        return json.loads(self.program_json)['schema']==_K_BOUNDED_SCHEMA

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
        bounded=plan['schema']!=_SCHEMA
        if self.gate.arbitrary_m is not bounded or self.down.arbitrary_m is not bounded:
            raise ValueError('native quantized MLP product row policy differs from its frame')
        # Only the gate contracts the bounded K; the down projection never sees it.
        if self.gate.runtime_k is not (plan['schema']==_K_BOUNDED_SCHEMA) or self.down.runtime_k:
            raise ValueError('native quantized MLP contraction policy differs from its frame')

    def to_manifest(self):
        self.validate()
        bounded=self.bounded
        # A bounded record carries the v5 plan (row frame, capacity and the
        # original request) and both arbitrary-M products; the two schemas are
        # bound to each other on replay.
        return {'schema':_PACKAGES[json.loads(self.program_json)['schema']],
                'program_json':self.program_json,'graph_ir':self.graph_ir,
                'gate':self.gate.to_manifest(_bounded_member=bounded),
                'down':self.down.to_manifest(_bounded_member=bounded),'binding_digest':self.binding_digest}

    @classmethod
    def from_manifest(cls,record):
        if (not isinstance(record,dict) or set(record)!={'schema','program_json','graph_ir','gate','down','binding_digest'}
                or record['schema'] not in _PACKAGES.values()
                or any(not isinstance(record[k],str) for k in ('program_json','graph_ir','binding_digest'))):
            raise ValueError('native quantized MLP portable record differs')
        bounded=record['schema']!=_PACKAGE
        try:
            plan_schema=json.loads(record['program_json']).get('schema')
        except (ValueError,AttributeError) as exc:
            raise ValueError('native quantized MLP portable record differs') from exc
        if plan_schema not in _PACKAGES or _PACKAGES[plan_schema]!=record['schema']:
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
    bounded=plan['schema']!=_SCHEMA
    # A bounded plan replays from the original request: the capacity source
    # alone carries no bound and would export the static v4 frame.
    if _export(tool,plan['original_graph_ir' if bounded else 'source_graph_ir'])!=plan:
        raise ValueError('native quantized MLP normalized source replay differs')
    gate,down=_components(plan)
    program=NativeQuantizedMLPProgram(json.dumps(plan,sort_keys=True),plan['source_graph_ir'],
        package_quantized_product_graph(gate['source_graph_ir'],arbitrary_m=bounded,
                                        runtime_k=plan['schema']==_K_BOUNDED_SCHEMA),
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


class _ShapeStep(C.Structure):
    _fields_=[('step',C.c_uint32),('scalar',C.c_uint32),('scalar_coef',C.c_int64),
              ('powers',C.c_uint32),('grid_axis',C.c_uint32),('grid_tile',C.c_uint32),
              ('grid_coef',C.c_int64)]


class _ShapeFrame(C.Structure):
    _fields_=[('capacity_m',C.c_int64),('capacity_g',C.c_int64),('slot_count',C.c_uint32),
              ('step_count',C.c_uint32),('slots',C.POINTER(C.c_uint32)),
              ('unit_bytes',C.POINTER(C.c_uint64)),('powers',C.POINTER(C.c_uint32)),
              ('steps',C.POINTER(_ShapeStep))]


def _shape_steps(steps,capacity_m,capacity_g,macro_m,expansion_rows):
    """Every M- and G-dependent launch field of the K-bounded MLP.

    G is the K16 group count. The gate quantizer counts M*G groups; the
    runtime-K expansion counts G groups on grid x over its rows on grid y; the
    gate product takes M (grid y over macro-tile rows) and K = 16*G. The SiLU
    parent view, down quantizer and down product are the row frame's entries."""
    q,e,g=steps[0],steps[1],steps[2]
    if (q.scalar_count!=1 or q.scalars[0]!=capacity_m*capacity_g or q.geometry[3]!=256
            or e.scalar_count!=1 or e.scalars[0]!=capacity_g or e.geometry[3]!=256
            or e.geometry[1]!=expansion_rows
            or g.scalar_count!=3 or g.scalars[0]!=capacity_m or g.scalars[2]!=16*capacity_g
            or macro_m<=0):
        raise ValueError('native quantized MLP contraction launch fields differ from their capacity')
    return [_ShapeStep(0,0,1,3,0,256,1),_ShapeStep(1,0,1,2,0,256,1),
            _ShapeStep(2,0,1,1,1,macro_m,1),_ShapeStep(2,2,16,2,3,0,0)]


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
        self._bounded=plan['schema']!=_SCHEMA
        self._k_bounded=plan['schema']==_K_BOUNDED_SCHEMA
        if self._bounded:
            roles,_,_=_frame(plan)
            self._active_m,self._capacity_m,self._row_slots,self._row_bytes=_row_frame(plan,roles)
            self._x_slot=roles[0]
            self._roles=tuple(roles)
            self._prepare_symbol='tessera_rocm_program_prepare_multi_output_row_frame'
        if self._k_bounded:
            self._active_k,self._capacity_k,self._k_axes=_k_frame(plan,roles)
            self._prepare_symbol='tessera_rocm_program_prepare_multi_output_shape_frame'
        super().__init__(package,inputs,runtime_library=runtime_library)

    def _prepare_tail(self):
        if not self._bounded:
            return (),()
        if self._k_bounded:
            return self._shape_tail()
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

    def _shape_tail(self):
        if not hasattr(self,'_shape_struct'):
            capacity_g=self._capacity_k//16
            rows=set(self._row_slots)
            slots=sorted(rows|set(self._k_axes))
            storage=self._binding.storage
            powers=[(1 if slot in rows else 0)|(2 if slot in self._k_axes else 0) for slot in slots]
            units=[]
            for slot,power in zip(slots,powers,strict=True):
                scale=(self._capacity_m if power&1 else 1)*(capacity_g if power&2 else 1)
                if storage[slot].bytes%scale:
                    raise ValueError('native quantized MLP frame slot is not a capacity monomial')
                units.append(storage[slot].bytes//scale)
            stages=self.package.native_stages
            entries=_shape_steps(self._binding.steps,self._capacity_m,capacity_g,
                stages[2].descriptor.provenance.get('macro_tile',(0,))[0],storage[9].shape[0])
            for i in _ROW_STEPS:
                if i in (0,2):
                    continue
                # The tail members are the v5 row entries: power M only.
                row=_row_step(i,self._binding.steps[i],self._capacity_m,
                              stages[i].descriptor.provenance.get('macro_tile',(0,))[0])
                entries.append(_ShapeStep(row.step,row.scalar,row.scalar_per_row,1,
                                          row.grid_axis,row.grid_tile,row.grid_per_row))
            self._shape_arrays=((C.c_uint32*len(slots))(*slots),(C.c_uint64*len(units))(*units),
                                (C.c_uint32*len(powers))(*powers),(_ShapeStep*len(entries))(*entries))
            a,b,c,d=self._shape_arrays
            self._shape_struct=_ShapeFrame(self._capacity_m,capacity_g,len(slots),len(entries),
                C.cast(a,C.POINTER(C.c_uint32)),C.cast(b,C.POINTER(C.c_uint64)),
                C.cast(c,C.POINTER(C.c_uint32)),C.cast(d,C.POINTER(_ShapeStep)))
        return ((C.POINTER(_ShapeFrame),C.c_int64,C.c_int64),
                (C.byref(self._shape_struct),self._active_m,self._active_k//16))

    def _storage(self,active_m,active_k=None):
        if not self._bounded:
            return self._binding.storage
        result=list(self._binding.storage)
        for slot,stride in zip(self._row_slots,self._row_bytes,strict=True):
            spec=result[slot]
            result[slot]=_StorageSpec((active_m,*spec.shape[1:]),spec.storage,stride*active_m,spec.numpy_dtype)
        if self._k_bounded:
            active_k=self._active_k if active_k is None else active_k
            for slot,(axis,divisor) in self._k_axes.items():
                spec=result[slot]
                shape=list(spec.shape)
                shape[axis]=active_k//divisor
                item=self._binding.storage[slot].bytes//math.prod(self._binding.storage[slot].shape)
                result[slot]=_StorageSpec(tuple(shape),spec.storage,math.prod(shape)*item,spec.numpy_dtype)
        return tuple(result)

    def _input_m(self,activation):
        shape=np.shape(activation)
        if len(shape)!=2 or not 0<shape[0]<=self._capacity_m:
            raise ValueError('native quantized MLP active M exceeds its capacity')
        if self._k_bounded:
            if not (0<shape[1]<=self._capacity_k and shape[1]%16==0):
                raise ValueError('native quantized MLP active K exceeds its K16 capacity')
        return shape[0]

    def _input_k(self,activation):
        return np.shape(activation)[1] if self._k_bounded else None

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
        roles,_,_=_frame(json.loads(self.package.program_json))
        if len(inputs)!=7:
            raise ValueError('native quantized MLP requires seven roots')
        for component,slots in ((self.package.gate,roles[1:4]),(self.package.down,roles[4:])):
            if component.expansion is None:
                raise ValueError('native quantized MLP requires both expansions')
            _,_,_,offsets,_,_=component.expansion.contract()
            checked_expansion_inputs(*(inputs[i] for i in slots),row_offsets=offsets,numeric_policy=nvfp4_expansion_policy(),_scale_bounds=self._nvfp4_scale_bounds())
        if not self._bounded:
            return super()._inputs(inputs)
        active_m=self._input_m(inputs[roles[0]])
        active_k=self._input_k(inputs[roles[0]])
        arrays=self._inputs_for_storage(inputs,self._storage(active_m,active_k))
        self._active_m=active_m
        if self._k_bounded:
            self._active_k=active_k
        return arrays

    def _update_shape(self,active_m,active_k,slots,arrays):
        update=self.lib.tessera_rocm_program_update_shape
        update.argtypes=[C.c_uint64,C.c_int64,C.c_int64,C.c_uint32,C.POINTER(C.c_uint32),
                         C.POINTER(C.c_void_p),C.POINTER(C.c_uint64)]
        update.restype=C.c_int
        count=len(slots)
        _status(update(self.handle,active_m,active_k//16,count,(C.c_uint32*count)(*slots),
                       (C.c_void_p*count)(*[a.ctypes.data for a in arrays]),
                       (C.c_uint64*count)(*[a.nbytes for a in arrays])))
        self._active_m,self._active_k=active_m,active_k

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
        # Validates against the requested M (and K) before any native state changes.
        previous=self._active_m,getattr(self,'_active_k',None)
        try:
            arrays,_,_=self._inputs(inputs)
            if self._k_bounded:
                self._update_shape(self._active_m,self._active_k,tuple(range(7)),arrays)
            else:
                self._update_rows(self._active_m,tuple(range(7)),arrays)
        except Exception:
            self._active_m=previous[0]
            if self._k_bounded:
                self._active_k=previous[1]
            raise

    def update_activations(self,activation):
        """Replace only the original activation; expanded weights stay resident."""
        from .native_scaled_program import pack_host_view
        if not self._bounded:
            raise ValueError('static native quantized MLP has no bounded row frame')
        active_m=self._input_m(activation)
        if self._k_bounded and self._input_k(activation)!=self._active_k:
            # The resident expanded weights were produced at the active K.
            raise ValueError('native quantized MLP activation K differs from the resident weights; update all roots')
        contract=self._storage(active_m)[self._x_slot]
        array=np.asarray(activation)
        if array.dtype!=contract.numpy_dtype or array.shape!=contract.shape or array.nbytes!=contract.bytes:
            raise ValueError('native quantized MLP activation storage/shape differs')
        if self._k_bounded:
            self._update_shape(active_m,self._active_k,(self._x_slot,),(pack_host_view(self.lib,array),))
        else:
            self._update_rows(active_m,(self._x_slot,),(pack_host_view(self.lib,array),))

    @property
    def active_k(self):
        if not self._k_bounded:
            raise ValueError('native quantized MLP has no bounded contraction frame')
        return self._active_k

    def shape_info(self):
        """(capacity M, capacity K, active M, active K, allocation bytes, buffers)."""
        if not self._k_bounded:
            raise ValueError('native quantized MLP has no bounded contraction frame')
        info=self.lib.tessera_rocm_program_shape_info
        info.argtypes=[C.c_uint64,*[C.POINTER(C.c_int64)]*4,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
        info.restype=C.c_int
        cm,cg,am,ag=(C.c_int64() for _ in range(4))
        size,count=C.c_uint64(),C.c_uint64()
        _status(info(self.handle,C.byref(cm),C.byref(cg),C.byref(am),C.byref(ag),C.byref(size),C.byref(count)))
        return cm.value,16*cg.value,am.value,16*ag.value,size.value,count.value

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
