"""Checked binding of compiler-owned gate/up and down-projection SSA plans."""
from __future__ import annotations
import base64
import ctypes as C
from dataclasses import dataclass,replace
import hashlib
import json
import math
from typing import Any

from .native_scaled_program import PreparedScaledProgram,_BoundScaledABI,_Buffer,_StorageSpec,_status
from .rocm_nvfp4_expansion_program import _MultiOutputStep
from .rocm_quantized_product_program import NativeQuantizedProductProgram,PreparedQuantizedProductProgram,_frame_shape,package_quantized_product_graph

_SCHEMA='tessera.native.quantized_product.v4'
_PACKAGE='tessera.rocm.quantized_product_package.v4'


def _components(plan):
    if (not isinstance(plan,dict) or plan.get('schema')!=_SCHEMA
            or set(plan)!={'schema','source_graph_ir','member_graphs','buffers','steps','output','subprograms'}
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
    return roles,shape,(gm,dm)


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
    def schedule_irs(self):
        return self.gate.schedule_irs+self.down.schedule_irs

    def _identity(self):
        return hashlib.sha256(json.dumps([self.program_json,self.graph_ir,
            self.gate.binding_digest,self.down.binding_digest],separators=(',',':')).encode()).hexdigest()

    def validate(self):
        self.gate.validate();self.down.validate()
        plan=json.loads(self.program_json)
        _frame(plan)
        components=_components(plan)
        if (plan['source_graph_ir']!=self.graph_ir or self.binding_digest!=self._identity()
                or json.loads(self.gate.program_json)!=components[0]
                or json.loads(self.down.program_json)!=components[1]):
            raise ValueError('native quantized MLP member binding differs')

    def to_manifest(self):
        self.validate()
        return {'schema':_PACKAGE,'program_json':self.program_json,'graph_ir':self.graph_ir,
                'gate':self.gate.to_manifest(),'down':self.down.to_manifest(),'binding_digest':self.binding_digest}

    @classmethod
    def from_manifest(cls,record):
        if (not isinstance(record,dict) or set(record)!={'schema','program_json','graph_ir','gate','down','binding_digest'}
                or record['schema']!=_PACKAGE
                or any(not isinstance(record[k],str) for k in ('program_json','graph_ir','binding_digest'))):
            raise ValueError('native quantized MLP portable record differs')
        # Exact component schemas prevent recursive full-program manifests.
        if (not isinstance(record['gate'],dict) or not isinstance(record['down'],dict)
                or record['gate'].get('schema')!='tessera.rocm.quantized_product_package.v3'
                or record['down'].get('schema')!='tessera.rocm.quantized_product_package.v2'):
            raise ValueError('native quantized MLP portable component schemas differ')
        program=cls(record['program_json'],record['graph_ir'],
            NativeQuantizedProductProgram.from_manifest(record['gate']),
            NativeQuantizedProductProgram.from_manifest(record['down']),record['binding_digest'])
        program.validate()
        return program

    def prepare(self,inputs,*,runtime_library):
        return PreparedQuantizedMLPProgram(self,inputs,runtime_library=runtime_library)


def package_quantized_mlp_plan(plan,tool):
    from .rocm_quantized_product_program import _export
    _frame(plan)
    if _export(tool,plan['source_graph_ir'])!=plan:
        raise ValueError('native quantized MLP normalized source replay differs')
    gate,down=_components(plan)
    program=NativeQuantizedMLPProgram(json.dumps(plan,sort_keys=True),plan['source_graph_ir'],
        package_quantized_product_graph(gate['source_graph_ir']),
        package_quantized_product_graph(down['source_graph_ir']),'')
    program=replace(program,binding_digest=program._identity())
    program.validate()
    return program


class PreparedQuantizedMLPProgram(PreparedScaledProgram):
    package: NativeQuantizedMLPProgram
    _prepare_symbol='tessera_rocm_program_prepare_multi_output'
    _step_type=_MultiOutputStep

    def __init__(self,package,inputs,*,runtime_library):
        from tessera import runtime as rt
        if rt._rocm_live_arch()!='gfx1201':
            raise RuntimeError('native quantized MLP requires owning gfx1201')
        super().__init__(package,inputs,runtime_library=runtime_library)

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
            checked_expansion_inputs(*(inputs[i] for i in slots),row_offsets=offsets,numeric_policy=nvfp4_expansion_policy())
        return super()._inputs(inputs)

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
