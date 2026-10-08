"""Verified frontend partition and checked replay for resident gfx1201 ingest."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import json

from .graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,tensor_ir_type
from .rocm_mxfp4_packed_folded import PACKED_FOLDED_PHYSICAL_V1
from .rocm_mxfp4_storage_native import build_mxfp4_storage_graph,package_mxfp4_storage_graph
from .rocm_nvfp4_ingest import nvfp4_requantization_policy
from .rocm_nvfp4_ingest_native import build_nvfp4_ingest_graph,package_nvfp4_ingest_graph
from .rocm_nvfp4_resident import NVFP4ResidentProgram,_program_digest,package_resident_packed_consumer
from .scheduled_matmul import find_tessera_opt,run_tessera_opt


def packed_consumer_attrs(k):
    return {
        "physical_contract":PACKED_FOLDED_PHYSICAL_V1,
        "numeric_policy":{"accum":"fp32","execution_mode":"folded_row_reference_explicit_approximate"},
        "scale_layout":{"granularity":"output_column","block":[1,k],
                        "format":"e8m0_k32_plus_row_reference"},
    }


def build_packed_consumer_module(m,n,k):
    # Shape/type admission is owned jointly by catalog inference and native ODS.
    from .graph_ir import _infer_result_types
    types=[tensor_ir_type((m,k),"uint8"),tensor_ir_type((n,k//2),"uint8"),
           tensor_ir_type((m,),"fp32"),tensor_ir_type((k//32+1,n),"uint8")]
    attrs=packed_consumer_attrs(k)
    results=_infer_result_types("tessera.scaled_matmul",types,attrs)
    op=IROp(result="out",op_name="tessera.scaled_matmul",operands=["%a","%b","%sa","%plane"],
        operand_types=list(map(str,types)),result_type=str(results[0]),kwargs=attrs,
        inferred_type=results[0],inferred_types=tuple(results))
    return GraphIRModule([GraphIRFunction("packed_folded_w4a8",
        args=[IRArg(name,ty) for name,ty in zip(("a","b","sa","plane"),types)],
        body=[op],result_types=list(results),return_values=["%out"])],
        module_attrs={"tessera.target":json.dumps("rocm_gfx1201"),"tessera.arch":json.dumps("gfx1201")})


def _full_graph(m,n,k,offsets,indices):
    converter=deepcopy(build_nvfp4_ingest_graph(n,k,offsets,
        numeric_policy=nvfp4_requantization_policy()).functions[0])
    storage=deepcopy(build_mxfp4_storage_graph(n,k).functions[0])
    consumer=deepcopy(build_packed_consumer_module(m,n,k).functions[0])
    role_types=[*[a.ir_type for a in converter.args],consumer.args[0].ir_type,consumer.args[2].ir_type]
    args: list[IRArg | None] = [None]*5
    for role,index in enumerate(indices):
        args[index]=IRArg("arg"+str(index),role_types[role])
    if any(arg is None for arg in args):
        raise ValueError("resident frontend argument projection must cover every role")
    ordered_args=[arg for arg in args if arg is not None]
    converter.body[0].operands=["%arg"+str(i) for i in indices[:3]]
    storage.body[0].operands=["%packed","%exponents"]
    consumer.body[0].operands=["%arg"+str(indices[3]),"%fragment","%arg"+str(indices[4]),"%plane"]
    fn=GraphIRFunction("nvfp4_resident",args=ordered_args,body=[
        converter.body[0],storage.body[0],consumer.body[0]],
        result_types=consumer.result_types,return_values=["%out"])
    return GraphIRModule([fn],module_attrs=consumer_module_attrs())


def consumer_module_attrs():
    return {"tessera.target":json.dumps("rocm_gfx1201"),"tessera.arch":json.dumps("gfx1201")}


def supports_resident_trace(module):
    return (len(module.functions)==1 and len(module.functions[0].body)==3
        and [op.op_name for op in module.functions[0].body]==[
            "tessera.nvfp4_requantize","tessera.mxfp4_folded_storage","tessera.scaled_matmul"])


def _preserve_operation(actual,expected):
    if (actual.op_name!=expected.op_name or actual.kwargs!=expected.kwargs
            or actual.operand_types!=expected.operand_types or actual.result_type!=expected.result_type
            or actual.attrs not in (None,"") or actual.numeric_policy is not None):
        raise ValueError("resident frontend operation type/attributes/policy differs from named contract")
    # Rename SSA and strip source locations; retain caller semantic operation/attributes.
    normalized=deepcopy(actual)
    normalized.result=expected.result
    normalized.operands=list(expected.operands)
    normalized.source_span=None
    return normalized


def package_traced_resident(module):
    """Verify the complete Graph before splitting its native stage functions."""
    module=deepcopy(module)
    if not supports_resident_trace(module):
        raise ValueError("resident frontend requires conversion, storage and scaled matmul")
    for key,value in consumer_module_attrs().items():
        existing=module.module_attrs.get(key)
        if existing is not None and existing!=value:
            raise ValueError("resident frontend target/architecture differs")
        module.module_attrs[key]=value
    fn=module.functions[0]
    if any(a.effect or a.shard_spec or a.dim_names or a.layout
           or a.model_parameter or a.model_parameter_bytes_bound is not None
           for a in fn.args):
        raise ValueError("resident frontend argument effect/layout/sharding/model metadata needs a native contract")
    allowed_fn={"tessera.frontend.authority","tessera.structured_cfg.schema",
                "tessera.structured_cfg.digest","tessera.structured_cfg.blocks"}
    if set(fn.fn_attrs)-allowed_fn:
        raise ValueError("resident frontend function metadata needs a native contract")
    converter,storage,consumer=fn.body
    names=[a.name for a in fn.args]
    if (len(names)!=5 or len(set(names))!=5 or len(fn.result_types)!=1
            or len(converter.operands)!=3 or len(converter.result_names)!=3
            or len(storage.operands)!=2 or len(storage.result_names)!=2
            or len(consumer.operands)!=4 or len(consumer.result_names)!=1
            or fn.return_values!=["%"+consumer.result_names[0]]
            or any(op.kwargs.get("_region") for op in fn.body)):
        raise ValueError("resident frontend needs five inputs and one final output")
    if (storage.operands!=["%"+name for name in converter.result_names[:2]]
            or consumer.operands[1]!="%"+storage.result_names[0]
            or consumer.operands[3]!="%"+storage.result_names[1]):
        raise ValueError("resident frontend lost converter/storage/matmul SSA edge")
    external=[*converter.operands,consumer.operands[0],consumer.operands[2]]
    if any(operand not in ["%"+name for name in names] for operand in external):
        raise ValueError("resident frontend inputs must be explicit arguments")
    indices=tuple(names.index(value[1:]) for value in external)
    if len(set(indices))!=5:
        raise ValueError("resident frontend argument roles must be distinct")
    try:
        m,k=map(int,fn.args[indices[3]].ir_type.shape)
        n,half_k=map(int,fn.args[indices[0]].ir_type.shape)
    except (ValueError,TypeError):
        raise ValueError("resident frontend requires static rank-two tensors") from None
    if half_k*2!=k:
        raise ValueError("resident frontend conversion and consumer K differs")
    # Frontend admission checks attributes without rebuilding any stage Graph.
    from .rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT
    if (consumer.kwargs != packed_consumer_attrs(k)
            or storage.kwargs != {"storage_contract": MXFP4_STORAGE_CONTRACT}
            or set(converter.kwargs) != {"row_offsets", "numeric_policy"}
            or converter.kwargs["numeric_policy"] != nvfp4_requantization_policy()
            or any(op.attrs not in (None, "") or op.numeric_policy is not None for op in fn.body)):
        raise ValueError("resident frontend operation attributes/policy differs from named contract")
    from .structured_cfg import recover_structured_cfg
    if fn.structured_cfg is not None and fn.structured_cfg.digest!=recover_structured_cfg(fn.body).digest:
        raise ValueError("resident frontend CFG differs from semantic operations")
    from .native_nvfp4_program import export_native_nvfp4_program
    native = export_native_nvfp4_program(module.to_mlir(target="rocm_gfx1201",canonical=True))
    plan = native.manifest
    if tuple(plan["role_indices"]) != indices:
        raise ValueError("resident frontend/native argument roles differ")
    members = [native.project_member(index) for index in range(3)]
    program=NVFP4ResidentProgram(
        package_nvfp4_ingest_graph(members[0]), package_mxfp4_storage_graph(members[1]),
        package_resident_packed_consumer(m,n,k,graph=members[2],native_plan_json=native.plan_json),
        native.plan_json)
    result=TracedNVFP4Program(program,tuple(names),indices,plan["source_graph_ir"])
    result.validate()
    return result


@dataclass(frozen=True)
class TracedNVFP4Program:
    native:NVFP4ResidentProgram
    argument_names:tuple[str,...]
    role_indices:tuple[int,...]
    graph_ir:str

    def validate(self):
        self.native.validate()
        if (len(self.argument_names)!=5 or len(set(self.argument_names))!=5
                or any(not isinstance(name,str) or not name.isidentifier() for name in self.argument_names)
                or len(self.role_indices)!=5 or any(type(i) is not int for i in self.role_indices)
                or sorted(self.role_indices)!=list(range(5))):
            raise ValueError("resident frontend argument ABI differs")
        for name in self.argument_names:
            inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if self.native.native_plan_json is not None:
            from .rocm_nvfp4_resident import _native_nvfp4_plan
            plan = _native_nvfp4_plan(self.native.native_plan_json)
            if (self.graph_ir != plan["source_graph_ir"]
                    or list(self.role_indices) != plan["role_indices"]):
                raise ValueError("resident frontend retained native Graph/role lineage differs")
        else:
            c=self.native.consumer
            offsets=self.native.ingest.native.descriptor.provenance["row_offsets"]
            expected=_full_graph(c.m,c.n,c.k,offsets,self.role_indices).to_mlir(
                target="rocm_gfx1201",canonical=True)
            if self.graph_ir!=expected:
                raise ValueError("resident frontend retained Graph/role lineage differs")

    def execute(self,*args,**kwargs):
        self.validate()
        signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                     for name in self.argument_names])
        bound=signature.bind(*args,**kwargs)
        ordered=[bound.arguments[name] for name in self.argument_names]
        with self.native.native_session(*(ordered[i] for i in self.role_indices),reuse=True) as session:
            session.run_combined()
            output=session.read_output()
        components=(self.native.ingest.native,self.native.storage.native,self.native.consumer.package)
        receipts=tuple({"ok":True,"execution_kind":"native_gpu","image_digest":p.image.image_digest,
            "descriptor_digest":p.descriptor.descriptor_digest,"abi_id":p.descriptor.abi_id,
            "native_call_binding":"native_cpp_nvfp4",
            "native_allocation_cache_hit":session.native_cache_hit}
            for p in components)
        return output,receipts

    def manifest(self):
        self.validate()
        data={"schema":"tessera.rocm.nvfp4_frontend_program.v1",
              "native_program":self.native.to_dict(),"argument_names":list(self.argument_names),
              "role_indices":list(self.role_indices),"graph_ir":self.graph_ir}
        return deepcopy({**data,"contract_digest":_program_digest(data)})


def program_from_manifest(data):
    if not isinstance(data,dict) or set(data)!={"schema","native_program","argument_names",
            "role_indices","graph_ir","contract_digest"}:
        raise ValueError("resident frontend manifest fields differ")
    data=deepcopy(data)
    if data["schema"]!="tessera.rocm.nvfp4_frontend_program.v1":
        raise ValueError("resident frontend manifest version differs")
    identity={key:value for key,value in data.items() if key!="contract_digest"}
    if data["contract_digest"]!=_program_digest(identity):
        raise ValueError("resident frontend manifest digest differs")
    if (not isinstance(data["argument_names"],list) or not isinstance(data["role_indices"],list)
            or not isinstance(data["graph_ir"],str)):
        raise ValueError("resident frontend manifest argument/Graph schema differs")
    program=TracedNVFP4Program(NVFP4ResidentProgram.from_dict(data["native_program"]),
        tuple(data["argument_names"]),tuple(data["role_indices"]),data["graph_ir"])
    program.validate()
    return program


def runtime_artifact(program):
    from tessera.runtime import RuntimeArtifact
    program.validate()
    return RuntimeArtifact(graph_ir=program.graph_ir,
        metadata={"target":"rocm_gfx1201","compiler_path":"canonical_rocm_nvfp4_program",
                  "execution_kind":"native_gpu","runtime_status":"ready","executable":True,
                  "native_graph_verified":True,"arg_names":list(program.argument_names),
                  "native_program":program.manifest()})


def reference_scaled_matmul(a,b,scale_a,scale_b,*,physical_contract,numeric_policy,scale_layout,
                            transposeA=False,transposeB=False,batching=None):
    """CPU oracle for the named folded physical profile; never used by JIT."""
    if type(transposeA) is not bool or type(transposeB) is not bool:
        raise ValueError("scaled_matmul transpose attributes must be boolean")
    if transposeA or transposeB or batching is not None:
        raise ValueError("packed folded reference requires no transpose or batching")
    import numpy as np
    import ml_dtypes
    from .rocm_mxfp4 import convert_weight_layout,MXFP4_GFX12_FRAGMENT_LAYOUT_V1,MXFP4_CHECKPOINT_LAYOUT_V1,folded_weights
    from .rocm_mxfp4_folded import prepare_folded_weights
    from .rocm_nvfp4_resident import _activation_inputs
    from .graph_ir import _infer_result_types
    values=[np.asarray(v) for v in (a,b,scale_a,scale_b)]
    if values[0].ndim!=2:
        raise ValueError("scaled_matmul reference requires rank-two A")
    m,k=values[0].shape
    _infer_result_types("tessera.scaled_matmul",
        [tensor_ir_type(v.shape,str(v.dtype)) for v in values],
        {"physical_contract":physical_contract})
    if {"physical_contract":physical_contract,"numeric_policy":numeric_policy,
            "scale_layout":scale_layout}!=packed_consumer_attrs(k):
        raise ValueError("scaled_matmul reference supports only the explicit packed folded contract")
    a,scale_a=_activation_inputs(m,k,values[0],values[2])
    checkpoint=convert_weight_layout(values[1],source=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
                                    destination=MXFP4_CHECKPOINT_LAYOUT_V1)
    folded=prepare_folded_weights(checkpoint,values[3][:-1],allow_approximate=True)
    if not np.array_equal(folded.row_reference,values[3][-1]):
        raise ValueError("scaled_matmul reference row scale differs")
    decoded=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*scale_a[:,None]
    return (decoded @ folded_weights(folded).astype(np.float64).T).astype(ml_dtypes.bfloat16)
