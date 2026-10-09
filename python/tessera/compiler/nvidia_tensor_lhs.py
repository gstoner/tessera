"""Verified frontend LHS producers over native Schedule/Tile tensor packages."""
from copy import deepcopy
from dataclasses import dataclass,replace
import hashlib
import inspect
import json
import math
import struct
from typing import cast

from . import nvidia_native as native
from .graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,tensor_ir_type
from .scheduled_matmul import find_tessera_opt,_SM120_SCHEDULED_MATMUL_PREFIX

PRODUCERS={"tessera.rmsnorm":"rmsnorm","tessera.layer_norm":"layernorm","tessera.softmax":"softmax"}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def candidate(module):
    if len(module.functions)!=1 or not 2<=len(module.functions[0].body)<=64:
        return False
    from .nvidia_tensor_dag import candidate as dag_candidate
    if dag_candidate(module):
        return True
    body=module.functions[0].body
    producers,c=body[:-1],body[-1]
    return (all(p.op_name in PRODUCERS and len(p.operands)==1 and p.result for p in producers)
            and all(p.operands==["%"+str(before.result)] for before,p in zip(producers,producers[1:]))
            and c.op_name in {"tessera.matmul","tessera.gemm"}
            and c.operands and c.operands[0]=="%"+str(producers[-1].result))


def project_rhs_storage(module, ordered, *, dynamic=False, rhs_storage_order=None):
    """Record host RHS storage facts on a copy of the semantic frontend trace."""
    if rhs_storage_order is not None and (type(rhs_storage_order) is not str or rhs_storage_order not in {"row_major","col_major"}):
        raise ValueError("RHS storage order must be row_major or col_major")
    module = deepcopy(module)
    if not candidate(module):
        return module
    fn = module.functions[0]
    consumer = fn.body[-1]
    if rhs_storage_order is not None:
        if consumer.kwargs.get("rhs_storage_order",rhs_storage_order)!=rhs_storage_order:
            raise ValueError("RHS storage request conflicts with the authored Graph")
        consumer.kwargs["rhs_storage_order"]=rhs_storage_order
    from .nvidia_tensor_dag import candidate as dag_candidate
    if dag_candidate(module):
        if consumer.kwargs.get("rhs_storage_order","row_major")!="row_major":
            raise ValueError("computed RHS requires row-major native materialization")
        return module
    if "rhs_storage_order" not in consumer.kwargs:
        names = [arg.name for arg in fn.args]
        rhs = ordered[names.index(consumer.operands[1].removeprefix("%"))]
        # Padded/sliced inputs retain the existing compact column-major upload.
        # Compact C storage can now feed the native row-RHS Schedule recipe.
        consumer.kwargs["rhs_storage_order"] = (
            "row_major" if rhs.flags.c_contiguous and not dynamic else "col_major")
    return module


def _checked_semantics(semantics):
    s=deepcopy(semantics)
    if set(s)-{"producer","producer_attrs","consumer","consumer_attrs","roles","producer_chain","rhs_chain"} or not {"producer","producer_attrs","consumer","consumer_attrs","roles"}<=set(s):
        raise ValueError("LHS semantic certificate fields differ")
    if s["producer"] not in PRODUCERS or s["consumer"] not in {"tessera.matmul","tessera.gemm"}:
        raise ValueError("LHS requires a registered normalization/softmax and matmul")
    if "rhs_chain" in s:
        right=s["rhs_chain"]
        if not isinstance(right,list) or not 1<=len(right)<=63:
            raise ValueError("RHS producer chain count differs")
        for row in right:
            if not isinstance(row,dict) or set(row)!={"producer","producer_attrs"}:
                raise ValueError("RHS producer semantic fields differ")
            _checked_semantics({**{k:v for k,v in s.items() if k not in {"producer_chain","rhs_chain"}},**row})
    if "producer_chain" in s:
        chain=s["producer_chain"]
        if not isinstance(chain,list) or not 2<=len(chain)<=63:
            raise ValueError("LHS producer chain count differs")
        for row in chain:
            if not isinstance(row,dict) or set(row)!={"producer","producer_attrs"}:
                raise ValueError("LHS producer chain semantic fields differ")
            _checked_semantics({**{k:v for k,v in s.items() if k!="producer_chain"},**row})
        if chain[0]!={"producer":s["producer"],"producer_attrs":s["producer_attrs"]}:
            raise ValueError("LHS first producer certificate differs")
    pa,ca=s["producer_attrs"],s["consumer_attrs"]
    if not isinstance(pa,dict) or not isinstance(ca,dict):
        raise ValueError("LHS attributes must be dictionaries")
    pa,ca=dict(sorted(pa.items())),dict(sorted(ca.items()))
    if s["producer"]=="tessera.softmax":
        if set(pa)-{"axis"} or pa.get("axis",-1)!=-1:
            raise ValueError("LHS softmax requires the last-axis native contract")
    elif (set(pa)-{"eps","gamma","beta"} or pa.get("gamma") is not None or pa.get("beta") is not None):
        raise ValueError("LHS normalization needs a separate affine/axis contract")
    eps=pa.get("eps",1e-5)
    if s["producer"]!="tessera.softmax" and (
        type(eps) not in (int,float) or not math.isfinite(eps) or eps<=0):
        raise ValueError("LHS normalization epsilon must be finite and positive")
    if (set(ca)-{"output_dtype","activation","bias","residual","rhs_storage_order"}
            or ca.get("activation","none") not in {"none","relu","gelu","silu"}
            or ca.get("output_dtype","fp32") not in {"fp16","fp32"}
            or ca.get("rhs_storage_order","col_major") not in {"row_major","col_major"}
            or ca.get("bias") not in (None,"bias") or ca.get("residual") not in (None,"residual")):
        raise ValueError("LHS matmul attributes differ from the complete native epilogue contract")
    roles=s["roles"]
    has_bias=ca.get("bias") is not None
    has_residual=ca.get("residual") is not None
    count=2+has_bias+has_residual
    wanted={"source","rhs"}|({"bias"} if has_bias else set())|({"residual"} if has_residual else set())
    if (not isinstance(roles,dict) or set(roles)!=wanted
            or any(type(v) is not int for v in roles.values())
            or sorted(roles.values())!=list(range(count))):
        raise ValueError("LHS frontend argument roles differ")
    return s,pa,ca,roles


def _semantic_graph(m,k,n,dtype,semantics, dynamic_axes=()):
    s,pa,ca,roles=_checked_semantics(semantics)
    has_bias="bias" in roles
    has_residual="residual" in roles
    types={"source":tensor_ir_type((m,k),dtype),"rhs":tensor_ir_type((k,n),dtype),
           "bias":tensor_ir_type((n,),"fp32"),"residual":tensor_ir_type((m,n),"fp32")}
    args=[IRArg("arg"+str(index),types[role])
          for index,role in sorted((index,role) for role,index in roles.items())]
    ptypes=[types["source"]]
    p=IROp(result="edge",op_name=s["producer"],operands=["%arg"+str(roles["source"])],
           operand_types=list(map(str,ptypes)),result_type=str(types["source"]),kwargs=pa)
    order=["rhs"]+(["bias"] if has_bias else [])+(["residual"] if has_residual else [])
    ctypes=[types["source"]]+[types[role] for role in order]
    result=tensor_ir_type((m,n),ca.get("output_dtype","fp32"))
    c=IROp(result="out",op_name=s["consumer"],
           operands=["%edge"]+["%arg"+str(roles[role]) for role in order],
           operand_types=list(map(str,ctypes)),result_type=str(result),kwargs=ca)
    graph=GraphIRModule([GraphIRFunction("native_lhs",args=args,body=[p,c],
        result_types=[result],return_values=["%out"])],
        module_attrs={"tessera.target":'"nvidia_sm120"',"tessera.arch":'"sm_120"'})
    if dynamic_axes:
        from .scheduled_matmul import with_bounded_dynamic_axes
        _,consumer=_partitions(graph)
        consumer=_epilogue_argument_names(consumer)
        consumer=with_bounded_dynamic_axes(consumer,tuple(dynamic_axes))
        projected=consumer.functions[0]
        by_name={a.name:a.ir_type for a in projected.args}
        for arg in graph.functions[0].args:
            if arg.name in by_name:arg.ir_type=by_name[arg.name]
        source_type=by_name["edge"]
        graph.functions[0].args[roles["source"]].ir_type=source_type
        p.operand_types=[str(source_type)]
        p.result_type=str(source_type)
        p.inferred_type=source_type
        graph.functions[0].body[1]=projected.body[0]
        graph.functions[0].result_types=projected.result_types
    return graph


def _partitions(graph):
    fn=graph.functions[0];p,c=fn.body
    by_name={a.name:a for a in fn.args}
    producer=replace(fn,name=fn.name+"__producer",args=[by_name[p.operands[0][1:]]],
                     body=[p],result_types=[by_name[p.operands[0][1:]].ir_type],
                     return_values=["%edge"])
    consumer_args=[IRArg("edge",producer.result_types[0])]+[by_name[v[1:]] for v in c.operands[1:]]
    consumer=replace(fn,name=fn.name+"__consumer",args=consumer_args,body=[c])
    return replace(graph,functions=[producer]),replace(graph,functions=[consumer])


def _epilogue_argument_names(consumer):
    """Project semantic role markers to the actual traced SSA argument names."""
    consumer=deepcopy(consumer)
    op=consumer.functions[0].body[0]
    position=2
    for role in ("bias","residual"):
        if op.kwargs.get(role) is not None:
            op.kwargs[role]=op.operands[position].removeprefix("%")
            position+=1
    return consumer


def package_traced_lhs(module, *, producer_schedule=None, softmax_schedule=None, dynamic_axes=(), shape_bounds=None):
    module=deepcopy(module)
    if shape_bounds is not None:
        from .bounded_nvidia_lhs import validate_bounds
        shape_bounds=dict(validate_bounds(shape_bounds))
        if dynamic_axes and set(dynamic_axes)!=set(shape_bounds):
            raise ValueError("LHS dynamic axes and shape bounds differ")
        dynamic_axes=tuple(shape_bounds)
    if (not isinstance(dynamic_axes,(tuple,list))
            or any(type(axis) is not str or axis not in {"M","N","K"} for axis in dynamic_axes)
            or len(set(dynamic_axes))!=len(dynamic_axes)):
        raise ValueError("LHS dynamic axes must be distinct M/N/K names")
    dynamic_axes=tuple(axis for axis in ("M","N","K") if axis in dynamic_axes)
    from .nvidia_tensor_dag import candidate as dag_candidate, package as package_dag
    if dag_candidate(module):
        if producer_schedule is not None or softmax_schedule is not None:
            raise ValueError("DAG Schedule overrides require per-node policies")
        return package_dag(module,dynamic_axes=dynamic_axes,shape_bounds=shape_bounds)
    if not candidate(module):
        raise ValueError("LHS trace requires matmul(producer(source), rhs)")
    if set(module.module_attrs)-{"tessera.ir.version","tessera.frontend.authority",
                                  "tessera.target","tessera.arch"}:
        raise ValueError("LHS module metadata requires a native contract")
    fn=module.functions[0];p,c=fn.body[0],fn.body[-1]
    producers=fn.body[:-1]
    if (any(a.effect or a.shard_spec or a.dim_names or a.layout or a.model_parameter
            or a.model_parameter_bytes_bound is not None for a in fn.args)
            or set(fn.fn_attrs)-{"tessera.frontend.authority","tessera.structured_cfg.schema",
                "tessera.structured_cfg.digest","tessera.structured_cfg.blocks"}):
        raise ValueError("LHS semantic argument/function metadata requires a native contract")
    for key,value in (("tessera.target",'"nvidia_sm120"'),("tessera.arch",'"sm_120"')):
        if module.module_attrs.get(key,value)!=value:
            raise ValueError("LHS target/architecture differs")
    if (len(p.operands)!=1 or not p.result or not c.result or len(fn.result_types)!=1
            or fn.return_values!=["%"+c.result] or len(c.operands)<2
            or any(op.numeric_policy is not None or op.attrs not in (None,"") for op in fn.body)):
        raise ValueError("LHS return/operation semantics differ")
    names=[a.name for a in fn.args]
    external=[p.operands[0],*c.operands[1:]]
    if any(v not in ["%"+name for name in names] for v in external):
        raise ValueError("LHS external operands must be direct arguments")
    role_names=["source","rhs"]+(["bias"] if c.kwargs.get("bias") is not None else [])+(
        ["residual"] if c.kwargs.get("residual") is not None else [])
    if len(role_names)!=len(external):
        raise ValueError("LHS optional epilogue operands differ")
    roles={role:names.index(value[1:]) for role,value in zip(role_names,external)}
    source=fn.args[roles["source"]].ir_type;rhs=fn.args[roles["rhs"]].ir_type
    try:
        m,k=map(int,source.shape);right_k,n=map(int,rhs.shape)
    except (ValueError,TypeError):
        raise ValueError("LHS source/RHS require static rank-two tensors") from None
    if right_k!=k or source.dtype not in {"fp16","bf16"} or source.dtype!=rhs.dtype:
        raise ValueError("LHS source/RHS storage or K differs")
    sem={"producer":p.op_name,"producer_attrs":p.kwargs,
         "consumer":c.op_name,"consumer_attrs":c.kwargs,"roles":roles}
    if len(producers)>1:
        sem["producer_chain"]=[{"producer":op.op_name,"producer_attrs":deepcopy(op.kwargs)} for op in producers]
    _checked_semantics(sem)
    output=fn.result_types[0]
    if (tuple(str(d) for d in output.shape)!=(str(m),str(n)) or
            output.dtype!=c.kwargs.get("output_dtype","fp32") or
            p.operand_types!=[str(source)] or p.result_type!=str(source) or
            c.operand_types!=[str(source),*[str(fn.args[roles[role]].ir_type) for role in role_names[1:]]] or
            c.result_type!=str(output)):
        raise ValueError("LHS operation storage/shape differs")
    for role in ("bias","residual"):
        if role in roles:
            arg=fn.args[roles[role]].ir_type
            wanted=(str(n),) if role=="bias" else (str(m),str(n))
            if tuple(str(d) for d in arg.shape)!=wanted or arg.dtype!="fp32":
                raise ValueError("LHS Graph argument storage/shape differs")
    from .structured_cfg import recover_structured_cfg
    if fn.structured_cfg is not None and fn.structured_cfg.digest!=recover_structured_cfg(fn.body).digest:
        raise ValueError("LHS CFG differs from semantic operations")
    if producer_schedule is not None:
        if p.op_name=="tessera.softmax" or producer_schedule not in {"serial","cooperative_128"}:
            raise ValueError("explicit Schedule policy requires a normalization")
        p.kwargs={**p.kwargs,"schedule":producer_schedule}
    if softmax_schedule is not None:
        if softmax_schedule not in {"serial", "cooperative_128"}:
            raise ValueError("explicit softmax Schedule policy is invalid")
        selected = [op for op in producers if op.op_name == "tessera.softmax"]
        if not selected:
            raise ValueError("explicit softmax Schedule policy needs a softmax producer")
        for op in selected:
            op.kwargs = {**op.kwargs, "schedule": softmax_schedule}
    module.module_attrs.update({"tessera.target":'"nvidia_sm120"',"tessera.arch":'"sm_120"'})
    if dynamic_axes:
        active={"M":m,"N":n,"K":k}
        bounds={axis:(shape_bounds or {}).get(axis,active[axis]) for axis in dynamic_axes}
        if any(active[axis]>bound for axis,bound in bounds.items()):
            raise ValueError("LHS active shape is outside its declared bound")
        module.module_attrs["tessera.native.sm120_tensor_bounds"]="{"+", ".join(
            f"{axis} = {bound} : i64" for axis,bound in bounds.items())+"}"
    from .native_sm120_tensor_program import export_native_sm120_tensor_graph
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError("LHS graph needs the matching native compiler")
    projected=export_native_sm120_tensor_graph(module.to_mlir(canonical=True,target="nvidia_sm120"),tool=tool)
    record=projected.manifest
    if record["role_indices"]!=[roles[role] for role in role_names]:
        raise ValueError("native LHS argument roles differ")
    artifacts=projected.scheduled_members()
    capacities=record.get("shape_bounds",[m,n,k])
    edge=native.package_scheduled_tensor_matmul(*artifacts[-2:],pipeline_name="tessera-nvidia-pipeline-sm120",
        dynamic_m_bound=capacities[0] if "M" in dynamic_axes else None,
        dynamic_n_bound=capacities[1] if "N" in dynamic_axes else None,
        dynamic_k_bound=capacities[2] if "K" in dynamic_axes else None)
    plan_digest=hashlib.sha256(projected.plan_json.encode()).hexdigest()
    def bind_plan(package):
        return replace(package,descriptor=replace(package.descriptor,provenance={
            **package.descriptor.provenance,"native_tensor_program_digest":plan_digest}))
    edge=replace(edge,producer=bind_plan(edge.producer),consumer=bind_plan(edge.consumer))
    def package_producer(artifact):
        package=native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
        for axis,dimension,helper in (("M",0,native._with_dynamic_m_capacity),
                                      ("K",1,native._with_dynamic_k_capacity)):
            if axis in dynamic_axes:
                package=helper(package,input_name=artifact.input_name,
                               output_name=artifact.output_name,bound=artifact.input_shape[dimension])
        return bind_plan(package)
    chain=tuple(package_producer(artifact) for artifact in artifacts[:-2])+(edge.producer,) if len(producers)>1 else ()
    result=TracedLhsProgram(edge,tuple(names),deepcopy(sem),
        record["source_graph_ir"],projected.plan_json,chain)
    result.validate()
    return result


@dataclass(frozen=True)
class TracedLhsProgram:
    edge:native.NVIDIANativeTensorProgram
    argument_names:tuple[str,...]
    semantics:dict
    graph_ir:str
    native_plan_json:str|None=None
    producer_chain:tuple=()
    rhs_chain:tuple=()

    def validate(self):
        self.edge.validate()
        if any(type(v) is not int or v<=0 for v in (self.edge.m,self.edge.k,self.edge.n)):
            raise ValueError("LHS frontend program requires positive native package capacities")
        if self.native_plan_json is not None:
            self._validate_native()
            return
        if self.producer_chain or self.rhs_chain or "producer_chain" in self.semantics or "rhs_chain" in self.semantics:
            raise ValueError("LHS producer chains require native Graph member and lifetime certificates")
        dynamic_axes=tuple(axis for axis,enabled in (
            ("M",self.edge.dynamic_m),("N",self.edge.dynamic_n),("K",self.edge.dynamic_k)) if enabled)
        graph=_semantic_graph(self.edge.m,self.edge.k,self.edge.n,self.edge.dtype,
                              self.semantics,dynamic_axes)
        count=len(graph.functions[0].args)
        if (len(self.argument_names)!=count or len(set(self.argument_names))!=count
                or any(not isinstance(name,str) or not name.isidentifier() for name in self.argument_names)):
            raise ValueError("LHS frontend argument ABI differs")
        for name in self.argument_names:inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if graph.to_mlir(canonical=True,target="nvidia_sm120")!=self.graph_ir:
            raise ValueError("LHS retained Graph/semantic certificate differs")
        for package in (self.edge.producer,self.edge.consumer):
            package.descriptor.validate_image(package.image)
            if hashlib.sha256(package.target_ir.encode()).hexdigest()!=package.image.target_ir_digest:
                raise ValueError("LHS package Target integrity differs")
        p=self.edge.producer.descriptor.provenance
        if p.get("kind")!=PRODUCERS[self.semantics["producer"]]:
            raise ValueError("LHS producer kind differs")
        if p["kind"]!="softmax":
            eps=self.semantics["producer_attrs"].get("eps",1e-5)
            if p.get("epsilon")!=struct.unpack("f",struct.pack("f",eps))[0]:
                raise ValueError("LHS producer epsilon differs")
        _,consumer=_partitions(graph)
        consumer.functions[0].name=_SM120_SCHEDULED_MATMUL_PREFIX+consumer.functions[0].name
        digest=hashlib.sha256(consumer.to_mlir(canonical=True,target="nvidia_sm120").encode()).hexdigest()
        if self.edge.consumer.descriptor.provenance.get("graph_ir_digest")!=digest:
            raise ValueError("LHS consumer Graph provenance differs")
        roles=self.semantics["roles"]
        if (self.edge.producer_input_name!="arg"+str(roles["source"]) or self.edge.intermediate_name!="edge"
                or self.edge.consumer_input_name!="edge" or self.edge.consumer_rhs_name!="arg"+str(roles["rhs"])
                or self.edge.output_name!="out"):
            raise ValueError("LHS native package bindings differ from frontend roles")

    def _validate_native(self):
        if self.rhs_chain:
            from .nvidia_tensor_dag import validate
            validate(self)
            return
        from .native_sm120_tensor_program import validate_native_tensor_plan
        plan=validate_native_tensor_plan(self.native_plan_json)
        _,pa,ca,roles=_checked_semantics(self.semantics)
        count=len(roles)
        producers=self.producer_chain or (self.edge.producer,)
        chain=bool(self.producer_chain)
        if chain != ("producer_chain" in self.semantics):
            raise ValueError("native LHS chain semantics/member count differs")
        if chain!=(plan["schema"] in {"tessera.native.sm120_tensor_program.v3","tessera.native.sm120_tensor_program.v4"}) or len(plan["steps"])!=len(producers)+1:
            raise ValueError("native LHS chain/schema differs")
        if chain and (producers[-1]!=self.edge.producer or len(self.semantics.get("producer_chain",[]))!=len(producers)):
            raise ValueError("native LHS producer package count differs")
        for package in producers:
            replace(self.edge,producer=package).validate()
        if (len(self.argument_names)!=count or len(set(self.argument_names))!=count
                or any(not isinstance(name,str) or not name.isidentifier() for name in self.argument_names)):
            raise ValueError("LHS frontend argument ABI differs")
        for name in self.argument_names:
            inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
        role_names=["source","rhs"]+([ "bias"] if "bias" in roles else [])+(
            ["residual"] if "residual" in roles else [])
        if plan["role_indices"]!=[roles[role] for role in role_names]:
            raise ValueError("native LHS role certificate differs")
        if plan["source_graph_ir"]!=self.graph_ir or plan["steps"][0]["operation"]!=self.semantics["producer"]:
            raise ValueError("native LHS retained Graph/semantic certificate differs")
        axes=tuple(axis for axis,enabled in (("M",self.edge.dynamic_m),
            ("N",self.edge.dynamic_n),("K",self.edge.dynamic_k)) if enabled)
        if axes!=tuple(plan.get("dynamic_axes",[])):
            raise ValueError("native LHS dynamic projection differs")
        if axes and plan["shape_bounds"]!=[self.edge.m,self.edge.n,self.edge.k]:
            raise ValueError("native LHS capacity projection differs")
        buffers=plan["buffers"]
        expected_shape={"source":[self.edge.m,self.edge.k],"rhs":[self.edge.k,self.edge.n],
            "bias":[self.edge.n],"residual":[self.edge.m,self.edge.n]}
        expected_storage={"source":"f16" if self.edge.dtype=="fp16" else "bf16",
                          "rhs":"f16" if self.edge.dtype=="fp16" else "bf16",
                          "bias":"f32","residual":"f32"}
        for role,index in roles.items():
            if (buffers[index]["shape"]!=expected_shape[role] or
                    buffers[index]["storage"]!=expected_storage[role]):
                raise ValueError("native LHS buffer shape/storage differs")
        if buffers[count]!=dict(buffers[roles["source"]],id=count,ownership="private_scratch",
                                first_write=0,last_read=1):
            raise ValueError("native LHS edge capacity/lifetime differs")
        if buffers[plan["output"]]["shape"]!=[self.edge.m,self.edge.n]:
            raise ValueError("native LHS output shape differs")
        for index,package in enumerate((*producers,self.edge.consumer)):
            if package.descriptor.provenance.get("native_tensor_program_digest")!=hashlib.sha256(
                    cast(str, self.native_plan_json).encode()).hexdigest():
                raise ValueError("native LHS program provenance differs")
            package.descriptor.validate_image(package.image)
            if hashlib.sha256(package.target_ir.encode()).hexdigest()!=package.image.target_ir_digest:
                raise ValueError("LHS package Target integrity differs")
            if package.descriptor.provenance.get("graph_ir_digest")!=hashlib.sha256(
                    plan["member_graphs"][index].encode()).hexdigest():
                raise ValueError("native LHS member Graph provenance differs")
        policies=self.semantics.get("producer_chain",[{"producer":self.semantics["producer"],"producer_attrs":pa}])
        for index,(package,policy) in enumerate(zip(producers,policies,strict=True)):
            p=package.descriptor.provenance
            if plan["steps"][index]["operation"]!=policy["producer"] or p.get("kind")!=PRODUCERS[policy["producer"]]:
                raise ValueError("LHS producer kind differs")
            if p["kind"]!="softmax" and p.get("epsilon")!=struct.unpack("f",struct.pack("f",policy["producer_attrs"].get("eps",1e-5)))[0]:
                raise ValueError("LHS producer epsilon differs")
        cp=self.edge.consumer.descriptor.provenance
        epilogue=cp.get("epilogue")
        if not isinstance(epilogue,dict):
            raise ValueError("native LHS consumer policy differs")
        if (cp.get("epilogue")!={"bias":"bias" in roles,"activation":ca.get("activation","none"),
                "residual":"residual" in roles,"order":["matmul","bias","activation","residual"],
                "output":"f16" if ca.get("output_dtype","fp32")=="fp16" else "f32"}
                or cp.get("b_layout")!=ca.get("rhs_storage_order","col_major")
                or buffers[plan["output"]]["storage"]!=epilogue["output"]):
            raise ValueError("native LHS consumer policy differs")
        if (self.edge.producer_input_name,self.edge.intermediate_name,
            self.edge.consumer_input_name,self.edge.consumer_rhs_name,self.edge.output_name)!=(
                "source","edge","edge","rhs","out"):
            raise ValueError("native LHS package bindings differ")

    def execute_resident(self,*args,**kwargs):
        self.validate()
        if self.rhs_chain:
            from .nvidia_tensor_dag import execute_resident
            return execute_resident(self,args,kwargs)
        signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                    for name in self.argument_names])
        bound=signature.bind(*args,**kwargs)
        values=[bound.arguments[name] for name in self.argument_names]
        roles=self.semantics["roles"]
        if self.producer_chain:
            from .prepared_nvidia_lhs import execute_chain_resident
            return execute_chain_resident(self,values)
        return self.edge.execute_resident(values[roles["source"]],values[roles["rhs"]],
            **{role:values[roles[role]] for role in ("bias","residual") if role in roles})

    def manifest(self):
        self.validate()
        from .nvidia_tensor_rhs import NvidiaNormRhsProgram
        package_artifact=NvidiaNormRhsProgram.runtime_artifact
        labels=("producer_input_name","intermediate_name","consumer_input_name","consumer_rhs_name",
                "output_name","m","k","n","dtype")
        data={"schema":"tessera.nvidia.lhs_tensor_program.v1","graph_ir":self.graph_ir,
              "argument_names":list(self.argument_names),"semantics":deepcopy(self.semantics),
              "edge":{label:getattr(self.edge,label) for label in labels},
              "producer":package_artifact(self.edge.producer).to_dict(),
              "consumer":package_artifact(self.edge.consumer).to_dict()}
        if self.native_plan_json is not None:
            data["schema"]="tessera.nvidia.lhs_tensor_program.v2"
            data["native_plan_json"]=self.native_plan_json
        if self.producer_chain:
            data["schema"]="tessera.nvidia.lhs_tensor_program.v3"
            data["producer_chain"]=[package_artifact(p).to_dict() for p in self.producer_chain]
        if self.rhs_chain:
            data["schema"]="tessera.nvidia.lhs_tensor_program.v4"
            data["producer_chain"]=[package_artifact(p).to_dict() for p in self.producer_chain]
            data["rhs_chain"]=[package_artifact(p).to_dict() for p in self.rhs_chain]
        return deepcopy({**data,"contract_digest":_digest(data)})


def from_manifest(data):
    from tessera import runtime as rt
    keys={"schema","graph_ir","argument_names","semantics","edge","producer","consumer","contract_digest"}
    if isinstance(data,dict) and data.get("schema") in {"tessera.nvidia.lhs_tensor_program.v2","tessera.nvidia.lhs_tensor_program.v3","tessera.nvidia.lhs_tensor_program.v4"}:
        keys.add("native_plan_json")
        if data["schema"].endswith((".v3",".v4")):
            keys.add("producer_chain")
        if data["schema"].endswith(".v4"):
            keys.add("rhs_chain")
    if (not isinstance(data,dict) or set(data)!=keys or
            data["schema"] not in {"tessera.nvidia.lhs_tensor_program.v1","tessera.nvidia.lhs_tensor_program.v2","tessera.nvidia.lhs_tensor_program.v3","tessera.nvidia.lhs_tensor_program.v4"}):
        raise ValueError("LHS native manifest schema differs")
    data=deepcopy(data)
    if data["schema"]!="tessera.nvidia.lhs_tensor_program.v1" and not isinstance(data["native_plan_json"],str):
        raise ValueError("native LHS plan must be serialized JSON")
    if _digest({k:v for k,v in data.items() if k!="contract_digest"})!=data["contract_digest"]:
        raise ValueError("LHS native manifest integrity differs")
    labels={"producer_input_name","intermediate_name","consumer_input_name","consumer_rhs_name",
            "output_name","m","k","n","dtype"}
    if not isinstance(data["edge"],dict) or set(data["edge"])!=labels or not isinstance(data["argument_names"],list):
        raise ValueError("LHS native edge/argument fields differ")
    packages=[]
    for role in ("producer","consumer"):
        a=rt.RuntimeArtifact.from_dict(data[role])
        if (a.native_image is None or a.launch_descriptor is None
                or data[role].get("artifact_hash")!=a.artifact_hash):
            raise ValueError("LHS native component artifact integrity differs")
        packages.append(native.NVIDIANativePackage(a.tile_ir,a.target_ir,"",a.native_image,a.launch_descriptor))
    chain=[]
    if data["schema"].endswith((".v3",".v4")):
        if not isinstance(data["producer_chain"],list) or not (0 if data["schema"].endswith(".v4") else 2)<=len(data["producer_chain"])<=63:
            raise ValueError("native LHS chain package count differs")
        for item in data["producer_chain"]:
            a=rt.RuntimeArtifact.from_dict(item)
            if a.native_image is None or a.launch_descriptor is None or item.get("artifact_hash") != a.artifact_hash:
                raise ValueError("LHS chain component artifact integrity differs")
            chain.append(native.NVIDIANativePackage(a.tile_ir,a.target_ir,"",a.native_image,a.launch_descriptor))
    right=[]
    if data["schema"].endswith(".v4"):
        if not isinstance(data["rhs_chain"],list) or not 1<=len(data["rhs_chain"])<=63:
            raise ValueError("native RHS chain count differs")
        for item in data["rhs_chain"]:
            a=rt.RuntimeArtifact.from_dict(item)
            if a.native_image is None or a.launch_descriptor is None or item.get("artifact_hash")!=a.artifact_hash:
                raise ValueError("native RHS component artifact integrity differs")
            right.append(native.NVIDIANativePackage(a.tile_ir,a.target_ir,"",a.native_image,a.launch_descriptor))
    result=TracedLhsProgram(native.NVIDIANativeTensorProgram(packages[0],packages[1],**data["edge"]),
                           tuple(data["argument_names"]),data["semantics"],data["graph_ir"],data.get("native_plan_json"),tuple(chain),tuple(right))
    result.validate()
    return result


def runtime_artifact(program):
    from tessera import runtime as rt
    program.validate()
    return rt.RuntimeArtifact(graph_ir=program.graph_ir,metadata={
        "target":"nvidia_sm120","compiler_path":"canonical_nvidia_lhs_program",
        "execution_kind":"native_gpu","runtime_status":"ready","executable":True,
        "native_graph_verified":True,"arg_names":list(program.argument_names),
        "native_program":program.manifest()})
