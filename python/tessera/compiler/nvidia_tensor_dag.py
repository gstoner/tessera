"""Compiler-owned two-operand tensor DAG packaging; no Graph/Tile reconstruction."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import struct

from .native_sm120_tensor_program import export_native_sm120_tensor_graph, validate_native_tensor_plan
from .scheduled_matmul import find_tessera_opt


def roots(module):
    if len(module.functions)!=1:
        return None
    fn=module.functions[0]
    if not 3<=len(fn.body)<=64:
        return None
    from .nvidia_tensor_lhs import PRODUCERS
    origins={"%"+arg.name:"%"+arg.name for arg in fn.args}
    for op in fn.body[:-1]:
        if (op.op_name not in PRODUCERS or len(op.operands)!=1 or not op.result or
                op.operands[0] not in origins):
            return None
        origins["%"+op.result]=origins[op.operands[0]]
    consumer=fn.body[-1]
    if (consumer.op_name not in {"tessera.matmul","tessera.gemm"} or len(consumer.operands)<2 or
            any(v not in origins for v in consumer.operands[:2]) or
            any(v in {"%"+arg.name for arg in fn.args} for v in consumer.operands[:2]) or
            origins[consumer.operands[0]]==origins[consumer.operands[1]]):
        return None
    return origins


def candidate(module):
    return roots(module) is not None


def package(module, *, dynamic_axes=(), shape_bounds=None):
    from . import nvidia_native as native
    from .nvidia_tensor_lhs import TracedLhsProgram, _checked_semantics
    module=deepcopy(module)
    origins=roots(module)
    if origins is None:
        raise ValueError("tensor DAG requires producers on both independent matmul operands")
    fn=module.functions[0];consumer=fn.body[-1]
    if (set(module.module_attrs)-{"tessera.ir.version","tessera.frontend.authority","tessera.target","tessera.arch"} or
            set(fn.fn_attrs)-{"tessera.frontend.authority","tessera.structured_cfg.schema",
                             "tessera.structured_cfg.digest","tessera.structured_cfg.blocks"} or
            any(arg.effect or arg.shard_spec or arg.dim_names or arg.layout or arg.model_parameter or
                arg.model_parameter_bytes_bound is not None for arg in fn.args) or
            any(op.numeric_policy is not None or op.attrs not in (None,"") for op in fn.body) or
            len(fn.result_types)!=1 or fn.return_values!=["%"+str(consumer.result)]):
        raise ValueError("tensor DAG metadata/return requires an explicit native contract")
    for key,value in (("tessera.target",'"nvidia_sm120"'),("tessera.arch",'"sm_120"')):
        if module.module_attrs.get(key,value)!=value:
            raise ValueError("tensor DAG architecture differs")
        module.module_attrs[key]=value
    from .structured_cfg import recover_structured_cfg
    if fn.structured_cfg is not None and fn.structured_cfg.digest!=recover_structured_cfg(fn.body).digest:
        raise ValueError("tensor DAG CFG differs")
    names=[arg.name for arg in fn.args]
    role_names=["source","rhs"]+(["bias"] if consumer.kwargs.get("bias") is not None else [])+(
        ["residual"] if consumer.kwargs.get("residual") is not None else [])
    external=[origins[v] for v in consumer.operands[:2]]+consumer.operands[2:]
    if len(external)!=len(role_names) or any(v not in origins or origins[v]!=v for v in external):
        raise ValueError("tensor DAG external roles differ")
    roles={role:names.index(value[1:]) for role,value in zip(role_names,external,strict=True)}
    # Physical materialization is row-major. An authored conflicting request
    # requires a native layout transform and must never be silently replaced.
    ca=deepcopy(consumer.kwargs)
    if ca.get("rhs_storage_order","row_major")!="row_major":
        raise ValueError("computed RHS requires row-major native materialization")
    ca["rhs_storage_order"]="row_major"
    indices=[[i for i,op in enumerate(fn.body[:-1]) if origins["%"+op.result]==external[side]]
             for side in (0,1)]
    policies=[[{"producer":fn.body[i].op_name,"producer_attrs":deepcopy(fn.body[i].kwargs)}
               for i in chain] for chain in indices]
    sem={**policies[0][0],"consumer":consumer.op_name,"consumer_attrs":ca,"roles":roles,
         "rhs_chain":policies[1]}
    if len(indices[0])>1:sem["producer_chain"]=policies[0]
    _checked_semantics(sem)
    lhs=fn.args[roles["source"]].ir_type;rhs=fn.args[roles["rhs"]].ir_type
    m,k=map(int,lhs.shape);rk,n=map(int,rhs.shape)
    if lhs.dtype not in {"fp16","bf16"} or lhs.dtype!=rhs.dtype or rk!=k:
        raise ValueError("tensor DAG input storage/shape differs")
    if dynamic_axes:
        active={"M":m,"N":n,"K":k}
        capacities={axis:(shape_bounds or {}).get(axis,active[axis]) for axis in dynamic_axes}
        if any(bound<active[axis] for axis,bound in capacities.items()):
            raise ValueError("tensor DAG trace exceeds bounds")
        module.module_attrs["tessera.native.sm120_tensor_bounds"]="{"+", ".join(
            f"{axis} = {bound} : i64" for axis,bound in capacities.items())+"}"
    tool=find_tessera_opt()
    if tool is None:raise RuntimeError("tensor DAG needs matching native compiler")
    projected=export_native_sm120_tensor_graph(module.to_mlir(canonical=True,target="nvidia_sm120"),tool=tool)
    plan=validate_native_tensor_plan(projected.plan_json)
    if plan["role_indices"]!=[roles[role] for role in role_names]:
        raise ValueError("tensor DAG native roles differ")
    artifacts=projected.scheduled_members()
    cap=plan.get("shape_bounds",[m,n,k])
    edge=native.package_scheduled_tensor_matmul(artifacts[indices[0][-1]],artifacts[-1],
        pipeline_name="tessera-nvidia-pipeline-sm120",
        dynamic_m_bound=cap[0] if "M" in dynamic_axes else None,
        dynamic_n_bound=cap[1] if "N" in dynamic_axes else None,
        dynamic_k_bound=cap[2] if "K" in dynamic_axes else None)
    digest=hashlib.sha256(projected.plan_json.encode()).hexdigest()
    def bind(p):
        return replace(p,descriptor=replace(p.descriptor,provenance={
            **p.descriptor.provenance,"native_tensor_program_digest":digest}))
    edge=replace(edge,producer=bind(edge.producer),consumer=bind(edge.consumer))
    chains=[]
    for side,chain in enumerate(indices):
        packages=[]
        for index in chain:
            if side==0 and index==chain[-1]:
                packages.append(edge.producer);continue
            artifact=artifacts[index]
            p=native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
            axes=("M","K") if side==0 else ("K","N")
            for dimension,axis in enumerate(axes):
                if axis in dynamic_axes:
                    transform=native._with_dynamic_m_capacity if dimension==0 else native._with_dynamic_k_capacity
                    p=transform(p,input_name=artifact.input_name,output_name=artifact.output_name,
                                bound=artifact.input_shape[dimension])
            packages.append(bind(p))
        chains.append(tuple(packages))
    program=TracedLhsProgram(edge,tuple(names),sem,plan["source_graph_ir"],projected.plan_json,
                            chains[0] if len(chains[0])>1 else (),chains[1])
    program.validate()
    return program


def validate(program):
    from . import nvidia_native as native
    from .nvidia_tensor_lhs import _checked_semantics, PRODUCERS
    plan=validate_native_tensor_plan(program.native_plan_json)
    _,pa,ca,roles=_checked_semantics(program.semantics)
    left=program.producer_chain or (program.edge.producer,)
    right=program.rhs_chain
    policies=[program.semantics.get("producer_chain",[{"producer":program.semantics["producer"],"producer_attrs":pa}]),
              program.semantics.get("rhs_chain",[])]
    if (plan["schema"] not in {"tessera.native.sm120_tensor_program.v5","tessera.native.sm120_tensor_program.v6"} or
            not right or len(left)!=len(policies[0]) or len(right)!=len(policies[1]) or
            left[-1]!=program.edge.producer or len(plan["steps"])!=len(left)+len(right)+1):
        raise ValueError("tensor DAG component count/schema differs")
    role_names=["source","rhs"]+(["bias"] if "bias" in roles else [])+(["residual"] if "residual" in roles else [])
    if (plan["role_indices"]!=[roles[r] for r in role_names] or plan["source_graph_ir"]!=program.graph_ir or
            len(program.argument_names)!=len(roles) or len(set(program.argument_names))!=len(roles)):
        raise ValueError("tensor DAG argument/Graph certificate differs")
    for name in program.argument_names:inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
    axes=tuple(axis for axis,enabled in (("M",program.edge.dynamic_m),("N",program.edge.dynamic_n),
                                        ("K",program.edge.dynamic_k)) if enabled)
    if tuple(plan.get("dynamic_axes",[]))!=axes or axes and plan["shape_bounds"]!=[
            program.edge.m,program.edge.n,program.edge.k]:
        raise ValueError("tensor DAG dynamic bounds differ")
    origins={i:i for i in range(len(roles))}
    offsets=[0,0];ordered=[]
    for step in plan["steps"][:-1]:
        origin=origins[step["inputs"][0]]
        side=0 if origin==roles["source"] else 1
        chain=(left,right)[side];index=offsets[side]
        if index>=len(chain):raise ValueError("tensor DAG producer root differs")
        p=chain[index];policy=policies[side][index];offsets[side]+=1
        origins[step["outputs"][0]]=origin
        pp=p.descriptor.provenance
        if step["operation"]!=policy["producer"] or pp.get("kind")!=PRODUCERS[policy["producer"]]:
            raise ValueError("tensor DAG producer semantic kind differs")
        if pp["kind"]!="softmax" and pp.get("epsilon")!=struct.unpack("f",struct.pack(
                "f",policy["producer_attrs"].get("eps",1e-5)))[0]:
            raise ValueError("tensor DAG producer epsilon differs")
        from .native_artifact import LaunchGeometry, WorkspaceRequirement
        norm=pp["kind"]!="softmax";storage="f16" if program.edge.dtype=="fp16" else "bf16"
        abi=(native.SM120_NORM_F16_ABI if storage=="f16" else native.SM120_NORM_BF16_ABI) if norm else (
            native.SM120_SOFTMAX_F16_ABI if storage=="f16" else native.SM120_SOFTMAX_BF16_ABI)
        schedule=pp.get("schedule")
        geometry=("sm120_norm_"+str(schedule)+"_rows" if norm else
                  "sm120_softmax_cooperative_128_rows" if schedule=="cooperative_128" else
                  "sm120_softmax_thread_per_row_128")
        scalar_names=("Rows","Columns") if norm else ("Rows","K")
        if (p.descriptor.abi_id!=abi or pp.get("route")!="canonical_scheduled_tile_consumer" or
                schedule not in {"serial","cooperative_128"} or
                p.descriptor.geometry!=LaunchGeometry(policy=geometry) or
                p.descriptor.workspace!=WorkspaceRequirement() or p.descriptor.dynamic_local_memory_bytes or
                p.descriptor.dynamic_local_memory_expression or p.descriptor.ordering!=program.edge.consumer.descriptor.ordering or
                tuple(s.name for s in sorted(p.descriptor.scalars,key=lambda s:s.ordinal))!=scalar_names or
                any(s.dtype!="int64" for s in p.descriptor.scalars)):
            raise ValueError("tensor DAG producer physical ABI differs")
        expected=(program.edge.m,program.edge.k) if side==0 else (program.edge.k,program.edge.n)
        varying=(program.edge.dynamic_m,program.edge.dynamic_k) if side==0 else (
            program.edge.dynamic_k,program.edge.dynamic_n)
        for name,direction in (("source","input"),("edge","output")):
            b=native.NVIDIANativeTensorProgram._binding(p,name,direction)
            shape,dynamic=native.NVIDIANativeTensorProgram._shape_bound(p,name,2)
            if b.dtype!=program.edge.dtype or tuple(shape)!=expected or tuple(dynamic)!=varying:
                raise ValueError("tensor DAG producer capacity/storage differs")
        ordered.append(p)
    if offsets!=[len(left),len(right)]:
        raise ValueError("tensor DAG omitted a component")
    ordered.append(program.edge.consumer)
    digest=hashlib.sha256(program.native_plan_json.encode()).hexdigest()
    for index,p in enumerate(ordered):
        p.descriptor.validate_image(p.image)
        pp=p.descriptor.provenance
        if (p.image.target!="nvidia_sm120" or p.image.architecture!="sm_120a" or
                hashlib.sha256(p.target_ir.encode()).hexdigest()!=p.image.target_ir_digest or
                pp.get("native_tensor_program_digest")!=digest or
                pp.get("graph_ir_digest")!=hashlib.sha256(plan["member_graphs"][index].encode()).hexdigest() or
                pp.get("tile_ir_digest")!=hashlib.sha256(p.tile_ir.encode()).hexdigest()):
            raise ValueError("tensor DAG native component ancestry differs")
    cp=program.edge.consumer.descriptor.provenance
    if (cp.get("b_layout")!="row_major" or ca.get("rhs_storage_order")!="row_major" or
            cp.get("epilogue")!={"bias":"bias" in roles,"residual":"residual" in roles,
                               "activation":ca.get("activation","none"),"order":["matmul","bias","activation","residual"],
                               "output":"f16" if ca.get("output_dtype","fp32")=="fp16" else "f32"}):
        raise ValueError("tensor DAG consumer policy differs")


class ResidentDagResult:
    """Device output and native private edges share one explicit lifetime."""
    def __init__(self,session,owner,output,receipts):
        self.device_session=session;self.owner=owner;self.output=output;self.intermediate=None
        self.producer_receipt={**receipts[0],"component_receipts":receipts[:-1]}
        self.consumer_receipt=receipts[-1];self.closed=False

    def close(self):
        if not self.closed:
            self.owner.close();self.device_session.close();self.closed=True

    def __enter__(self):return self

    def __exit__(self,*args):self.close()


def _checked_device_arguments(program,ordered):
    """Project only CUDA root metadata; no host tensor coercion or computation."""
    import math
    import numpy as np
    from .resident_nvidia_tensor import _cuda_metadata_dtype
    roles=program.semantics["roles"]
    selected=[ordered[roles[name]] for name in ("source","rhs","bias","residual") if name in roles]
    interfaces=[value.__cuda_array_interface__ for value in selected]
    expected_names=["float16" if program.edge.dtype=="fp16" else "bfloat16"]*2+["float32"]*(len(selected)-2)
    shapes=[]
    for value,interface,expected in zip(selected,interfaces,expected_names,strict=True):
        if not isinstance(interface,dict) or interface.get("version")!=3:
            raise ValueError("ordered resident CUDA roots require version-three metadata")
        shape=interface.get("shape")
        if (not isinstance(shape,(tuple,list)) or not 1<=len(shape)<=2 or
                any(type(extent) is not int or not 0<extent<2**63 for extent in shape)):
            raise ValueError("ordered resident CUDA root shape is malformed")
        try:
            dtype=_cuda_metadata_dtype(value,interface)
            physical=np.dtype(interface.get("typestr"))
        except (TypeError,KeyError) as error:
            raise ValueError("ordered resident CUDA root dtype metadata is malformed") from error
        if (dtype.name!=expected or
                (physical!=dtype and not (expected=="bfloat16" and physical==np.dtype("V2")))):
            raise ValueError("ordered resident CUDA root storage differs")
        if math.prod(shape)*dtype.itemsize>=2**63:
            raise ValueError("ordered resident CUDA root byte capacity overflows")
        data=interface.get("data")
        if (not isinstance(data,(tuple,list)) or len(data)!=2 or
                type(data[0]) is not int or not 0<data[0]<2**64 or type(data[1]) is not bool):
            raise ValueError("ordered resident CUDA root pointer/readonly metadata differs")
        strides=interface.get("strides")
        dense=(dtype.itemsize,) if len(shape)==1 else (shape[1]*dtype.itemsize,dtype.itemsize)
        if strides is not None and (
                not isinstance(strides,(tuple,list)) or len(strides)!=len(shape) or
                any(type(stride) is not int or stride<=0 or stride%dtype.itemsize for stride in strides) or
                any(extent>1 and actual!=wanted for extent,actual,wanted in zip(shape,strides,dense,strict=True))):
            raise ValueError("ordered resident CUDA roots require compact row-major strides")
        stream=interface.get("stream")
        if type(stream) is not int or not 0<stream<2**64:
            raise ValueError("ordered resident CUDA roots require explicit producer streams")
        shapes.append(tuple(shape))
    if len(shapes[0])!=2 or len(shapes[1])!=2 or shapes[0][1]!=shapes[1][0]:
        raise ValueError("ordered resident CUDA matmul root dimensions differ")
    m,k=shapes[0];n=shapes[1][1]
    for active,bound,dynamic in zip((m,n,k),(program.edge.m,program.edge.n,program.edge.k),
                                  (program.edge.dynamic_m,program.edge.dynamic_n,program.edge.dynamic_k),strict=True):
        if active>bound or (not dynamic and active!=bound):
            raise ValueError("ordered resident CUDA extent outside compiled capacity")
    index=2
    if "bias" in roles:
        if shapes[index]!=(n,):raise ValueError("ordered resident CUDA bias shape differs")
        index+=1
    if "residual" in roles and shapes[index]!=(m,n):
        raise ValueError("ordered resident CUDA residual shape differs")
    return selected,(m,n)


def execute_resident(program,args,kwargs):
    from .prepared_nvidia_lhs import PreparedLhsCall, _portable_arrays
    from .resident_nvidia_tensor import resident_views
    from .emit.nvidia_cuda import NvidiaDeviceSession
    import ctypes as ct
    import numpy as np
    signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                for name in program.argument_names])
    bound=signature.bind(*args,**kwargs)
    supplied=[bound.arguments[name] for name in program.argument_names]
    cuda=[getattr(value,"__cuda_array_interface__",None) is not None for value in supplied]
    borrowed=any(cuda)
    if borrowed and not all(cuda):
        raise ValueError("ordered resident DAG requires all roots resident")
    ordered=[]
    roots=[]
    if borrowed:
        roots,shape=_checked_device_arguments(program,supplied)
    else:
        ordered=_portable_arrays(program,bound.arguments)
        shape=(ordered[program.semantics["roles"]["source"]].shape[0],
               ordered[program.semantics["roles"]["rhs"]].shape[1])
    owner=PreparedLhsCall(program);session=None
    try:
        if not hasattr(owner.lib,"tessera_nvidia_matmul_invoke_dag_resident"):
            raise RuntimeError("native owned-edge resident DAG API unavailable")
        session=NvidiaDeviceSession()
        # These allocations are frontend roots and returned output only.
        # Every intermediate and ping-pong capacity is allocated by C++.
        values=list(roots) if borrowed else [
            session.upload(np.asarray(ordered[index],order="C")) for index in owner.input_positions]
        output=session.empty(shape,owner.output_dtype);values.append(output)
        from .prepared_nvidia_matmul import HostView
        if borrowed:
            owner.invoke_resident(supplied,output,stream=session.stream)
        else:
            views=resident_views(values,session.stream,writable_from=len(values)-1)
            fn=owner.lib.tessera_nvidia_matmul_invoke_dag_resident
            fn.argtypes=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,ct.c_void_p];fn.restype=ct.c_int
            owner._check(fn(owner.handle,views,len(values),ct.c_void_p(session.stream)))
        binding=("prepared_cpp_ordered_resident_tensor_dag" if borrowed
                 else "prepared_cpp_owned_resident_tensor_dag")
        receipts=tuple({**r,"native_call_binding":binding}
                       for r in owner.component_receipts)
        return ResidentDagResult(session,owner,output,receipts)
    except Exception:
        owner.close()
        if session is not None:session.close()
        raise
