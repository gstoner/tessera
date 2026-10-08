"""Checked resident normalization RHS edge; semantics live in native Schedule packages."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import hashlib
from typing import Any
from . import nvidia_native as native
from .graph_ir import GraphIRModule


@dataclass(frozen=True)
class NvidiaNormRhsProgram:
    producer: native.NVIDIANativePackage
    consumer: native.NVIDIANativePackage
    source_name: str
    edge_name: str
    lhs_name: str
    rhs_name: str
    output_name: str
    m: int
    k: int
    n: int
    dtype: str

    def validate(self) -> None:
        if self.dtype not in {"fp16", "bf16"} or any(type(x) is not int or x <= 0 for x in (self.m,self.k,self.n)):
            raise ValueError("RHS edge requires positive static half-storage dimensions")
        storage="f16" if self.dtype=="fp16" else "bf16"
        expected_producer=native.SM120_NORM_F16_ABI if storage=="f16" else native.SM120_NORM_BF16_ABI
        expected_consumer=native.SM120_ROW_B_F16_ABI if storage=="f16" else native.SM120_ROW_B_BF16_ABI
        for package,abi in ((self.producer,expected_producer),(self.consumer,expected_consumer)):
            provenance=package.descriptor.provenance
            if (package.image.target!="nvidia_sm120" or package.image.architecture!="sm_120a"
                    or package.descriptor.image_digest!=package.image.image_digest
                    or package.descriptor.abi_id!=abi
                    or provenance.get("route")!="canonical_scheduled_tile_consumer"
                    or provenance.get("storage")!=storage
                    or not provenance.get("schedule_digest")
                    or provenance.get("tile_ir_digest")!=hashlib.sha256(package.tile_ir.encode()).hexdigest()
                    or not package.descriptor.ordering.ordered_submission
                    or "completion" not in package.descriptor.ordering.synchronization):
                raise ValueError("RHS edge requires identity-bound ordered native Schedule packages")
        kind=self.producer.descriptor.provenance.get("kind")
        if (kind not in {"rmsnorm","layernorm"}
                or not self.producer.descriptor.entry_symbol.startswith(f"tessera_tile_norm_{kind}_{storage}_")):
            raise ValueError("RHS producer must be native RMSNorm or LayerNorm")
        if 'role = "b", transpose' not in self.consumer.tile_ir:
            raise ValueError("RHS consumer must carry typed transposed B fragments")
        epilogue=self.consumer.descriptor.provenance.get("epilogue")
        if (self.consumer.descriptor.provenance.get("b_layout")!="row_major"
                or epilogue!={"bias":False,"residual":False,"activation":"none","output":"f32","order":["matmul","bias","activation","residual"]}):
            raise ValueError("RHS consumer must be static unfused row-major with fp32 output")
        bind=native.NVIDIANativeTensorProgram._binding
        shape=native.NVIDIANativeTensorProgram._static_shape
        specifications=((self.producer,self.source_name,"input",(self.k,self.n),self.dtype),
            (self.producer,self.edge_name,"output",(self.k,self.n),self.dtype),
            (self.consumer,self.lhs_name,"input",(self.m,self.k),self.dtype),
            (self.consumer,self.rhs_name,"input",(self.k,self.n),self.dtype),
            (self.consumer,self.output_name,"output",(self.m,self.n),"fp32"))
        if len(self.producer.descriptor.buffers)!=2 or len(self.consumer.descriptor.buffers)!=3:
            raise ValueError("RHS edge has unexpected buffers")
        for package,name,direction,dimensions,dtype in specifications:
            b=bind(package,name,direction)
            if b.dtype!=dtype or b.rank!=2 or b.layout!="row_major" or shape(package,name,2)!=dimensions:
                raise ValueError("RHS edge buffer storage, layout or shape drift")
        if bind(self.producer,self.edge_name,"output").alignment < bind(self.consumer,self.rhs_name,"input").alignment:
            raise ValueError("RHS edge alignment does not satisfy consumer")

    @staticmethod
    def runtime_artifact(package: native.NVIDIANativePackage) -> Any:
        from tessera import runtime as rt
        return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,
            launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)

    def arguments(self,source: Any,lhs: Any,edge: Any,output: Any) -> tuple[dict,dict]:
        return ({self.source_name:source,self.edge_name:edge,"Rows":self.k,"Columns":self.n},
            {self.lhs_name:lhs,self.rhs_name:edge,self.output_name:output,"M":self.m,"N":self.n,"K":self.k})

    def execute_resident(self,source: Any,lhs: Any) -> native.NVIDIANativeTensorProgramResult:
        self.validate()
        import numpy as np
        from tessera import runtime as rt
        from .emit.nvidia_cuda import NvidiaDeviceSession
        dtype=np.dtype(np.float16)
        if self.dtype=="bf16":
            import ml_dtypes
            dtype=np.dtype(ml_dtypes.bfloat16)
        source=np.asarray(source)
        lhs=np.asarray(lhs)
        if source.shape!=(self.k,self.n) or lhs.shape!=(self.m,self.k) or source.dtype!=dtype or lhs.dtype!=dtype:
            raise ValueError("RHS producer input and matmul LHS must match static shapes and storage")
        session=NvidiaDeviceSession()
        try:
            x=session.upload(np.array(source,copy=True,order="C"))
            a=session.upload(np.array(lhs,copy=True,order="C"))
            edge=session.empty((self.k,self.n),dtype)
            out=session.empty((self.m,self.n),np.float32)
            pa,ca=self.arguments(x,a,edge,out)
            receipts=[]
            for package,args in ((self.producer,pa),(self.consumer,ca)):
                receipt=rt.launch(self.runtime_artifact(package),args,stream=session.stream)
                if receipt.get("ok") is not True or receipt.get("execution_kind")!="native_gpu":
                    raise RuntimeError(f"native RHS edge launch failed: {receipt}")
                receipts.append(receipt)
            if session.synchronize()!=0:
                raise RuntimeError("RHS producer/consumer stream failed")
            return native.NVIDIANativeTensorProgramResult(receipts[0],receipts[1],edge,out,session)
        except Exception:
            session.close()
            raise


def package_norm_rhs_matmul(producer: GraphIRModule,consumer: GraphIRModule,*,pipeline_name: str) -> NvidiaNormRhsProgram:
    from .scheduled_kernel import lower_scheduled_kernel
    from .scheduled_matmul import lower_scheduled_matmul
    if not isinstance(producer,GraphIRModule) or not isinstance(consumer,GraphIRModule):
        raise TypeError("RHS edge requires Graph modules")
    # Preserve caller semantics; an explicit conflicting order is never replaced.
    consumer=deepcopy(consumer)
    ops=[op for function in consumer.functions for op in function.body if op.op_name in {"tessera.matmul","tessera.gemm"}]
    if len(ops)!=1 or ops[0].kwargs.get("rhs_storage_order","row_major")!="row_major":
        raise ValueError("RHS edge requires one row-major matmul")
    ops[0].kwargs["rhs_storage_order"]="row_major"
    p=lower_scheduled_kernel(producer,target="nvidia_sm120")
    c=lower_scheduled_matmul(consumer,target="nvidia_sm120")
    if p.family!="norm" or p.kind not in {"rmsnorm","layernorm"} or p.input_shape!=p.output_shape:
        raise ValueError("RHS edge requires shape-preserving normalization")
    program=NvidiaNormRhsProgram(native.package_scheduled_kernel(p,pipeline_name=pipeline_name),
        native.package_scheduled_matmul(c,pipeline_name=pipeline_name),p.input_name,p.output_name,
        c.a_name,c.b_name,c.output_name,c.m,c.k,c.n,p.dtype)
    program.validate()
    return program


@dataclass(frozen=True)
class NvidiaTracedRhsProgram:
    edge: NvidiaNormRhsProgram
    argument_names: tuple[str, ...]
    source_index: int
    lhs_index: int
    graph_ir: str

    def execute_resident(self,*args: Any,**kwargs: Any) -> native.NVIDIANativeTensorProgramResult:
        import inspect
        signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                     for name in self.argument_names])
        bound=signature.bind(*args,**kwargs)
        values=[bound.arguments[name] for name in self.argument_names]
        return self.edge.execute_resident(values[self.source_index],values[self.lhs_index])


def package_traced_norm_rhs(module: GraphIRModule,*,pipeline_name: str) -> NvidiaTracedRhsProgram:
    """Partition one verified semantic edge without reconstructing its operations."""
    from dataclasses import replace
    from .graph_ir import IRArg
    module=deepcopy(module)
    module.to_mlir(target="nvidia_sm120")  # Verify complete semantics before partitioning.
    if len(module.functions)!=1:
        raise ValueError("RHS trace requires one function")
    fn=module.functions[0]
    if len(fn.args)!=2 or len(fn.body)!=2 or len(fn.result_types)!=1:
        raise ValueError("RHS trace requires exactly normalization and matmul with two inputs")
    from .structured_cfg import recover_structured_cfg
    if fn.structured_cfg is not None and fn.structured_cfg.digest != recover_structured_cfg(fn.body).digest:
        raise ValueError("RHS trace CFG does not match semantic operations")
    norm,matmul=fn.body
    names=[arg.name for arg in fn.args]
    if (norm.op_name not in {"tessera.rmsnorm","tessera.layer_norm"} or matmul.op_name not in {"tessera.matmul","tessera.gemm"}
            or not norm.result or not matmul.result or len(norm.operands)!=1 or len(matmul.operands)!=2
            or matmul.operands[1]!="%"+norm.result
            or norm.operands[0] not in ["%"+name for name in names]
            or matmul.operands[0] not in ["%"+name for name in names]
            or matmul.operands[0]==norm.operands[0]
            or fn.return_values!=["%"+matmul.result]
            or any(op.kwargs.get("_region") for op in fn.body)):
        raise ValueError("RHS trace must return matmul(lhs, norm(source))")
    if (set(norm.kwargs) - {"eps", "gamma", "beta"} or norm.kwargs.get("gamma") is not None
            or norm.kwargs.get("beta") is not None
            or norm.numeric_policy is not None):
        raise ValueError("RHS normalization package has no affine gamma/beta, alternate-axis or explicit-policy contract")
    allowed={"output_dtype","activation","bias","residual","epilogue","rhs_storage_order"}
    if (set(matmul.kwargs)-allowed or matmul.numeric_policy is not None
            or matmul.kwargs.get("bias") is not None or matmul.kwargs.get("residual") is not None
            or matmul.kwargs.get("activation") not in {None,"none"}
            or matmul.kwargs.get("epilogue") is not None):
        raise ValueError("RHS matmul has unsupported explicit semantic attributes")
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError("RHS graph verification requires the native compiler")
    native_graph=run_tessera_opt(tool,module.to_mlir(canonical=True,target="nvidia_sm120"),"--verify-each")
    source_index=names.index(norm.operands[0].removeprefix("%"))
    lhs_index=names.index(matmul.operands[0].removeprefix("%"))
    source=fn.args[source_index]
    lhs=fn.args[lhs_index]
    if norm.result_type!=str(source.ir_type) or norm.operand_types!=[str(source.ir_type)]:
        raise ValueError("RHS norm must preserve the typed source")
    producer=replace(fn,name=fn.name+"__rhs",args=[source],body=[norm],
        result_types=[source.ir_type],return_values=["%"+norm.result])
    consumer=replace(fn,name=fn.name+"__matmul",args=[lhs,IRArg(norm.result,source.ir_type)],
        body=[matmul])
    for partition in (producer,consumer):
        partition.structured_cfg=recover_structured_cfg(partition.body)
        partition.fn_attrs={**partition.fn_attrs,
            "tessera.structured_cfg.digest": '"'+partition.structured_cfg.digest+'"',
            "tessera.structured_cfg.blocks": str(len(partition.structured_cfg.blocks))}
    edge=package_norm_rhs_matmul(replace(module,functions=[producer]),
                                   replace(module,functions=[consumer]),pipeline_name=pipeline_name)
    return NvidiaTracedRhsProgram(edge,tuple(names),source_index,lhs_index,native_graph)


def validate_traced_program(program: NvidiaTracedRhsProgram) -> None:
    """Check portable argument/shape/epsilon lineage without invoking a compiler."""
    import re
    import struct
    program.edge.validate()
    names=program.argument_names
    if (len(names)!=2 or len(set(names))!=2
            or any(not isinstance(name,str) or not name.isidentifier() for name in names)
            or type(program.source_index) is not int or type(program.lhs_index) is not int
            or {program.source_index,program.lhs_index}!={0,1}):
        raise ValueError("native RHS program has an invalid frontend argument ABI")
    graph=program.graph_ir
    norms=re.findall(r"(%\w+) = tessera\.(?:rmsnorm|layer_norm) (%arg[01])\s",graph)
    matmuls=re.findall(r"(%\w+) = tessera\.(?:matmul|gemm) (%arg[01]), (%\w+)\s",graph)
    if (len(norms)!=1 or len(matmuls)!=1 or norms[0][1]!=f"%arg{program.source_index}"
            or matmuls[0][1]!=f"%arg{program.lhs_index}" or matmuls[0][2]!=norms[0][0]
            or not re.search(r"\breturn "+re.escape(matmuls[0][0])+r"\s*:",graph)):
        raise ValueError("native RHS program operand lineage differs from its verified Graph")
    norm_line=next(line for line in graph.splitlines() if " = tessera.rmsnorm " in line or " = tessera.layer_norm " in line)
    graph_kind="rmsnorm" if " = tessera.rmsnorm " in norm_line else "layernorm"
    if graph_kind != program.edge.producer.descriptor.provenance.get("kind"):
        raise ValueError("native RHS program normalization kind differs from its producer")
    if re.search(r"\b(gamma|beta|axis|numeric_policy)\s*=",norm_line):
        raise ValueError("native RHS program Graph has unsupported norm semantics")
    matrix_line=next(line for line in graph.splitlines() if " = tessera.matmul " in line or " = tessera.gemm " in line)
    if (re.search(r"\b(alpha|beta|numeric_policy|transpose_a|transpose_b|scale_layout|epilogue|bias|residual)\s*=",matrix_line)
            or re.search(r'\bactivation = "(?!none")[^"]+"',matrix_line)):
        raise ValueError("native RHS program Graph has unsupported matmul semantics")
    edge=program.edge
    elem="f16" if edge.dtype=="fp16" else "bf16"
    arguments=re.findall(r"%arg([01]): tensor<([0-9]+)x([0-9]+)x(f16|bf16)>",graph)
    expected={program.source_index:(edge.k,edge.n,elem),program.lhs_index:(edge.m,edge.k,elem)}
    if len(arguments)!=2 or any(expected[int(index)]!=(int(rows),int(cols),dtype) for index,rows,cols,dtype in arguments):
        raise ValueError("native RHS program Graph argument shapes/storage differ from its packages")
    if (len(re.findall(r"\bfunc\.func\s",graph))!=1
            or len(re.findall(r"^\s*%\w+\s*=",graph,re.M))!=2
            or not re.search(r"\breturn "+re.escape(matmuls[0][0])+rf" : tensor<{edge.m}x{edge.n}xf32>",graph)):
        raise ValueError("native RHS program needs the declared two-operation fp32 result Graph")
    epsilon=re.search(r"\beps = ([0-9.eE+-]+) : f64",graph)
    eps32=struct.unpack("f",struct.pack("f",float(epsilon[1]) if epsilon else 1e-5))[0]
    if eps32!=edge.producer.descriptor.provenance.get("epsilon"):
        raise ValueError("native RHS program epsilon differs from its producer")


def rhs_program_manifest(program: NvidiaTracedRhsProgram) -> dict:
    import json
    validate_traced_program(program)
    edge=program.edge
    fields=("source_name","edge_name","lhs_name","rhs_name","output_name","m","k","n","dtype")
    manifest=dict(schema="tessera.nvidia.norm_rhs_program.v2",graph_ir=program.graph_ir,
        graph_digest=hashlib.sha256(program.graph_ir.encode()).hexdigest(),
        argument_names=list(program.argument_names),source_index=program.source_index,lhs_index=program.lhs_index,
        edge={name:getattr(edge,name) for name in fields},
        producer=edge.runtime_artifact(edge.producer).to_dict(),
        consumer=edge.runtime_artifact(edge.consumer).to_dict())
    manifest["contract_digest"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return manifest


def rhs_program_from_manifest(manifest: dict) -> NvidiaTracedRhsProgram:
    import json
    from tessera import runtime as rt
    keys={"schema","graph_ir","graph_digest","argument_names","source_index","lhs_index",
          "edge","producer","consumer","contract_digest"}
    if not isinstance(manifest,dict) or set(manifest)!=keys or manifest["schema"] not in {"tessera.nvidia.rmsnorm_rhs_program.v1","tessera.nvidia.norm_rhs_program.v2"}:
        raise ValueError("invalid native RHS program schema")
    body={key:value for key,value in manifest.items() if key!="contract_digest"}
    if hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()!=manifest["contract_digest"]:
        raise ValueError("native RHS program contract digest mismatch")
    graph=manifest["graph_ir"]
    if not isinstance(graph,str) or hashlib.sha256(graph.encode()).hexdigest()!=manifest["graph_digest"]:
        raise ValueError("native RHS program Graph digest mismatch")
    fields={"source_name","edge_name","lhs_name","rhs_name","output_name","m","k","n","dtype"}
    if not isinstance(manifest["edge"],dict) or set(manifest["edge"])!=fields:
        raise ValueError("invalid native RHS program edge schema")
    packages=[]
    for role in ("producer","consumer"):
        artifact=rt.RuntimeArtifact.from_dict(manifest[role])
        if artifact.native_image is None or artifact.launch_descriptor is None:
            raise ValueError("native RHS program component needs an image and checked descriptor")
        if manifest[role].get("artifact_hash")!=artifact.artifact_hash:
            raise ValueError("native RHS program requires complete component artifact hashes")
        packages.append(native.NVIDIANativePackage(artifact.tile_ir,artifact.target_ir,"",
            artifact.native_image,artifact.launch_descriptor))
    edge=NvidiaNormRhsProgram(packages[0],packages[1],**manifest["edge"])
    if not isinstance(manifest["argument_names"],list):
        raise ValueError("native RHS program argument names must be an array")
    program=NvidiaTracedRhsProgram(edge,tuple(manifest["argument_names"]),
        manifest["source_index"],manifest["lhs_index"],graph)
    validate_traced_program(program)
    if manifest["schema"]=="tessera.nvidia.rmsnorm_rhs_program.v1" and edge.producer.descriptor.provenance.get("kind")!="rmsnorm":
        raise ValueError("legacy RHS schema requires RMSNorm")
    return program


def rhs_runtime_artifact(program: NvidiaTracedRhsProgram) -> Any:
    from tessera import runtime as rt
    return rt.RuntimeArtifact(graph_ir=program.graph_ir,
        metadata={"target":"nvidia_sm120","compiler_path":"canonical_nvidia_rhs_program",
            "execution_kind":"native_gpu","runtime_status":"ready","executable":True,
            "native_graph_verified":True,"arg_names":list(program.argument_names),
            "native_program":rhs_program_manifest(program)})


# Preserve the original public RMSNorm-only entry points. New traced dispatch
# selects the normalization contract explicitly and binds it in the manifest.
NvidiaRmsnormRhsProgram = NvidiaNormRhsProgram


def package_rmsnorm_rhs_matmul(producer, consumer, *, pipeline_name):
    if any(op.op_name != "tessera.rmsnorm" for fn in producer.functions for op in fn.body):
        raise ValueError("RMSNorm RHS entry point requires RMSNorm")
    return package_norm_rhs_matmul(producer, consumer, pipeline_name=pipeline_name)


def package_traced_rmsnorm_rhs(module, *, pipeline_name):
    if any(fn.body and fn.body[0].op_name != "tessera.rmsnorm" for fn in module.functions):
        raise ValueError("RMSNorm RHS entry point requires RMSNorm")
    return package_traced_norm_rhs(module, pipeline_name=pipeline_name)
