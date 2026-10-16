"""Canonical family adapter for compiler-owned SM120 saved-LSE reverse."""
from __future__ import annotations
from collections import OrderedDict
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.runtime import RuntimeArtifact
import hashlib
from pathlib import Path
import re
import os
import threading

_cache: OrderedDict[tuple[str, tuple[int, ...], str, int, int, int, int, str],
                    tuple[str, str, str, dict[str, str], RuntimeArtifact]] = OrderedDict()
_lock=threading.RLock()
_LIMIT=96
_pid=os.getpid()

def execute_unprepared(metadata,args):
    import numpy as np
    from .native_attention_program import NativeAttentionVJPProgram
    from .resident_attention import checkpoint_shapes
    from .emit.nvidia_cuda import NvidiaDeviceSession,CudaOwnedDeviceBuffer
    program=NativeAttentionVJPProgram.from_json(
        metadata["program_json"],expected_digest=metadata["program_digest"])
    dims,physical=checkpoint_shapes(program.pair)
    b,hq,hkv,sq,sk,_,_=dims
    names=[f"primal_{i}" for i in range(len(program.input_indices))]+["cotangent"]
    if metadata.get("arg_names")!=names or len(args)!=len(names):
        raise ValueError("attention VJP launch names/arity differ from native contract")
    shapes=list(physical[:3])
    if len(program.input_indices)==4:
        shapes.append(tuple(program.pair.forward.descriptor.provenance.get("bias_shape",())) or (b,hq,sq,sk))
    frontend=[None]*len(shapes)
    for role,index in enumerate(program.input_indices):frontend[index]=shapes[role]
    expected=(*frontend,physical[3])
    values=tuple(np.asarray(x) for x in args)
    if any(x.dtype!=np.float32 or x.shape!=shape
           for x,shape in zip(values,expected,strict=True)):
        raise ValueError("attention VJP host storage differs from native contract")
    # Upload ownership and stream completion precede the private saved state.
    # Both products consume native packages; no Graph reconstruction at runtime.
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in values]
        with program.capture(*resident[:-1]) as frame:
            outputs=frame.backward(resident[-1])
            result=[]
            for value in outputs:
                interface=value.__cuda_array_interface__
                shape=tuple(interface["shape"])
                view=CudaOwnedDeviceBuffer(session,int(interface["data"][0]),shape,np.float32,
                                          int(np.prod(shape))*4,owns=False)
                result.append(session.download(view))
            return tuple(result)

def execute_family(*,source,target,ordered_inputs,arg_names,source_arg_names,
                   out_cotangents,wrt_names,declaration,source_graph_ir):
    if os.getpid()!=_pid:
        raise ValueError("attention reverse planner cannot cross fork")
    import numpy as np
    from .native_attention_program import compile_attention_vjp_program
    from .native_vjp_plugins import NativeVJPResult
    from .scheduled_matmul import find_tessera_opt
    from tessera import runtime as rt
    if target!="nvidia_sm120" or source.op_name!="tessera.flash_attn" or not source_graph_ir:
        raise ValueError("native attention reverse requires traced SM120 flash_attn")
    if (len(arg_names)!=len(source_arg_names) or len(arg_names)!=len(ordered_inputs)
            or len(set(arg_names))!=len(arg_names) or len(set(source_arg_names))!=len(source_arg_names)):
        raise ValueError("attention VJP frontend argument arity/identity disagrees")
    cots=tuple(out_cotangents) if isinstance(out_cotangents,(tuple,list)) else (out_cotangents,)
    seeded = source.kwargs.get("lse_checkpoint") == "saved"
    if len(cots) != 1+int(seeded):
        raise ValueError("attention VJP cotangents must match the native primary/LSE result roles")
    roots=(*ordered_inputs,*cots)
    resident=any(hasattr(value,"__cuda_array_interface__") for value in roots)
    if resident:
        from .resident_nvidia_tensor import cuda_frontend_specs
        if not all(hasattr(value,"__cuda_array_interface__") for value in roots):
            raise ValueError("SM120 native attention reverse requires all resident roots")
        specs=cuda_frontend_specs(roots,ranks=(3,4))
        if any(dtype.name not in ("float32", "float16", "bfloat16") for _,dtype in specs):
            raise ValueError("SM120 native attention reverse requires f16/bf16/f32 resident roots")
        values=tuple(roots)
    else:
        values=tuple(np.asarray(x) for x in roots)
        if any(x.dtype.name not in ("float32", "float16", "bfloat16") for x in values):
            raise ValueError("SM120 native attention reverse requires f16/bf16/f32 host tensors")
    active=tuple(arg_names.index(name) for name in wrt_names)
    compiler=find_tessera_opt()
    if compiler is None:raise ValueError("attention VJP requires the selected native compiler")
    st=Path(compiler).stat()
    row_policy=os.environ.get("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","inline")
    if row_policy not in ("inline","prepass"):
        raise ValueError("attention row correction policy must be inline or prepass")
    key=(source_graph_ir,active,str(Path(compiler).resolve()),st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,row_policy)
    with _lock:
        cached=_cache.get(key)
        if cached is None:
            graph=re.sub(r'=\s+(tessera\.[A-Za-z0-9_.]+)\(',r'= "\1"(',source_graph_ir)
            if row_policy=="prepass":
                program=compile_attention_vjp_program(graph,active,compiler=compiler,compact_gradients=True,row_correction=True)
            else:
                program=compile_attention_vjp_program(graph,active,compiler=compiler,compact_gradients=True)
            members=getattr(program,"members",(program.pair.forward,program.pair.backward))
            text=program.to_json()
            pin=program.program_digest
            proof=dict(
                schedule_artifact_hash=hashlib.sha256("".join(
                    x.descriptor.provenance["schedule_digest"]
                    for x in members).encode()).hexdigest(),
                tile_program_digest=hashlib.sha256(
                    "".join(x.tile_ir for x in members).encode()).hexdigest(),
                state_lineage_digest=program.pair.contract_digest,
            )
            # The compiler owns the root/result roles; retain only immutable
            # compiled facts, never root values or invocation results.
            names=[f"primal_{i}" for i in range(len(program.input_indices))]+["cotangent"]
            if program.pair.backward.descriptor.provenance.get("lse_cotangent",False):
                names.append("lse_cotangent")
            prototype=rt.RuntimeArtifact(graph_ir=source_graph_ir,target_ir=program.pair.backward.target_ir,
                metadata=dict(target=target,compiler_path="nvidia_sm120_attention_vjp_compiled",
                              execution_kind="native_gpu",execution_mode="cuda_runtime",executable=True,
                              arg_names=names,program_json=text,program_digest=pin))
            # RuntimeArtifact already memoizes its canonical identity. Keep the
            # prototype private and detach all mutable metadata before dispatch.
            _ = prototype.artifact_hash
            cached=(text,pin,program.pair.backward.target_ir,proof,prototype)
            _cache[key]=cached
            while len(_cache)>_LIMIT:_cache.popitem(last=False)
        _cache.move_to_end(key)
    text,pin,target_ir,proof,prototype=cached
    names=[f"primal_{i}" for i in range(len(ordered_inputs))]+["cotangent"]+(["lse_cotangent"] if seeded else [])
    if names!=(prototype.metadata or {}).get("arg_names"):
        raise ValueError("attention VJP frontend roles disagree with native checkpoint")
    artifact=copy.deepcopy(prototype)
    receipt=rt.launch(artifact,values)
    if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
        raise RuntimeError(f"native attention VJP launch failed: {receipt}")
    return NativeVJPResult(tuple(receipt["output"]),dict(
        compiler_path="nvidia_sm120_attention_vjp_compiled",execution_kind="native_gpu",
        execution_mode="cuda_runtime",evidence_target=target,implementation="family_plugin",
        family=declaration.family,graph_consumer=source.op_name,
        schedule_consumer=declaration.schedule_consumer,tile_consumer=declaration.tile_consumer,
        target_consumer=declaration.target_consumers[target],residual_policy="save",
        source_graph_ir_digest=hashlib.sha256(source_graph_ir.encode()).hexdigest(),
        **proof,program_digest=pin,
        artifact_hash=artifact.artifact_hash,frontend_authority="tracer",
        host_preparation="native_ordered_resident_snapshot" if resident else "compact_host_frame",
        physical_attestation=receipt.get("physical_attestation"),
        kernel_elapsed_ms=receipt.get("kernel_elapsed_ms")),artifact)


def execute(metadata,args):
    from .prepared_attention_vjp import execute as native_execute
    return native_execute(metadata,args)
