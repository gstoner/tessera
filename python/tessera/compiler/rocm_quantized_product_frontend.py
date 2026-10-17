"""Authored frontend and portable runtime binding for native quantized products."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import inspect
import json
import os

from .rocm_block_quant_native import _NAMES
from .rocm_quantized_product_program import NativeQuantizedProductProgram,package_quantized_product_graph,_export,_frame_shape
from .scheduled_matmul import find_tessera_opt

_SCHEMA="tessera.rocm.quantized_product_frontend.v1"
_EXPANDED_SCHEMA="tessera.rocm.quantized_product_frontend.v2"
_OWNER_PID=os.getpid()


def requests_quantized_product(module):
    if module is None or len(module.functions)!=1:
        return False
    fn=module.functions[0]
    names=[op.op_name for op in fn.body if op.op_name!="arith.constant"]
    return (len(fn.result_types)==1 and (
        (len(fn.args)==3 and names==[*_NAMES,"tessera.scaled_matmul"])
        or (len(fn.args)==4 and names==[*_NAMES,"tessera.nvfp4_expand","tessera.scaled_matmul"])))


@dataclass(frozen=True)
class TracedQuantizedProduct:
    native: NativeQuantizedProductProgram
    argument_names: tuple[str,...]

    @property
    def graph_ir(self):
        return self.native.graph_ir

    def validate(self):
        self.native.validate()
        roles,_=_frame_shape(json.loads(self.native.program_json))
        if (len(self.argument_names)!=len(roles) or len(set(self.argument_names))!=len(roles)
                or any(not isinstance(name,str) or not name.isidentifier() for name in self.argument_names)):
            raise ValueError("native quantized-product frontend argument names differ")
        for name in self.argument_names:
            inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def manifest(self):
        self.validate()
        return {"schema":_EXPANDED_SCHEMA if self.native.expansion is not None else _SCHEMA,"argument_names":list(self.argument_names),"native_program":self.native.to_manifest()}

    def execute(self,args):
        from tessera import runtime as rt
        if rt._rocm_live_arch()!="gfx1201":
            raise RuntimeError("native quantized product requires owning gfx1201")
        self.validate()
        signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in self.argument_names])
        if isinstance(args,dict):
            bound=signature.bind(**args)
        elif isinstance(args,(tuple,list)):
            bound=signature.bind(*args)
        else:
            raise TypeError("native quantized product requires its tensor bindings")
        values=[bound.arguments[name] for name in self.argument_names]
        lib=rt._load_rocm_native_movement_runtime()
        if lib is None:
            raise RuntimeError("native quantized product requires multi-output HIP provider")
        with self.native.prepare(values,runtime_library=lib._name) as owner:
            generation,_=owner.invoke()
            output,=owner.read(generation)
        receipts=tuple({"ok":True,"execution_kind":"native_gpu","abi_id":p.descriptor.abi_id,
                        "image_digest":p.image.image_digest,"descriptor_digest":p.descriptor.descriptor_digest,
                        "native_call_binding":"native_cpp_multi_output"} for p in self.native.native_stages)
        return output,receipts


def _source(module):
    if not requests_quantized_product(module):
        raise ValueError("requires the original typed quantizer/product Graph")
    source=deepcopy(module)
    for key,value in (("tessera.target","rocm_gfx1201"),("tessera.arch","gfx1201")):
        old=source.module_attrs.get(key)
        if old is not None and json.loads(old)!=value:
            raise ValueError("quantized-product source target conflicts")
        source.module_attrs[key]=json.dumps(value)
    old=source.module_attrs.get("tessera.native.quantized_product")
    if old not in (None,"unit"):
        raise ValueError("quantized-product source intent conflicts")
    source.module_attrs["tessera.native.quantized_product"]="unit"
    return source.to_mlir(target="rocm_gfx1201",canonical=True)


def supports_quantized_product(module):
    if not requests_quantized_product(module):
        return False
    tool=find_tessera_opt()
    if tool is None:
        return False
    try:
        _export(tool,_source(module))
    except (RuntimeError,ValueError):
        return False
    return True


def package_traced_quantized_product(module):
    native=package_quantized_product_graph(_source(module))
    program=TracedQuantizedProduct(native,tuple(arg.name for arg in module.functions[0].args))
    program.validate()
    return program


def runtime_artifact(program):
    from tessera.runtime import RuntimeArtifact
    program.validate()
    native=program.native
    return RuntimeArtifact(graph_ir=program.graph_ir,
        schedule_ir="\n".join(native.schedule_irs),
        tile_ir="\n".join(p.tile_ir for p in native.native_stages),
        target_ir="\n".join(p.target_ir for p in native.native_stages),
        metadata={"target":"rocm_gfx1201","compiler_path":"canonical_rocm_quantized_product_program",
                  "execution_kind":"native_gpu","runtime_status":"ready","executable":True,
                  "native_graph_verified":True,"arg_names":list(program.argument_names),"native_program":program.manifest()})


@lru_cache(maxsize=24)
def _resolve(record,graph,names):
    value=json.loads(record)
    if (not isinstance(value,dict) or set(value)!={"schema","argument_names","native_program"}
            or value["schema"] not in (_SCHEMA,_EXPANDED_SCHEMA) or not isinstance(value["argument_names"],list)):
        raise ValueError("native quantized-product frontend manifest differs")
    program=TracedQuantizedProduct(NativeQuantizedProductProgram.from_manifest(value["native_program"]),tuple(value["argument_names"]))
    program.validate()
    if (value["schema"]==_EXPANDED_SCHEMA)!=(program.native.expansion is not None):
        raise ValueError("native quantized-product frontend schema differs")
    if program.graph_ir!=graph or list(program.argument_names)!=json.loads(names):
        raise ValueError("native quantized-product parent Graph or arguments differ")
    return program


def execute_artifact(artifact,args):
    from tessera.runtime import NativeProgramExecution
    if os.getpid()!=_OWNER_PID:
        raise RuntimeError("native quantized-product retention cannot cross fork")
    metadata=artifact.metadata or {}
    if metadata.get("target")!="rocm_gfx1201":
        raise ValueError("native quantized-product parent target differs")
    program=_resolve(json.dumps(metadata.get("native_program"),sort_keys=True,allow_nan=False),
                     artifact.graph_ir,json.dumps(metadata.get("arg_names")))
    native=program.native
    if (artifact.schedule_ir!="\n".join(native.schedule_irs)
            or artifact.tile_ir!="\n".join(p.tile_ir for p in native.native_stages)
            or artifact.target_ir!="\n".join(p.target_ir for p in native.native_stages)):
        raise ValueError("native quantized-product parent stage witnesses differ")
    output,receipts=program.execute(args)
    return NativeProgramExecution(output,receipts)
