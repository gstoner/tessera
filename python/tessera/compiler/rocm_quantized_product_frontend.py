"""Authored frontend and portable runtime binding for native quantized products."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import inspect
import json
import os
import threading

import numpy as np

from .rocm_block_quant_native import _NAMES
from .rocm_quantized_product_program import NativeQuantizedProductProgram,package_quantized_product_graph,_export,_frame_shape
from .rocm_quantized_mlp_program import NativeQuantizedMLPProgram
from .scheduled_matmul import find_tessera_opt

_SCHEMA="tessera.rocm.quantized_product_frontend.v1"
_EXPANDED_SCHEMA="tessera.rocm.quantized_product_frontend.v2"
_ACTIVATED_SCHEMA="tessera.rocm.quantized_product_frontend.v3"
_MLP_SCHEMA="tessera.rocm.quantized_product_frontend.v4"
_BOUNDED_MLP_SCHEMA="tessera.rocm.quantized_product_frontend.v5"
_K_BOUNDED_MLP_SCHEMA="tessera.rocm.quantized_product_frontend.v6"
_N_BOUNDED_MLP_SCHEMA="tessera.rocm.quantized_product_frontend.v7"
_H_BOUNDED_MLP_SCHEMA="tessera.rocm.quantized_product_frontend.v8"
_FRONTEND_SCHEMAS=(_SCHEMA,_EXPANDED_SCHEMA,_ACTIVATED_SCHEMA,_MLP_SCHEMA,_BOUNDED_MLP_SCHEMA,
                   _K_BOUNDED_MLP_SCHEMA,_N_BOUNDED_MLP_SCHEMA,_H_BOUNDED_MLP_SCHEMA)
_OWNER_PID=os.getpid()


def requests_quantized_product(module):
    if module is None or len(module.functions)!=1:
        return False
    fn=module.functions[0]
    names=[op.op_name for op in fn.body if op.op_name!="arith.constant"]
    return (len(fn.result_types)==1 and (
        (len(fn.args)==3 and names==[*_NAMES,"tessera.scaled_matmul"])
        or (len(fn.args)==4 and names==[*_NAMES,"tessera.nvfp4_expand","tessera.scaled_matmul"])
        or (len(fn.args)==4 and names==[*_NAMES,"tessera.nvfp4_expand","tessera.scaled_matmul",
            "tessera.cast","tessera.slice","tessera.slice","tessera.cast","tessera.cast","tessera.silu_mul","tessera.cast"])
        or (len(fn.args)==7 and names==[*_NAMES,"tessera.nvfp4_expand","tessera.scaled_matmul",
            "tessera.cast","tessera.slice","tessera.slice","tessera.cast","tessera.cast","tessera.silu_mul","tessera.cast",
            *_NAMES,"tessera.nvfp4_expand","tessera.scaled_matmul","tessera.cast"])))


@dataclass(frozen=True)
class TracedQuantizedProduct:
    native: NativeQuantizedProductProgram | NativeQuantizedMLPProgram
    argument_names: tuple[str,...]

    @property
    def graph_ir(self):
        return self.native.graph_ir

    @property
    def bounded(self):
        return bool(getattr(self.native,"bounded",False))

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
        return {"schema":_frontend_schema(self.native),"argument_names":list(self.argument_names),"native_program":self.native.to_manifest()}

    def _bind(self,args):
        from tessera import runtime as rt
        if rt._rocm_live_arch()!=self.native.arch:
            raise RuntimeError(f"native quantized product requires owning {self.native.arch}")
        signature=inspect.Signature([inspect.Parameter(name,inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in self.argument_names])
        if isinstance(args,dict):
            bound=signature.bind(**args)
        elif isinstance(args,(tuple,list)):
            bound=signature.bind(*args)
        else:
            raise TypeError("native quantized product requires its tensor bindings")
        lib=rt._load_rocm_native_movement_runtime()
        if lib is None:
            raise RuntimeError("native quantized product requires multi-output HIP provider")
        return [bound.arguments[name] for name in self.argument_names],lib

    def receipts(self):
        return tuple({"ok":True,"execution_kind":"native_gpu","abi_id":p.descriptor.abi_id,
                      "image_digest":p.image.image_digest,"descriptor_digest":p.descriptor.descriptor_digest,
                      "native_call_binding":"native_cpp_multi_output"} for p in self.native.native_stages)

    def execute(self,args):
        self.validate()
        values,lib=self._bind(args)
        with self.native.prepare(values,runtime_library=lib._name) as owner:
            generation,_=owner.invoke()
            output,=owner.read(generation)
        return output,self.receipts()


class RetainedBoundedOwner:
    """One prepared bounded-MLP owner reused across public calls.

    A call whose six weight roots are byte-identical to the ones the owner last
    admitted replaces only the activation rows; the expanded weights stay
    resident. Any other weight set takes the owner's full validated update.
    An admission refusal (``ValueError``, raised before any native write) keeps
    the owner; a native failure closes it, so the next call prepares a fresh
    one. The owner never crosses fork.
    """

    def __init__(self,program):
        if not program.bounded:
            raise ValueError("only a bounded native quantized MLP retains its owner")
        program.validate()
        self.program=program
        self._receipts=program.receipts()
        self._owner=None
        self._weights=None
        self._pid=os.getpid()
        self._lock=threading.Lock()

    @staticmethod
    def _checked(value):
        """The caller's array, its backing span proven before any byte is read.

        A held snapshot certifies only the allocation it was taken from; a
        replacement view (``as_strided``) must pass the same span admission the
        native upload applies, or reading it could leave its allocation.
        """
        from .paged_host_span import checked_host_span
        array=np.asarray(value)
        checked_host_span(array,label="quantized weight",min_rank=1)
        return array

    @classmethod
    def _snapshot(cls,values):
        snapshot=[]
        for value in values:
            array=cls._checked(value)
            snapshot.append((array.dtype,array.shape,np.ascontiguousarray(array).view(np.uint8).copy()))
        return snapshot

    def _same_weights(self,values):
        if self._weights is None:
            return False
        arrays=[self._checked(value) for value in values]
        for array,(dtype,shape,held) in zip(arrays,self._weights,strict=True):
            if array.dtype!=dtype or array.shape!=shape:
                return False
            if not np.array_equal(np.ascontiguousarray(array).view(np.uint8),held):
                return False
        return True

    def execute(self,args):
        if os.getpid()!=self._pid:
            raise RuntimeError("native quantized-product retention cannot cross fork")
        values,lib=self.program._bind(args)
        with self._lock:
            try:
                if self._owner is None:
                    self._owner=self.program.native.prepare(values,runtime_library=lib._name)
                    self._weights=self._snapshot(values[1:])
                elif self._same_weights(values[1:]):
                    # Validates the activation before any native state changes.
                    self._owner.update_activations(values[0])
                else:
                    self._owner.update(values)
                    self._weights=self._snapshot(values[1:])
                generation,_=self._owner.invoke()
                output,=self._owner.read(generation)
            except ValueError:
                # Admission refusals are raised before any native write, so the
                # resident rows and weights still match the held snapshot.
                raise
            except Exception:
                # A native failure may have written part of the owner.
                self._close_locked()
                raise
        return output,self._receipts

    def _close_locked(self):
        owner,self._owner,self._weights=self._owner,None,None
        if owner is not None:
            owner.close()

    def close(self):
        if os.getpid()!=self._pid:
            return
        with self._lock:
            self._close_locked()


#: The exact targets that own a native quantized program: gfx1201, and gfx1151's
#: compile-only decoded-F16 program (GFX1151-QUANTIZED-MLP-20261030), whose
#: executable images still wait on exact-device family promotion.
QUANTIZED_PRODUCT_TARGETS=("rocm_gfx1201","rocm_gfx1151")


def _source(module,target="rocm_gfx1201"):
    if target not in QUANTIZED_PRODUCT_TARGETS:
        raise ValueError(f"native quantized product requires one of {QUANTIZED_PRODUCT_TARGETS}")
    if not requests_quantized_product(module):
        raise ValueError("requires the original typed quantizer/product Graph")
    source=deepcopy(module)
    for key,value in (("tessera.target",target),("tessera.arch",target.removeprefix("rocm_"))):
        old=source.module_attrs.get(key)
        if old is not None and json.loads(old)!=value:
            raise ValueError("quantized-product source target conflicts")
        source.module_attrs[key]=json.dumps(value)
    old=source.module_attrs.get("tessera.native.quantized_product")
    if old not in (None,"unit"):
        raise ValueError("quantized-product source intent conflicts")
    # tessera.native.quantized_{m,k,n}_bound requests pass through unchanged: the
    # native export owns their admission (seven-member MLP, checked capacity).
    source.module_attrs["tessera.native.quantized_product"]="unit"
    return source.to_mlir(target=target,canonical=True)


def supports_quantized_product(module,target="rocm_gfx1201"):
    if not requests_quantized_product(module):
        return False
    tool=find_tessera_opt()
    if tool is None:
        return False
    try:
        _export(tool,_source(module,target))
    except (RuntimeError,ValueError):
        return False
    return True


def package_traced_quantized_product(module,target="rocm_gfx1201"):
    native=package_quantized_product_graph(_source(module,target))
    program=TracedQuantizedProduct(native,tuple(arg.name for arg in module.functions[0].args))
    program.validate()
    return program


def _frontend_schema(native):
    schema=json.loads(native.program_json)["schema"]
    return dict(zip((f"tessera.native.quantized_product.v{i}" for i in range(1,9)),
        _FRONTEND_SCHEMAS,strict=True))[schema]


def _compiler_path(native):
    return {_SCHEMA:"canonical_rocm_quantized_product_program",
            _EXPANDED_SCHEMA:"canonical_rocm_quantized_nvfp4_product_program",
            _ACTIVATED_SCHEMA:"canonical_rocm_quantized_gate_up_program",
            _MLP_SCHEMA:"canonical_rocm_quantized_mlp_program",
            _BOUNDED_MLP_SCHEMA:"canonical_rocm_quantized_bounded_mlp_program",
            _K_BOUNDED_MLP_SCHEMA:"canonical_rocm_quantized_k_bounded_mlp_program",
            _N_BOUNDED_MLP_SCHEMA:"canonical_rocm_quantized_n_bounded_mlp_program",
            _H_BOUNDED_MLP_SCHEMA:"canonical_rocm_quantized_h_bounded_mlp_program"}[_frontend_schema(native)]


def runtime_artifact(program):
    from tessera.runtime import RuntimeArtifact
    program.validate()
    native=program.native
    return RuntimeArtifact(graph_ir=program.graph_ir,
        schedule_ir="\n".join(native.schedule_irs),
        tile_ir="\n".join(p.tile_ir for p in native.native_stages),
        target_ir="\n".join(p.target_ir for p in native.native_stages),
        metadata={"target":"rocm_"+native.arch,"compiler_path":_compiler_path(native),
                  "execution_kind":"native_gpu","runtime_status":"ready","executable":True,
                  "native_graph_verified":True,"arg_names":list(program.argument_names),"native_program":program.manifest()})


@lru_cache(maxsize=24)
def _resolve(record,graph,names):
    value=json.loads(record)
    if (not isinstance(value,dict) or set(value)!={"schema","argument_names","native_program"}
            or value["schema"] not in _FRONTEND_SCHEMAS
            or not isinstance(value["argument_names"],list)):
        raise ValueError("native quantized-product frontend manifest differs")
    program=TracedQuantizedProduct(NativeQuantizedProductProgram.from_manifest(value["native_program"]),tuple(value["argument_names"]))
    program.validate()
    expected=_frontend_schema(program.native)
    if value["schema"]!=expected:
        raise ValueError("native quantized-product frontend schema differs")
    if program.graph_ir!=graph or list(program.argument_names)!=json.loads(names):
        raise ValueError("native quantized-product parent Graph or arguments differ")
    return program


def execute_artifact(artifact,args):
    from tessera.runtime import NativeProgramExecution
    if os.getpid()!=_OWNER_PID:
        raise RuntimeError("native quantized-product retention cannot cross fork")
    metadata=artifact.metadata or {}
    if metadata.get("target") not in QUANTIZED_PRODUCT_TARGETS:
        raise ValueError("native quantized-product parent target differs")
    program=_resolve(json.dumps(metadata.get("native_program"),sort_keys=True,allow_nan=False),
                     artifact.graph_ir,json.dumps(metadata.get("arg_names")))
    if metadata["target"]!="rocm_"+program.native.arch:
        raise ValueError("native quantized-product parent target differs from its source Graph")
    native=program.native
    if metadata.get("compiler_path")!=_compiler_path(native):
        raise ValueError("native quantized-product parent compiler path differs")
    if (artifact.schedule_ir!="\n".join(native.schedule_irs)
            or artifact.tile_ir!="\n".join(p.tile_ir for p in native.native_stages)
            or artifact.target_ir!="\n".join(p.target_ir for p in native.native_stages)):
        raise ValueError("native quantized-product parent stage witnesses differ")
    output,receipts=program.execute(args)
    return NativeProgramExecution(output,receipts)
