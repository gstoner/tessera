"""Checked gfx1201 checkpoint conversion packages; MLIR owns conversion semantics."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferArgument, BufferBinding, LaunchDescriptor, LaunchGeometry,
    NativeEntryPoint, NativeImageArtifact, OrderingSemantics, ShapeGuard,
)
from .rocm_native import ROCMNativePackage, _compile_native_tile_ir
from .rocm_nvfp4_ingest import nvfp4_requantization_policy
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

NVFP4_INGEST_ABI = "tessera.rocm.nvfp4_requantize.six_memrefs.static.v1"


@dataclass(frozen=True)
class NVFP4IngestPackage:
    graph_ir: str
    schedule_ir: str
    native: ROCMNativePackage

    def validate(self):
        image, desc = self.native.image, self.native.descriptor
        desc.validate_image(image)
        if (image.target != "rocm_gfx1201" or image.architecture != "gfx1201"
                or image.binary_format != "hsaco" or desc.abi_id != NVFP4_INGEST_ABI):
            raise ValueError("NVFP4 ingest needs its exact gfx1201 native package")
        if image.target_ir_digest != hashlib.sha256(self.native.target_ir.encode()).hexdigest():
            raise ValueError("NVFP4 ingest Target digest differs")
        info = _contract(self.native.tile_ir, self.native.target_ir)
        if re.findall(r'\bhash = "([0-9a-f]{64})"', self.schedule_ir) != [info[-2]]:
            raise ValueError("NVFP4 ingest Schedule record digest differs")
        if desc != _descriptor(image, *info, graph_digest=hashlib.sha256(self.graph_ir.encode()).hexdigest(),
                schedule_ir_digest=hashlib.sha256(self.schedule_ir.encode()).hexdigest(),
                tile_ir_digest=hashlib.sha256(self.native.tile_ir.encode()).hexdigest()):
            raise ValueError("NVFP4 ingest descriptor differs from native ownership contract")
        if desc.provenance["graph_digest"] != hashlib.sha256(self.graph_ir.encode()).hexdigest():
            raise ValueError("NVFP4 ingest Graph digest differs")


def _required_field(pattern: str, source: str) -> str:
    match = re.search(pattern, source)
    if match is None:
        raise ValueError("NVFP4 native contract field is missing")
    return match.group(1)

def _contract(tile: str, target: str) -> tuple[tuple[str, ...], int, int, tuple[int, ...], str, str]:
    if tile.count("tile.nvfp4_requantize_kernel") != 1:
        raise ValueError("NVFP4 package requires one isolated typed Tile converter")
    if target.count("tessera_rocm.nvfp4_requantize") != 1:
        raise ValueError("NVFP4 package requires one native Target converter")
    names = json.loads(_required_field(r"bindings = (\[[^\]]+\])", tile))
    n = int(_required_field(r"\bn = (\d+) : i64", target))
    k = int(_required_field(r"\bk = (\d+) : i64", target))
    offsets = tuple(json.loads(_required_field(r"row_offsets = (\[[^\]]+\])", target)))
    digest = _required_field(r'schedule_hash = "([0-9a-f]{64})"', target)
    entry = _required_field(r'name = "([^"]+)"', target)
    if (len(names) != 6 or len(set(names)) != 6 or n <= 0 or k <= 0 or k % 32
            or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n
            or any(a >= b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("NVFP4 package lost its shape/binding/projection contract")
    return tuple(names), n, k, offsets, digest, entry


def _descriptor(image, names, n, k, offsets, digest, entry, *, graph_digest, schedule_ir_digest, tile_ir_digest):
    shapes = ((n,k//2),(n,k//16),(len(offsets)-1,),
              (n,k//2),(k//32,n),(n,k//32,2))
    dtypes = ("uint8","fp8_e4m3","fp64","uint8","uint8","fp64")
    return LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=NVFP4_INGEST_ABI,
        buffers=tuple(BufferBinding(i,name,"input" if i < 3 else "output",
            dtypes[i],len(shapes[i]),"row_major",8 if i in (2,5) else 1)
            for i,name in enumerate(names)),
        shape_guards=tuple(ShapeGuard(name,axis,"eq",dim)
            for name,shape in zip(names,shapes) for axis,dim in enumerate(shape)),
        geometry=LaunchGeometry(grid=((n*(k//32)+255)//256,1,1),workgroup=(256,1,1)),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",
                                  synchronization=("completion",)),
        provenance={"work_item":"ROCM-NVFP4-INGEST-1",
            "route":"GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
            "numeric_policy":nvfp4_requantization_policy(),"row_offsets":list(offsets),
            "schedule_digest":digest,"shape_nk":[n,k],
            "ownership":"private_outputs_distinct_readonly_inputs",
            "graph_digest":graph_digest,"schedule_ir_digest":schedule_ir_digest,
            "tile_ir_digest":tile_ir_digest})


def package_nvfp4_ingest_graph(graph_ir: str | GraphIRModule) -> NVFP4IngestPackage:
    """Package a caller-authored semantic Graph without policy normalization."""
    if not isinstance(graph_ir,str):
        graph_ir=graph_ir.to_mlir(target="rocm_gfx1201",canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("NVFP4 ingest requires production tessera-opt")
    schedule = run_tessera_opt(tool,graph_ir,"--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    target,backend,binary,compiler,toolchain,libraries,state = _compile_native_tile_ir(
        tile,directive="tessera_rocm.nvfp4_requantize",family="quant_fp",architecture="gfx1201")
    info = _contract(tile,target)
    graph_digest = hashlib.sha256(graph_ir.encode()).hexdigest()
    image = NativeImageArtifact(target="rocm_gfx1201",architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=compiler,toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),binary_format="hsaco",
        payload=binary,entry_points=(NativeEntryPoint(info[-1],NVFP4_INGEST_ABI),),
        device_libraries=libraries,compile_state=state)
    descriptor = _descriptor(image,*info,graph_digest=graph_digest,
        schedule_ir_digest=hashlib.sha256(schedule.encode()).hexdigest(),
        tile_ir_digest=hashlib.sha256(tile.encode()).hexdigest())
    package = NVFP4IngestPackage(graph_ir,schedule,
        ROCMNativePackage(tile,target,backend,image,descriptor))
    package.validate()
    return package


def execute_nvfp4_ingest(package: NVFP4IngestPackage, codes, scales, globals_, *, event_samples=None):
    """Synchronous checked host-buffer boundary, with private device outputs."""
    import numpy as np

    package.validate()
    names,n,k,offsets,_,_ = _contract(package.native.tile_ir,package.native.target_ir)
    inputs = _checked_inputs(n,k,offsets,codes,scales,globals_)
    arrays = [np.ascontiguousarray(x) for x in inputs] + [
        np.empty((n,k//2),np.uint8),np.empty((k//32,n),np.uint8),
        np.empty((n,k//32,2),np.float64)]
    return _execute_arrays(package.native.image,package.native.descriptor,arrays,
        event_samples=event_samples)


def _execute_arrays(image,desc,arrays,*,event_samples=None):
    from tessera import runtime as rt
    bindings=sorted(desc.buffers,key=lambda b:b.ordinal)
    if len(bindings)!=len(arrays) or any(b.direction not in {"input","output"} for b in bindings):
        raise ValueError("flat native storage ABI requires complete input/output arrays")
    dtype_names=tuple(b.dtype for b in bindings)
    input_ordinals={i for i,b in enumerate(bindings) if b.direction=="input"}
    output_ordinals=[i for i,b in enumerate(bindings) if b.direction=="output"]
    desc.validate_invocation(image,{
        name:BufferArgument(dtype,array.shape,"row_major",
            int(array.ctypes.data & -array.ctypes.data))
        for name,dtype,array in zip(
            (b.name for b in sorted(desc.buffers,key=lambda b:b.ordinal)),dtype_names,arrays)}, {})
    if rt._rocm_live_arch() != "gfx1201":
        raise ValueError("NVFP4 ingest execution requires live gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("HIP runtime unavailable")
    def check(rc):
        if rc:
            raise RuntimeError(f"NVFP4 ingest HIP status {rc}")
    check(hip.hipInit(0))
    blob = C.create_string_buffer(image.payload)
    module,function = C.c_void_p(),C.c_void_p()
    pointers: list[C.c_void_p] = []
    values: list[C.c_void_p | C.c_int64] = []
    check(hip.hipModuleLoadData(C.byref(module),blob))
    try:
        check(hip.hipModuleGetFunction(C.byref(function),module,desc.entry_symbol.encode()))
        for i,array in enumerate(arrays):
            ptr = C.c_void_p()
            check(hip.hipMalloc(C.byref(ptr),array.nbytes))
            pointers.append(ptr)
            if i in input_ordinals:
                check(hip.hipMemcpy(ptr,C.c_void_p(array.ctypes.data),array.nbytes,1))
            values.extend((C.c_void_p(ptr.value),C.c_void_p(ptr.value),
                           C.c_int64(0),C.c_int64(array.size),C.c_int64(1)))
        params = (C.c_void_p*len(values))(
            *[C.cast(C.byref(value),C.c_void_p) for value in values])
        check(hip.hipModuleLaunchKernel(function,*desc.geometry.grid,
            *desc.geometry.workgroup,0,None,params,None))
        check(hip.hipDeviceSynchronize())
        if event_samples is not None:
            signatures={
                "hipEventCreate":[C.POINTER(C.c_void_p)],
                "hipEventRecord":[C.c_void_p,C.c_void_p],
                "hipEventSynchronize":[C.c_void_p],
                "hipEventElapsedTime":[C.POINTER(C.c_float),C.c_void_p,C.c_void_p],
                "hipEventDestroy":[C.c_void_p]}
            for symbol,types in signatures.items():
                getattr(hip,symbol).argtypes=types
                getattr(hip,symbol).restype=C.c_int
            events=[C.c_void_p(),C.c_void_p()]
            try:
                for event in events:
                    check(hip.hipEventCreate(C.byref(event)))
                for _ in range(3):
                    check(hip.hipEventRecord(events[0],None))
                    for _ in range(10):
                        check(hip.hipModuleLaunchKernel(function,*desc.geometry.grid,
                            *desc.geometry.workgroup,0,None,params,None))
                    check(hip.hipEventRecord(events[1],None))
                    check(hip.hipEventSynchronize(events[1]))
                    elapsed=C.c_float()
                    check(hip.hipEventElapsedTime(C.byref(elapsed),*events))
                    event_samples.append(elapsed.value/10)
            finally:
                hip.hipDeviceSynchronize()
                for event in events:
                    if event.value:
                        hip.hipEventDestroy(event)
        for i in output_ordinals:
            check(hip.hipMemcpy(C.c_void_p(arrays[i].ctypes.data),
                pointers[i],arrays[i].nbytes,2))
    finally:
        hip.hipDeviceSynchronize()
        for ptr in pointers:
            hip.hipFree(ptr)
        hip.hipModuleUnload(module)
    return tuple(arrays[i] for i in output_ordinals)


def build_nvfp4_ingest_graph(n: int, k: int, row_offsets, *, numeric_policy):
    """Build typed semantic Graph IR using the catalog's multi-result inference."""
    from .graph_ir import (
        GraphIRFunction,IRArg,IROp,tensor_ir_type,_infer_result_types)
    offsets=tuple(row_offsets)
    if (type(n) is not int or type(k) is not int or n <= 0 or k <= 0 or k % 32
            or len(offsets)<2 or any(type(x) is not int for x in offsets)
            or offsets[0]!=0 or offsets[-1]!=n
            or any(a>=b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("NVFP4 Graph requires positive static N/K32 and ordered boundaries")
    if numeric_policy != nvfp4_requantization_policy():
        raise ValueError("NVFP4 Graph requires the explicit requantization policy")
    types=[tensor_ir_type((n,k//2),"uint8"),
           tensor_ir_type((n,k//16),"fp8_e4m3"),
           tensor_ir_type((len(offsets)-1,),"fp64")]
    attrs={"row_offsets":list(offsets),"numeric_policy":dict(numeric_policy)}
    results=_infer_result_types("tessera.nvfp4_requantize",types,attrs)
    operation=IROp(result="packed, exponents, stats",
        op_name="tessera.nvfp4_requantize",
        operands=["%codes","%scales","%globals"],
        operand_types=[str(t) for t in types],
        result_type="("+", ".join(str(t) for t in results)+")",
        kwargs=attrs,inferred_type=results[0],inferred_types=tuple(results))
    fn=GraphIRFunction("ingest",
        args=[IRArg(name,t) for name,t in zip(("codes","scales","globals"),types)],
        result_types=list(results),body=[operation],
        return_values=["%packed","%exponents","%stats"],
        fn_attrs={"tessera.bindings":json.dumps(
            ["codes","scales","globals","packed","exponents","stats"])})
    return GraphIRModule([fn],module_attrs={
        "tessera.target":json.dumps("rocm_gfx1201"),"tessera.arch":json.dumps("gfx1201")})

def _checked_inputs(n,k,offsets,codes,scales,globals_):
    import ml_dtypes
    import numpy as np
    inputs = [np.asarray(codes),np.asarray(scales),np.asarray(globals_)]
    if (inputs[0].dtype != np.uint8 or inputs[0].shape != (n,k//2)
            or inputs[1].dtype != np.dtype(ml_dtypes.float8_e4m3fn)
            or inputs[1].shape != (n,k//16)
            or inputs[2].dtype != np.float64 or inputs[2].shape != (len(offsets)-1,)):
        raise ValueError("NVFP4 ingest input dtype/shape differs from checked ABI")
    if not np.isfinite(inputs[2]).all() or np.any(inputs[2] <= 0):
        raise ValueError("NVFP4 projection globals must be finite and positive")
    for i,(a,b) in enumerate(zip(offsets,offsets[1:])):
        effective = inputs[1][a:b].astype(np.float64)*inputs[2][i]
        if not np.isfinite(effective).all() or np.any(effective < 0):
            raise ValueError("NVFP4 effective scales must be finite and non-negative")
        maximum = effective.reshape(b-a,k//32,2).max(axis=-1)
        if np.any((maximum > 0) & (maximum < np.ldexp(1.,-126))):
            raise ValueError("NVFP4 scale is outside the representable E8M0 range")
        # Squared-error statistics must remain representable in the checked f64 ABI.
        if np.any(maximum > np.sqrt(np.finfo(np.float64).max/1152.)):
            raise ValueError("NVFP4 scale would overflow f64 block loss statistics")
    return inputs


def ingest_runtime_artifact(package: NVFP4IngestPackage):
    from tessera.runtime import RuntimeArtifact
    package.validate()
    return RuntimeArtifact(
        graph_ir=package.graph_ir,schedule_ir=package.schedule_ir,
        tile_ir=package.native.tile_ir,target_ir=package.native.target_ir,
        native_image=package.native.image,launch_descriptor=package.native.descriptor,
        metadata={"target":"rocm_gfx1201","compiler_path":"canonical_rocm_nvfp4_ingest",
            "executable":True,"execution_kind":"native_gpu"})


def validate_ingest_runtime_artifact(artifact):
    package=NVFP4IngestPackage(artifact.graph_ir,artifact.schedule_ir,
        ROCMNativePackage(artifact.tile_ir,artifact.target_ir,"",
            artifact.native_image,artifact.launch_descriptor))
    package.validate()


def submit_nvfp4_ingest(image,desc,buffers,scalars,stream):
    """Submit the checked six-buffer ABI through the common native runtime."""
    import numpy as np
    if scalars or stream is not None:
        raise ValueError("NVFP4 ingest has no scalar ABI and uses completion on the HIP default stream")
    desc.validate_image(image)
    if (image.target!="rocm_gfx1201" or image.architecture!="gfx1201"
            or image.binary_format!="hsaco" or desc.abi_id!=NVFP4_INGEST_ABI):
        raise ValueError("NVFP4 ingest requires its gfx1201 HSACO ABI")
    ordered=sorted(desc.buffers,key=lambda b:b.ordinal)
    names=tuple(b.name for b in ordered)
    info=desc.provenance
    n,k=info["shape_nk"]
    offsets=tuple(info["row_offsets"])
    if (type(n) is not int or type(k) is not int or n<=0 or k<=0 or k%32
            or len(offsets)<2 or any(type(x) is not int for x in offsets)
            or offsets[0]!=0 or offsets[-1]!=n
            or any(a>=b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("NVFP4 descriptor projection geometry changed")
    expected=_descriptor(image,names,n,k,offsets,info["schedule_digest"],desc.entry_symbol,
        graph_digest=info["graph_digest"],schedule_ir_digest=info["schedule_ir_digest"],
        tile_ir_digest=info["tile_ir_digest"])
    if desc!=expected or set(buffers)!=set(names):
        raise ValueError("NVFP4 ingest descriptor/bindings differ from native ABI")
    inputs=_checked_inputs(n,k,offsets,*(buffers[name] for name in names[:3]))
    outputs=[np.asarray(buffers[name]) for name in names[3:]]
    for output,shape,dtype in zip(outputs,((n,k//2),(k//32,n),(n,k//32,2)),
            (np.uint8,np.uint8,np.float64)):
        if (output.shape!=shape or output.dtype!=np.dtype(dtype)
                or not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("NVFP4 output shape/storage/writeability differs from native ABI")
    arrays=[np.ascontiguousarray(x) for x in inputs]+outputs
    for i,output in enumerate(outputs):
        if any(np.shares_memory(output,other) for other in inputs+outputs[:i]):
            raise ValueError("NVFP4 outputs must not alias inputs or one another")
    return _execute_arrays(image,desc,arrays)

def supports_nvfp4_ingest(module):
    """Select only the semantic operation; native verification owns admission."""
    return (len(module.functions)==1 and len(module.functions[0].body)==1
        and module.functions[0].body[0].op_name=="tessera.nvfp4_requantize")


def project_nvfp4_ingest_graph(module):
    from copy import deepcopy
    if not supports_nvfp4_ingest(module):
        raise ValueError("NVFP4 package requires one isolated semantic Graph operation")
    source=deepcopy(module)
    for key,value in (("tessera.target","rocm_gfx1201"),("tessera.arch","gfx1201")):
        existing=source.module_attrs.get(key)
        if existing is not None and json.loads(existing)!=value:
            raise ValueError("NVFP4 ingest Graph target/architecture conflicts with package")
        source.module_attrs[key]=json.dumps(value)
    fn=source.functions[0]
    op=fn.body[0]
    names=[arg.name for arg in fn.args]+op.result_names
    if (len(fn.args)!=3 or len(op.result_names)!=3 or len(set(names))!=6
            or fn.return_values!=["%"+name for name in op.result_names]):
        raise ValueError("NVFP4 ingest argument/result ownership differs")
    existing=fn.fn_attrs.get("tessera.bindings")
    if existing is not None and json.loads(existing)!=names:
        raise ValueError("NVFP4 Graph binding names conflict with operand/result lineage")
    fn.fn_attrs["tessera.bindings"]=json.dumps(names)
    return source
