"""Checked Graph/Schedule/Tile/Target native MXFP4 storage bridge."""
from dataclasses import dataclass
import hashlib
import json
import re
import numpy as np

from .graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,tensor_ir_type,_infer_result_types
from .native_artifact import BufferBinding,LaunchDescriptor,LaunchGeometry,NativeEntryPoint,NativeImageArtifact,OrderingSemantics,ShapeGuard
from .rocm_native import ROCMNativePackage,_compile_native_tile_ir
from .rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT,checked_storage_inputs
from .rocm_nvfp4_ingest_native import _execute_arrays
from .scheduled_matmul import find_tessera_opt,run_tessera_opt

MXFP4_STORAGE_ABI="tessera.rocm.mxfp4_folded_storage.four_memrefs.static.v1"

def _required_field(pattern: str, source: str) -> str:
    match = re.search(pattern, source)
    if match is None:
        raise ValueError("storage bridge native contract field is missing")
    return match.group(1)

def _contract(tile: str, target: str) -> tuple[tuple[str, ...], int, int, str, str]:
    if tile.count("tile.mxfp4_folded_storage_kernel")!=1 or target.count("tessera_rocm.mxfp4_folded_storage")!=1:
        raise ValueError("storage bridge requires one native Tile/Target operation")
    names=tuple(json.loads(_required_field(r"bindings = (\[[^\]]+\])", tile)))
    n=int(_required_field(r"\bn = (\d+) : i64", target))
    k=int(_required_field(r"\bk = (\d+) : i64", target))
    digest=_required_field(r'schedule_hash = "([0-9a-f]{64})"', target)
    entry=_required_field(r'name = "([^"]+)"', target)
    policy=_required_field(r'storage_contract = "([^"]+)"', target)
    if len(names)!=4 or len(set(names))!=4 or n<=0 or n%16 or k<=0 or k%64 or policy!=MXFP4_STORAGE_CONTRACT:
        raise ValueError("storage bridge shape, bindings or lossless contract changed")
    return names,n,k,digest,entry

def _descriptor(image,names,n,k,digest,entry,graph=None,schedule=None,tile=None,*,
        graph_digest=None,schedule_ir_digest=None,tile_ir_digest=None):
    shapes=((n,k//2),(k//32,n),(n,k//2),(k//32+1,n))
    return LaunchDescriptor(image_digest=image.image_digest,entry_symbol=entry,abi_id=MXFP4_STORAGE_ABI,
        buffers=tuple(BufferBinding(i,name,"input" if i<2 else "output","uint8",2,"row_major",1)
            for i,name in enumerate(names)),
        shape_guards=tuple(ShapeGuard(name,axis,"eq",dim)
            for name,shape in zip(names,shapes) for axis,dim in enumerate(shape)),
        geometry=LaunchGeometry(grid=((n*(k//2)+255)//256,1,1),workgroup=(256,1,1)),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",)),
        provenance={"work_item":"ROCM-NVFP4-INGEST-1","storage_contract":MXFP4_STORAGE_CONTRACT,
            "shape_nk":[n,k],"schedule_digest":digest,"lossy_steps":[],
            "ownership":"private_outputs_distinct_readonly_inputs",
            "route":"GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
            "graph_digest":hashlib.sha256(graph.encode()).hexdigest() if graph is not None else graph_digest,
            "schedule_ir_digest":hashlib.sha256(schedule.encode()).hexdigest() if schedule is not None else schedule_ir_digest,
            "tile_ir_digest":hashlib.sha256(tile.encode()).hexdigest() if tile is not None else tile_ir_digest})

@dataclass(frozen=True)
class MXFP4StoragePackage:
    graph_ir:str
    schedule_ir:str
    native:ROCMNativePackage

    def validate(self):
        image,desc=self.native.image,self.native.descriptor
        desc.validate_image(image)
        if image.target!="rocm_gfx1201" or image.architecture!="gfx1201" or image.binary_format!="hsaco":
            raise ValueError("storage bridge requires exact gfx1201 HSACO")
        if image.target_ir_digest!=hashlib.sha256(self.native.target_ir.encode()).hexdigest():
            raise ValueError("storage bridge Target digest differs")
        info=_contract(self.native.tile_ir,self.native.target_ir)
        if re.findall(r'\bhash = "([0-9a-f]{64})"',self.schedule_ir)!=[info[-2]]:
            raise ValueError("storage bridge Schedule digest differs")
        if desc!=_descriptor(image,*info,self.graph_ir,self.schedule_ir,self.native.tile_ir):
            raise ValueError("storage bridge descriptor differs from native ownership")

def build_mxfp4_storage_graph(n,k):
    if type(n) is not int or type(k) is not int or n<=0 or n%16 or k<=0 or k%64:
        raise ValueError("storage bridge requires positive static N16/K64")
    types=[tensor_ir_type((n,k//2),"uint8"),tensor_ir_type((k//32,n),"uint8")]
    attrs={"storage_contract":MXFP4_STORAGE_CONTRACT}
    results=_infer_result_types("tessera.mxfp4_folded_storage",types,attrs)
    operation=IROp(result="fragment, plane",op_name="tessera.mxfp4_folded_storage",
        operands=["%codes","%exponents"],operand_types=list(map(str,types)),
        result_type="("+", ".join(map(str,results))+")",kwargs=attrs,
        inferred_type=results[0],inferred_types=tuple(results))
    fn=GraphIRFunction("storage",
        args=[IRArg(name,ty) for name,ty in zip(("codes","exponents"),types)],
        result_types=list(results),body=[operation],return_values=["%fragment","%plane"],
        fn_attrs={"tessera.bindings":json.dumps(["codes","exponents","fragment","plane"])})
    return GraphIRModule([fn],module_attrs={"tessera.target":json.dumps("rocm_gfx1201"),
        "tessera.arch":json.dumps("gfx1201")})

def package_mxfp4_storage_graph(graph):
    if not isinstance(graph,str):
        graph=graph.to_mlir(target="rocm_gfx1201",canonical=True)
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError("storage bridge requires matching native compiler")
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    target,backend,binary,compiler,toolchain,libraries,state=_compile_native_tile_ir(
        tile,directive="tessera_rocm.mxfp4_folded_storage",family="quant_fp",architecture="gfx1201")
    info=_contract(tile,target)
    image=NativeImageArtifact(target="rocm_gfx1201",architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",compiler_fingerprint=compiler,
        toolchain_fingerprint=toolchain,target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),
        binary_format="hsaco",payload=binary,
        entry_points=(NativeEntryPoint(info[-1],MXFP4_STORAGE_ABI),),
        device_libraries=libraries,compile_state=state)
    desc=_descriptor(image,*info,graph,schedule,tile)
    package=MXFP4StoragePackage(graph,schedule,ROCMNativePackage(tile,target,backend,image,desc))
    package.validate()
    return package

def execute_mxfp4_storage(package,codes,exponents,*,event_samples=None):
    package.validate()
    _,n,k,_,_=_contract(package.native.tile_ir,package.native.target_ir)
    inputs=checked_storage_inputs(codes,exponents,storage_contract=MXFP4_STORAGE_CONTRACT)
    if inputs[0].shape!=(n,k//2):
        raise ValueError("storage bridge inputs differ from package shape")
    arrays=[np.ascontiguousarray(x) for x in inputs]+[
        np.empty((n,k//2),np.uint8),np.empty((k//32+1,n),np.uint8)]
    return _execute_arrays(package.native.image,package.native.descriptor,arrays,event_samples=event_samples)


def supports_mxfp4_storage(module):
    return (len(module.functions)==1 and len(module.functions[0].body)==1
        and module.functions[0].body[0].op_name=="tessera.mxfp4_folded_storage")

def project_mxfp4_storage_graph(module):
    from copy import deepcopy
    if not supports_mxfp4_storage(module):
        raise ValueError("storage bridge requires one isolated semantic operation")
    result=deepcopy(module)
    for key,value in (("tessera.target","rocm_gfx1201"),("tessera.arch","gfx1201")):
        existing=result.module_attrs.get(key)
        if existing is not None and json.loads(existing)!=value:
            raise ValueError("storage bridge target/architecture differs")
        result.module_attrs[key]=json.dumps(value)
    fn=result.functions[0]
    names=[a.name for a in fn.args]+fn.body[0].result_names
    if len(fn.args)!=2 or len(names)!=4 or len(set(names))!=4 or fn.return_values!=[
            "%"+name for name in fn.body[0].result_names]:
        raise ValueError("storage bridge argument/result ownership differs")
    existing=fn.fn_attrs.get("tessera.bindings")
    if existing is not None and json.loads(existing)!=names:
        raise ValueError("storage bridge Graph binding names differ")
    fn.fn_attrs["tessera.bindings"]=json.dumps(names)
    return result

def validate_storage_runtime_artifact(artifact):
    package=MXFP4StoragePackage(artifact.graph_ir,artifact.schedule_ir,
        ROCMNativePackage(artifact.tile_ir,artifact.target_ir,"",
            artifact.native_image,artifact.launch_descriptor))
    package.validate()

def submit_mxfp4_storage(image,desc,buffers,scalars,stream):
    if scalars or stream is not None:
        raise ValueError("MXFP4 storage bridge uses its static ABI and HIP default-stream completion")
    desc.validate_image(image)
    if (image.target!="rocm_gfx1201" or image.architecture!="gfx1201"
            or image.binary_format!="hsaco" or desc.abi_id!=MXFP4_STORAGE_ABI):
        raise ValueError("MXFP4 storage bridge requires exact gfx1201 native ABI")
    names=tuple(b.name for b in sorted(desc.buffers,key=lambda b:b.ordinal))
    if len(names)!=4 or set(buffers)!=set(names):
        raise ValueError("MXFP4 storage bridge bindings differ")
    n,k=desc.provenance["shape_nk"]
    if type(n) is not int or type(k) is not int or n<=0 or n%16 or k<=0 or k%64:
        raise ValueError("MXFP4 storage bridge shape differs")
    info=desc.provenance
    expected=_descriptor(image,names,n,k,info["schedule_digest"],desc.entry_symbol,
        graph_digest=info["graph_digest"],schedule_ir_digest=info["schedule_ir_digest"],
        tile_ir_digest=info["tile_ir_digest"])
    if desc!=expected:
        raise ValueError("MXFP4 storage descriptor differs from its exact native ABI")
    inputs=checked_storage_inputs(*(buffers[name] for name in names[:2]),
        storage_contract=desc.provenance["storage_contract"])
    if inputs[0].shape!=(n,k//2):
        raise ValueError("MXFP4 storage bridge inputs differ from static shape")
    outputs=[np.asarray(buffers[name]) for name in names[2:]]
    for output,shape in zip(outputs,((n,k//2),(k//32+1,n))):
        if output.dtype!=np.uint8 or output.shape!=shape or not output.flags.c_contiguous or not output.flags.writeable:
            raise ValueError("MXFP4 storage bridge output storage differs")
    for i,output in enumerate(outputs):
        if any(np.shares_memory(output,value) for value in list(inputs)+outputs[:i]):
            raise ValueError("MXFP4 storage bridge outputs must not alias inputs/one another")
    return _execute_arrays(image,desc,[np.ascontiguousarray(x) for x in inputs]+outputs)
