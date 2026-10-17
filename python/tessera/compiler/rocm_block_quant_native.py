"""Source-derived K16 quantizer packaging; native passes own every arithmetic op."""
from __future__ import annotations

import base64
from copy import deepcopy
import ctypes as C
from dataclasses import dataclass
import hashlib
import json

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)
from .native_scaled_program import (
    PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec,
    _manifest, _storage_numpy_dtype,
)
from .rocm_nvfp4_expansion_program import _MultiOutputStep
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

BLOCK_QUANT_ABI = "tessera.rocm.block_quant.x_q_s_groups.bf16_e4m3_f32.v1"
_TARGET = "rocm_gfx1201"
_NAMES = ["tessera.reshape", "tessera.cast", "tessera.absolute", "tessera.reduce",
          "tessera.div", "tessera.maximum", "tessera.expand", "tessera.div",
          "tessera.cast", "tessera.reshape", "tessera.reshape"]


def requests_block_quant(module: GraphIRModule) -> bool:
    if len(module.functions) != 1:
        return False
    fn = module.functions[0]
    names = [op.op_name for op in fn.body if op.op_name != "arith.constant"]
    return len(fn.args) == 1 and len(fn.result_types) == 2 and names == _NAMES


def supports_block_quant(module: GraphIRModule) -> bool:
    if not requests_block_quant(module):
        return False
    ty = module.functions[0].args[0].ir_type
    return ty.rank == 2 and ty.dtype == "bf16"


@dataclass(frozen=True)
class BlockQuantRecipe:
    graph_ir: str
    schedule_ir: str
    tile_ir: str

    def validate(self):
        from .rocm_pass_cache import run
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("native block quantization requires matching tessera-opt")
        schedule = run(tool,self.graph_ir,"--tessera-graph-to-schedule",execute=run_tessera_opt)
        tile = run(tool,schedule,"--tessera-schedule-to-tile",execute=run_tessera_opt)
        if schedule != self.schedule_ir or tile != self.tile_ir:
            raise ValueError("block quantization differs from native Graph/Schedule replay")


def lower_block_quant_text(graph: str) -> BlockQuantRecipe:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native block quantization requires matching tessera-opt")
    schedule = run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    recipe = BlockQuantRecipe(graph,schedule,tile)
    recipe.validate()
    return recipe


def lower_block_quant_graph(module: GraphIRModule,target: str) -> BlockQuantRecipe:
    if target != _TARGET or not supports_block_quant(module):
        raise ValueError("block quantization requires its original compact BF16 gfx1201 source")
    source = deepcopy(module)
    for key,value in (("tessera.target",_TARGET),("tessera.arch","gfx1201")):
        old = source.module_attrs.get(key)
        if old is not None and json.loads(old) != value:
            raise ValueError("block quantization source target conflicts")
        source.module_attrs[key] = json.dumps(value)
    marker = source.module_attrs.get("tessera.native.block_quant")
    if marker not in (None,"unit"):
        raise ValueError("block quantization source intent conflicts")
    source.module_attrs["tessera.native.block_quant"] = "unit"
    fn = source.functions[0]
    if len(fn.return_values) != 2:
        raise ValueError("block quantization must return its two original results")
    names = [fn.args[0].name,*[v.removeprefix("%") for v in fn.return_values]]
    if len(set(names)) != 3:
        raise ValueError("block quantization bindings alias")
    old = source.module_attrs.get("tessera.launch_bindings")
    if old is not None and json.loads(old) != names:
        raise ValueError("block quantization source bindings conflict")
    source.module_attrs["tessera.launch_bindings"] = json.dumps(names)
    return lower_block_quant_text(source.to_mlir(target=target,canonical=True))


def _validate_frame(program,member):
    if (program.get("schema") != 1 or program.get("kind") != "block_quant"
            or program.get("argument_count") != 1 or program.get("outputs") != [1,2]
            or member.get("schema") != 1 or member.get("step") != 0
            or member.get("architecture") != "gfx1201"
            or member.get("profile") != "gfx1201_block_quant_bf16_e4m3_f32_k16_v1"
            or member.get("abi") != "flattened_rank1_memref_then_i64"
            or member.get("inputs") != [0] or member.get("outputs") != [1,2]):
        raise ValueError("block quantization native profile differs")
    buffers = program.get("buffers")
    if not isinstance(buffers,list) or len(buffers) != 3:
        raise ValueError("block quantization native buffer count differs")
    shape = buffers[0].get("shape")
    names = member.get("bindings")
    if (not isinstance(shape,list) or len(shape) != 2
            or any(type(v) is not int or not 0 < v <= 2**31-1 for v in shape)
            or shape[1] % 16 or shape[0] * (shape[1]//16) > 2**31-1
            or not isinstance(names,list) or len(names) != 3
            or any(not isinstance(v,str) or not v for v in names) or len(set(names)) != 3):
        raise ValueError("block quantization shape or bindings differ")
    m,k = shape
    elements,groups = m*k,m*k//16
    expected = [
        {"shape":shape,"storage":"bf16","elements":elements,"bytes":2*elements,"first_write":-1,"last_read":0,"ownership":0},
        {"shape":shape,"storage":"f8E4M3FN","elements":elements,"bytes":elements,"first_write":0,"last_read":1,"ownership":2},
        {"shape":[m,k//16],"storage":"f32","elements":groups,"bytes":4*groups,"first_write":0,"last_read":1,"ownership":2},
    ]
    scalars,geometry = member.get("scalars"),member.get("geometry")
    if (not isinstance(scalars,list) or len(scalars)!=1 or type(scalars[0]) is not int
            or not isinstance(geometry,list) or len(geometry)!=6 or any(type(v) is not int for v in geometry)):
        raise ValueError("block quantization scalar/geometry fields require integers")
    if (buffers != expected or member.get("scalars") != [groups]
            or member.get("geometry") != [(groups+255)//256,1,1,256,1,1]
            or program.get("steps") != [{"step":0,"operation":"explicit_k16_quantization",
                "lowering":"native_block_quant_ssa","inputs":[0],"outputs":[1,2]}]):
        raise ValueError("block quantization byte spans, lifetimes or geometry differ")
    root = program.get("root_ir")
    if (not isinstance(root,str) or not root
            or hashlib.sha256(root.encode()).hexdigest() != program.get("root_ir_sha256")
            or json.loads(base64.b64decode(member["program_base64"],validate=True)) != program):
        raise ValueError("block quantization original source witness differs")
    for key in ("source_hash","artifact_hash","tile_body_hash","target_body_hash"):
        value = member.get(key)
        if not isinstance(value,str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("block quantization native seal differs")
    if not isinstance(member.get("entry"),str) or not member["entry"].startswith("tessera_block_quant_"):
        raise ValueError("block quantization native entry differs")


def _frame(text):
    program = _manifest(text,"tessera.rocm.block_quant_program_json")
    member = _manifest(text,"tessera.rocm.block_quant_member_json")
    _validate_frame(program,member)
    return program,member


def _descriptor(image,program,member,lineage):
    _validate_frame(program,member)
    names = member["bindings"]
    return LaunchDescriptor(image_digest=image.image_digest,entry_symbol=member["entry"],abi_id=BLOCK_QUANT_ABI,
        buffers=tuple(BufferBinding(i,name,"input" if i == 0 else "output",dtype,2,"row_major",width)
            for i,(name,dtype,width) in enumerate(zip(names,("bf16","fp8_e4m3","fp32"),(2,1,4),strict=True))),
        scalars=(ScalarArgument(3,"Groups","int64"),),
        shape_guards=tuple(ShapeGuard(name,axis,"eq",extent) for name,buffer in zip(names,program["buffers"],strict=True)
            for axis,extent in enumerate(buffer["shape"])),
        geometry=LaunchGeometry(grid=tuple(member["geometry"][:3]),workgroup=tuple(member["geometry"][3:])),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",)),
        provenance={"work_item":"ROCM-NVFP4-INGEST-1","sync_key":"ROCM-K16-QUANT-SCHEDULE-20261017",
            "route":"canonical_native_block_quant_schedule","native_block_quant":{
                "program_json":json.dumps(program,sort_keys=True),"member_json":json.dumps(member,sort_keys=True)},**lineage})


def _lineage(graph,schedule,tile):
    return {key+"_digest":hashlib.sha256(text.encode()).hexdigest()
        for key,text in (("graph",graph),("schedule_ir",schedule),("tile_ir",tile))}


def package_block_quant_recipe(recipe: BlockQuantRecipe,*,pipeline_name="tessera-rocm-executable"):
    from .rocm_native import ROCMNativePackage,_compile_native_tile_ir
    if pipeline_name != "tessera-rocm-executable":
        raise ValueError("block quantization image producer differs")
    recipe.validate()
    target,backend,payload,compiler,toolchain,libraries,state = _compile_native_tile_ir(
        recipe.tile_ir,directive="tessera_rocm.row_region_kernel",family="block_quant",architecture="gfx1201")
    program,member = _frame(target)
    if _frame(backend) != (program,member):
        raise ValueError("block quantization binary frame differs from native Target")
    image = NativeImageArtifact(target=_TARGET,architecture="gfx1201",pipeline_name=pipeline_name,
        compiler_fingerprint=compiler,toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),binary_format="hsaco",payload=payload,
        entry_points=(NativeEntryPoint(member["entry"],BLOCK_QUANT_ABI),),compile_state=state,device_libraries=libraries)
    return ROCMNativePackage(recipe.tile_ir,target,backend,image,
        _descriptor(image,program,member,_lineage(recipe.graph_ir,recipe.schedule_ir,recipe.tile_ir)))


@dataclass(frozen=True)
class BlockQuantProgram:
    program_json: str
    members_json: tuple[str,...]
    images: tuple[bytes,...]
    graph_ir: str

    def validate(self):
        if len(self.members_json) != 1 or len(self.images) != 1 or not self.graph_ir:
            raise ValueError("block quantization package ownership differs")
        _validate_frame(json.loads(self.program_json),json.loads(self.members_json[0]))
        if not self.images[0].startswith(b"\x7fELF"):
            raise ValueError("block quantization requires native HSACO")


class PreparedBlockQuantProgram(PreparedScaledProgram):
    _prepare_symbol = "tessera_rocm_program_prepare_multi_output"
    _step_type = _MultiOutputStep

    def _bind_abi(self,package):
        package.validate()
        program,member = json.loads(package.program_json),json.loads(package.members_json[0])
        rows = program["buffers"]
        blob = C.create_string_buffer(package.images[0]);entry = member["entry"].encode()
        steps = (_MultiOutputStep*1)();step = steps[0]
        step.image,step.image_bytes,step.entry = C.cast(blob,C.c_void_p),len(package.images[0]),entry
        step.input_count,step.output_count,step.scalar_count = 1,2,1
        step.inputs[0]=0;step.outputs[0]=1;step.outputs[1]=2
        step.scalars[0]=member["scalars"][0];step.geometry[:]=member["geometry"]
        buffers = (_Buffer*3)(*(_Buffer(b["bytes"],b["elements"],b["first_write"],b["last_read"],b["ownership"],0) for b in rows))
        storage = tuple(_StorageSpec(tuple(b["shape"]),b["storage"],b["bytes"],_storage_numpy_dtype(b["storage"])) for b in rows)
        return _BoundScaledABI(buffers,steps,(blob,),(entry,),1,storage,(1,2),())


def validate_block_quant_runtime_artifact(artifact):
    image,desc = artifact.native_image,artifact.launch_descriptor
    program,member = _frame(artifact.target_ir)
    if (image.target != _TARGET or image.architecture != "gfx1201" or image.binary_format != "hsaco"
            or image.pipeline_name != "tessera-rocm-executable"
            or image.target_ir_digest != hashlib.sha256(artifact.target_ir.encode()).hexdigest()
            or image.entry_points != (NativeEntryPoint(member["entry"],BLOCK_QUANT_ABI),)
            or desc != _descriptor(image,program,member,_lineage(artifact.graph_ir,artifact.schedule_ir,artifact.tile_ir))):
        raise ValueError("block quantization artifact stage/ABI lineage differs")
    package = BlockQuantProgram(json.dumps(program,sort_keys=True),(json.dumps(member,sort_keys=True),),
                                (image.payload,),artifact.graph_ir)
    package.validate()
    return package


def submit_block_quant(image,descriptor,buffers,scalars,stream):
    import numpy as np
    import ml_dtypes
    from tessera import runtime as rt
    if (stream not in (None,0) or rt._rocm_live_arch() != "gfx1201"
            or image.target != _TARGET or image.architecture != "gfx1201"
            or image.binary_format != "hsaco" or image.pipeline_name != "tessera-rocm-executable"):
        raise ValueError("block quantization requires owning gfx1201 and its owned completion stream")
    info = descriptor.provenance["native_block_quant"]
    program,member = json.loads(info["program_json"]),json.loads(info["member_json"])
    _validate_frame(program,member)
    if (set(buffers) != set(member["bindings"]) or set(scalars) != {"Groups"}
            or type(scalars["Groups"]) is not int or [scalars["Groups"]] != member["scalars"]):
        raise ValueError("block quantization invocation roles/count differ")
    arrays = tuple(buffers[name] for name in member["bindings"])
    for array,row,dtype in zip(arrays,program["buffers"],(ml_dtypes.bfloat16,ml_dtypes.float8_e4m3fn,np.float32),strict=True):
        if (not isinstance(array,np.ndarray) or array.dtype != np.dtype(dtype) or list(array.shape) != row["shape"]
                or not array.flags.c_contiguous or array.nbytes != row["bytes"]):
            raise ValueError("block quantization invocation storage/capacity differs")
    if any(not a.flags.writeable for a in arrays[1:]) or any(np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:]):
        raise ValueError("block quantization outputs must be writable and independent")
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise RuntimeError("block quantization requires checked HIP program runtime")
    package = BlockQuantProgram(json.dumps(program,sort_keys=True),(json.dumps(member,sort_keys=True),),(image.payload,),program["root_ir"])
    with PreparedBlockQuantProgram(package,[arrays[0]],runtime_library=lib._name) as owner:
        generation,_ = owner.invoke()
        for out,value in zip(arrays[1:],owner.read(generation),strict=True):
            np.copyto(out,value,casting="no")
    return arrays[1:]
