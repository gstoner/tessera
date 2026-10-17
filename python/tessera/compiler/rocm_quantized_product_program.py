"""Native SSA packaging for a resident BF16 quantizer and exact scaled product.

C++ owns source admission, partitioning, lifetimes and all numerical lowering.
Python validates the exported frame and marshals existing native member ABIs.
"""
from __future__ import annotations

import base64
import ctypes as C
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

from .native_artifact import LaunchGeometry, ScalarArgument
from .native_scaled_program import PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec, _storage_numpy_dtype
from .rocm_block_quant_native import BlockQuantRecipe, lower_block_quant_text, package_block_quant_recipe, validate_block_quant_runtime_artifact, _frame
from .rocm_fp8_blockscale import BlockScaleProgram, BlockScaleShape, PACKAGE_ABIS, check_blockscale_target_ir, package_blockscale, verify_blockscale_schedule
from .rocm_native import ROCMNativePackage, _directive_symbol
from .rocm_nvfp4_expansion_program import _MultiOutputStep
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

_OPTION = "--tessera-autodiff-forward=export-scaled-primal=true"
_SCHEMA = "tessera.native.quantized_product.v1"


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _export(tool, source):
    text = run_tessera_opt(tool,source,_OPTION)
    encoded = re.findall(r'tessera.native.quantized_product_json = "([^"\n]+)"',text)
    if len(encoded) != 1:
        raise ValueError("requires one native quantized-product SSA record")
    plan = json.loads(base64.b64decode(encoded[0],validate=True))
    if not isinstance(plan,dict) or plan.get("schema") != _SCHEMA:
        raise ValueError("native quantized-product schema differs")
    return plan


def _frame_shape(plan):
    if (not isinstance(plan,dict) or set(plan) != {"schema","source_graph_ir","member_graphs","buffers","steps","output"}
            or plan["schema"] != _SCHEMA or type(plan["output"]) is not int or plan["output"] != 5
            or not isinstance(plan["source_graph_ir"],str) or not plan["source_graph_ir"]
            or not isinstance(plan["member_graphs"],list) or len(plan["member_graphs"]) != 2
            or not all(isinstance(v,str) and v for v in plan["member_graphs"])
            or not isinstance(plan["steps"],list) or len(plan["steps"]) != 2
            or not isinstance(plan["buffers"],list) or len(plan["buffers"]) != 6):
        raise ValueError("native quantized-product frame differs")
    steps,rows = plan["steps"],plan["buffers"]
    if (not all(isinstance(step,dict) for step in steps)
            or steps[0].get("operation") != "native_block_quant"
            or steps[1].get("operation") != "tessera.scaled_matmul"
            or not isinstance(steps[0].get("inputs"),list) or len(steps[0]["inputs"]) != 1
            or not isinstance(steps[1].get("inputs"),list) or len(steps[1]["inputs"]) != 4):
        raise ValueError("native quantized-product step roles differ")
    roles = [steps[0]["inputs"][0],steps[1]["inputs"][1],steps[1]["inputs"][3]]
    if any(type(v) is not int for v in roles) or sorted(roles) != [0,1,2]:
        raise ValueError("native quantized-product original arguments differ")
    for i,step in enumerate(steps):
        if (set(step) != {"step","operation","member","inputs","outputs"}
                or type(step["step"]) is not int or step["step"] != i
                or not isinstance(step["member"],str) or not step["member"]
                or not isinstance(step["outputs"],list)
                or any(type(v) is not int for v in step["inputs"]+step["outputs"])):
            raise ValueError("native quantized-product step encoding differs")
    if (steps[0]["outputs"] != [3,4] or steps[1]["outputs"] != [5]
            or steps[1]["inputs"] != [3,roles[1],4,roles[2]]):
        raise ValueError("native quantized-product lost its two private producer edges")
    if not all(isinstance(row,dict) for row in rows):
        raise ValueError("native quantized-product buffer rows differ")
    xshape = rows[roles[0]].get("shape")
    bshape = rows[roles[1]].get("shape")
    if (not isinstance(xshape,list) or len(xshape) != 2
            or not isinstance(bshape,list) or len(bshape) != 2
            or any(type(d) is not int or d<=0 for d in xshape+bshape)):
        raise ValueError("native quantized-product capacity differs")
    m,k = xshape
    n,bk = bshape
    if bk != k or k%16 or m*(k//16)>2147483647:
        raise ValueError("native quantized-product K16 capacity differs")
    out = rows[5].get("storage")
    if not isinstance(out,str) or out not in {"f32","bf16"}:
        raise ValueError("native quantized-product output storage differs")
    specs: list[tuple[list[int],str,int,int] | None] = [None]*6
    specs[roles[0]] = ([m,k],"bf16",-1,0)
    specs[roles[1]] = ([n,k],"f8E4M3FN",-1,1)
    specs[roles[2]] = ([k//16,n],"f32",-1,1)
    specs[3:] = [([m,k],"f8E4M3FN",0,1),([m,k//16],"f32",0,1),([m,n],out,1,2)]
    sizes = {"bf16":2,"f8E4M3FN":1,"f32":4}
    for i,(row,spec) in enumerate(zip(rows,specs,strict=True)):
        if spec is None:
            raise ValueError("native quantized-product missing buffer role")
        shape,storage,write,read = spec
        expected = {"id":i,"shape":shape,"storage":storage,"bytes":math.prod(shape)*sizes[storage],
                    "ownership":"readonly_input" if i<3 else "returned_output" if i==5 else "private_scratch",
                    "first_write":write,"last_read":read}
        if (row != expected or any(type(row.get(key)) is not int for key in ("id","bytes","first_write","last_read"))
                or any(type(d) is not int for d in row["shape"])):
            raise ValueError("native quantized-product buffer storage/lifetime differs")
    return roles,BlockScaleShape(m,n,k,16,1,"nk",out)


@dataclass(frozen=True)
class NativeQuantizedProductProgram:
    program_json: str
    graph_ir: str
    quantizer_recipe: BlockQuantRecipe
    quantizer: ROCMNativePackage
    product_recipe: BlockScaleProgram
    product: ROCMNativePackage
    checked_product_target_ir: str
    binding_digest: str

    def _identity(self):
        parts = [self.program_json,self.graph_ir,self.quantizer_recipe.graph_ir,
                 self.quantizer_recipe.schedule_ir,self.quantizer_recipe.tile_ir,
                 self.product_recipe.graph_ir,self.product_recipe.schedule_ir,self.product_recipe.tile_ir,
                 self.checked_product_target_ir]
        for native in self.native_stages:
            parts.extend([native.tile_ir,native.target_ir,
                          native.image.image_digest,native.descriptor.descriptor_digest])
        return _digest(json.dumps(parts,separators=(",",":")))

    @property
    def native_stages(self):
        return (self.quantizer,self.product)

    def validate(self):
        if self.binding_digest != self._identity():
            raise ValueError("native quantized-product member binding differs")
        plan = json.loads(self.program_json)
        roles,shape = _frame_shape(plan)
        if (plan["source_graph_ir"] != self.graph_ir
                or plan["member_graphs"] != [self.quantizer_recipe.graph_ir,self.product_recipe.graph_ir]
                or self.product_recipe.shape != shape
                or self.quantizer.tile_ir != self.quantizer_recipe.tile_ir
                or self.product.tile_ir != self.product_recipe.tile_ir):
            raise ValueError("native quantized-product projected Graph differs")
        from tessera.runtime import RuntimeArtifact
        quant = self.quantizer
        quant_bound=validate_block_quant_runtime_artifact(RuntimeArtifact(
            graph_ir=self.quantizer_recipe.graph_ir,schedule_ir=self.quantizer_recipe.schedule_ir,
            tile_ir=quant.tile_ir,target_ir=quant.target_ir,native_image=quant.image,
            launch_descriptor=quant.descriptor,
            metadata={"target":"rocm_gfx1201","compiler_path":"rocm_block_quant_native_descriptor"}))
        quant_buffers=json.loads(quant_bound.program_json)["buffers"]
        for local,shared in zip(quant_buffers,(plan["buffers"][slot] for slot in (roles[0],3,4)),strict=True):
            if any(local[key]!=shared[key] for key in ("shape","storage","bytes")):
                raise ValueError("native quantized-product producer/shared buffer interface differs")
        native = self.product
        native.descriptor.validate_image(native.image)
        checked = check_blockscale_target_ir(shape,native.tile_ir,self.checked_product_target_ir)
        descriptor = native.descriptor
        if (native.image.target != "rocm_gfx1201" or native.image.architecture != "gfx1201"
                or native.image.binary_format != "hsaco" or native.image.pipeline_name != "tessera-rocm-executable"
                or native.image.target_ir_digest != _digest(native.target_ir)
                or descriptor.abi_id != PACKAGE_ABIS[("nk",shape.output)]
                or descriptor.entry_symbol != _directive_symbol(native.target_ir,"tessera_rocm.scaled_wmma_gemm")
                or descriptor.geometry != LaunchGeometry(policy="rocm_wmma_macro_tile_grid")
                or descriptor.scalars != tuple(ScalarArgument(i+5,name,"int64") for i,name in enumerate(("M","N","K")))
                or descriptor.provenance["shape"] != [shape.m,shape.n,shape.k]
                or descriptor.provenance["scale_k"] != 16 or descriptor.provenance["scale_n"] != 1
                or descriptor.provenance["macro_tile"] != [checked["block_m"],checked["block_n"]]
                or descriptor.provenance["workgroup"] != [32*checked["warps"],1,1]):
            raise ValueError("native quantized-product consumer ABI/geometry differs")

    def to_manifest(self):
        from .rocm_nvfp4_resident import _portable_stage
        self.validate()
        return {"schema":"tessera.rocm.quantized_product_package.v1",
                "program_json":self.program_json,"graph_ir":self.graph_ir,
                "checked_product_target_ir":self.checked_product_target_ir,
                "binding_digest":self.binding_digest,
                "stages":[_portable_stage(self.quantizer_recipe.graph_ir,self.quantizer_recipe.schedule_ir,self.quantizer),
                          _portable_stage(self.product_recipe.graph_ir,self.product_recipe.schedule_ir,self.product)]}

    @classmethod
    def from_manifest(cls,record):
        from .native_artifact import NativeImageArtifact,LaunchDescriptor
        if (not isinstance(record,dict) or set(record)!={"schema","program_json","graph_ir","checked_product_target_ir","binding_digest","stages"}
                or record["schema"]!="tessera.rocm.quantized_product_package.v1"
                or any(not isinstance(record[key],str) for key in ("program_json","graph_ir","checked_product_target_ir","binding_digest"))
                or not isinstance(record["stages"],list) or len(record["stages"])!=2):
            raise ValueError("native quantized-product portable record differs")
        _,shape=_frame_shape(json.loads(record["program_json"]))
        stages=[]
        for stage in record["stages"]:
            if (not isinstance(stage,dict) or set(stage)!={"graph_ir","schedule_ir","tile_ir","target_ir","image","descriptor"}
                    or any(not isinstance(stage[key],str) for key in ("graph_ir","schedule_ir","tile_ir","target_ir"))):
                raise ValueError("native quantized-product portable member differs")
            native=ROCMNativePackage(stage["tile_ir"],stage["target_ir"],"",
                NativeImageArtifact.from_dict(stage["image"]),LaunchDescriptor.from_dict(stage["descriptor"]))
            stages.append((stage["graph_ir"],stage["schedule_ir"],native))
        first,second=stages
        program=cls(record["program_json"],record["graph_ir"],
            BlockQuantRecipe(first[0],first[1],first[2].tile_ir),first[2],
            BlockScaleProgram(shape,"quantized_product",second[0],second[1],second[2].tile_ir),second[2],
            record["checked_product_target_ir"],record["binding_digest"])
        program.validate()
        return program

    def prepare(self, inputs, *, runtime_library):
        return PreparedQuantizedProductProgram(self,inputs,runtime_library=runtime_library)


def package_quantized_product_graph(graph_ir):
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native quantized product requires matching tessera-opt")
    plan = _export(tool,graph_ir)
    _,shape = _frame_shape(plan)
    # Replay normalized source before accepting a package witness.
    if _export(tool,plan["source_graph_ir"]) != plan:
        raise ValueError("native quantized-product normalized source replay differs")
    quantizer_recipe = lower_block_quant_text(plan["member_graphs"][0])
    quantizer = package_block_quant_recipe(quantizer_recipe)
    child = plan["member_graphs"][1]
    schedule = run_tessera_opt(tool,child,"--tessera-graph-to-schedule")
    verify_blockscale_schedule(shape,schedule)
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    recipe = BlockScaleProgram(shape,"quantized_product",child,schedule,tile)
    product = package_blockscale(recipe,pipeline_name="tessera-rocm-executable",project_image_identity=False)
    program = NativeQuantizedProductProgram(json.dumps(plan,sort_keys=True),plan["source_graph_ir"],
        quantizer_recipe,quantizer,recipe,product,product.target_ir,"")
    program = replace(program,binding_digest=program._identity())
    program.validate()
    return program


class PreparedQuantizedProductProgram(PreparedScaledProgram):
    _prepare_symbol = "tessera_rocm_program_prepare_multi_output"
    _step_type = _MultiOutputStep

    def __init__(self,package,inputs,*,runtime_library):
        from tessera import runtime as rt
        if rt._rocm_live_arch() != "gfx1201":
            raise RuntimeError("native quantized product requires owning gfx1201")
        super().__init__(package,inputs,runtime_library=runtime_library)

    def _bind_abi(self,package):
        package.validate()
        plan = json.loads(package.program_json)
        _,shape = _frame_shape(plan)
        rows,native = plan["buffers"],package.native_stages
        blobs = tuple(C.create_string_buffer(p.image.payload) for p in native)
        entries = tuple(p.descriptor.entry_symbol.encode() for p in native)
        steps = (_MultiOutputStep*len(native))()
        for i,p in enumerate(native):
            step,spec = steps[i],plan["steps"][i]
            step.image,step.image_bytes,step.entry = C.cast(blobs[i],C.c_void_p),len(p.image.payload),entries[i]
            step.input_count,step.output_count = len(spec["inputs"]),len(spec["outputs"])
            step.inputs[:step.input_count],step.outputs[:step.output_count] = spec["inputs"],spec["outputs"]
            if i==0:
                _,member = _frame(p.target_ir)
                scalars,geometry = member["scalars"],member["geometry"]
            else:
                bm,bn = p.descriptor.provenance["macro_tile"]
                scalars = [shape.m,shape.n,shape.k]
                geometry = [(shape.n+bn-1)//bn,(shape.m+bm-1)//bm,1,*p.descriptor.provenance["workgroup"]]
            step.scalar_count = len(scalars)
            step.scalars[:step.scalar_count],step.geometry[:] = scalars,geometry
        ownership = {"readonly_input":0,"private_scratch":1,"returned_output":2}
        buffers = (_Buffer*len(rows))(*[
            _Buffer(b["bytes"],math.prod(b["shape"]),b["first_write"],b["last_read"],ownership[b["ownership"]],0)
            for b in rows])
        storage = tuple(_StorageSpec(tuple(b["shape"]),b["storage"],b["bytes"],_storage_numpy_dtype(b["storage"])) for b in rows)
        return _BoundScaledABI(buffers,steps,blobs,entries,3,storage,(plan["output"],),())

    def invoke_range(self,first,count,repeats=1,*,timed=True):
        if (type(first) is not int or type(count) is not int or first<0 or count<=0
                or first+count>len(self._binding.steps) or type(repeats) is not int
                or not 1<=repeats<=1048576):
            raise ValueError("native quantized-product invocation range differs")
        from .native_scaled_program import _status
        invoke=self.lib.tessera_rocm_program_invoke_range
        invoke.argtypes=[C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_float)]
        invoke.restype=C.c_int
        generation,elapsed=C.c_uint64(),C.c_float()
        _status(invoke(self.handle,first,count,repeats,C.byref(generation),C.byref(elapsed) if timed else None))
        return generation.value,float(elapsed.value) if timed else None
