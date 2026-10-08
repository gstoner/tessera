"""Native ROCm math recipes and checked descriptor projection.

Python binds frontend buffers and decodes the verified native record; MLIR owns
the Schedule, Tile construction and physical image identity.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

MATH_ABIS = {
    "unary": "tessera.rocm.math.x_o_n.f32.v1",
    "binary": "tessera.rocm.math.a_b_o_n.f32.v1",
    "scan": "tessera.rocm.math.x_o_rows_columns.f32.v1",
}
for _family, _signature in (("unary","x_o_n"),("binary","a_b_o_n"),("scan","x_o_rows_columns")):
    for _storage in ("f16","bf16"):
        MATH_ABIS[_family+"_"+_storage] = "tessera.rocm.math."+_signature+"."+_storage+"_f32.v1"


def math_abi(info):
    key = info["family"] if info["storage"] == "f32" else info["family"]+"_"+info["storage"]
    return MATH_ABIS[key]


OPS = frozenset("tessera."+x for x in ("sqrt","exp","add","div","cumsum","cummax"))


def requests_math(module):
    if len(module.functions) != 1 or not module.functions[0].body:
        return False
    body = module.functions[0].body
    return body[-1].op_name in OPS and all(op.op_name == "tessera.cast" for op in body[:-1])


def supports_math(module):
    if not requests_math(module):
        return False
    fn = module.functions[0]
    return (bool(fn.args) and len(fn.result_types) == 1
            and all(arg.ir_type.dtype in {"fp32","fp16","bf16"} for arg in fn.args)
            and fn.result_types[0].dtype == "fp32")


def project_math_graph(module, target):
    if target not in {"rocm_gfx1151","rocm_gfx1201"} or not requests_math(module):
        raise ValueError("native math requires one named ROCm Graph operation")
    source = deepcopy(module)
    for key, value in (("tessera.target","rocm"),("tessera.arch",target.removeprefix("rocm_"))):
        existing = source.module_attrs.get(key)
        if existing is not None and json.loads(existing) not in ({value, target} if key == "tessera.target" else {value}):
            raise ValueError("native math Graph target/architecture conflicts")
        source.module_attrs[key] = json.dumps(value)
    fn = source.functions[0]
    op = fn.body[-1]
    names = [arg.name for arg in fn.args]+op.result_names
    if (len(op.result_names) != 1 or len(set(names)) != len(names)
            or fn.return_values != ["%"+op.result_names[0]]):
        raise ValueError("native math Graph must return its isolated result")
    existing = source.module_attrs.get("tessera.launch_bindings")
    if existing is not None and json.loads(existing) != names:
        raise ValueError("native math Graph binding aliases conflict")
    source.module_attrs["tessera.launch_bindings"] = json.dumps(names)
    return source


def _info(tile):
    records = re.findall(r'tessera.rocm_math_contract = \{([^{}]+)\}', tile)
    entries = re.findall(r'llvm.func @([A-Za-z0-9_]+)\(', tile)
    if len(records) != 1 or len(entries) != 1:
        raise ValueError("native math requires one Tile ownership record")
    c = records[0]
    def text(key):
        match = re.search(r'\b'+key+r' = "([^"]+)"', c)
        if match is None: raise ValueError("native math missing "+key)
        return match[1]
    def integer(key):
        match = re.search(r'\b'+key+r' = ([0-9]+) : i64', c)
        if match is None: raise ValueError("native math missing "+key)
        return int(match[1])
    shape_match = re.search(r'shape = array<i64: ([0-9, ]+)>', c)
    names_match = re.search(r'bindings = (\[[^\]]+\])', c)
    roles_match = re.search(r'roles = (\[[^\]]+\])', c)
    if shape_match is None or names_match is None or roles_match is None:
        raise ValueError("native math lost shape/binding/role record")
    shape = tuple(int(x) for x in shape_match[1].split(","))
    names = tuple(json.loads(names_match[1]))
    roles = tuple(json.loads(roles_match[1]))
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 2 or hashes[0] != hashes[1]:
        raise ValueError("native math Tile hash lineage differs")
    info = dict(architecture=text("architecture"),family=text("family"),kind=text("kind"),
                storage=text("storage"),output_storage=text("output_storage"),
                numeric_policy=text("numeric_policy"),shape=shape,bindings=names,roles=roles,
                elements=integer("elements"),rows=integer("rows"),columns=integer("columns"),
                schedule_digest=hashes[0])
    _validate_info(info)
    return info


def _validate_info(info):
    shape, names, roles = info["shape"], info["bindings"], info["roles"]
    family, kind = info["family"], info["kind"]
    allowed = {"unary":{"sqrt","exp"}, "binary":{"add","div"}, "scan":{"sum","max"}}
    arity = 2 if family == "binary" else 1
    if (family not in allowed or kind not in allowed[family]
            or info["architecture"] not in {"gfx1151","gfx1201"}
            or info["storage"] not in {"f32","f16","bf16"} or info["output_storage"] != "f32"
            or info["numeric_policy"] != ("f32_inclusive_scan" if family == "scan" else "f32_compute")
            or not shape or any(type(x) is not int or x <= 0 for x in shape)
            or math.prod(shape) > 2**63-1
            or len(names) != arity+1 or len(set(names)) != len(names)
            or any(type(x) is not str or not x for x in names)
            or len(roles) != arity or any(type(x) is not int or not 0 <= x < arity for x in roles)
            or any(type(info[x]) is not int for x in ("elements","rows","columns"))
            or info["elements"] != math.prod(shape) or info["columns"] != shape[-1]
            or info["rows"] != math.prod(shape[:-1])
            or re.fullmatch(r"[0-9a-f]{64}",info["schedule_digest"]) is None):
        raise ValueError("native math descriptor ownership record differs")
    # Public isolated binary entries bind each argument exactly once.
    if len(set(roles)) != arity:
        raise ValueError("native math requires distinct operand roles")


@dataclass(frozen=True)
class MathRecipe:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str

    def validate(self):
        tool = find_tessera_opt()
        if tool is None: raise RuntimeError("native math construction requires tessera-opt")
        from .rocm_pass_cache import run as cached_replay
        def replay(source, option):
            return cached_replay(tool, source, option, execute=run_tessera_opt)
        schedule = replay(self.graph_ir,"--tessera-graph-to-schedule")
        if schedule != self.schedule_ir:
            raise ValueError("native math Schedule differs from original Graph replay")
        if replay(schedule,"--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("native math Tile differs from Schedule replay")
        if self.target != "rocm_"+_info(self.tile_ir)["architecture"]:
            raise ValueError("native math target differs from the native contract")


def lower_math_graph(module, target):
    source = project_math_graph(module,target)
    graph = source.to_mlir(target=target,canonical=True)
    tool = find_tessera_opt()
    if tool is None: raise RuntimeError("native math requires production tessera-opt")
    schedule = run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    recipe = MathRecipe(graph,schedule,tile,target)
    recipe.validate()
    return recipe


def _descriptor(image, info, lineage):
    _validate_info(info)
    names, roles, shape = info["bindings"],info["roles"],info["shape"]
    scan = info["family"] == "scan"
    # Generated GPU ABI consumes canonical operand order. Alias names retain
    # original Graph argument identity even when authored operands are reversed.
    ordered = tuple(names[role] for role in roles)+(names[-1],)
    count = len(ordered)
    scalar_names = ("Rows","Columns") if scan else ("N",)
    return LaunchDescriptor(
        image_digest=image.image_digest,entry_symbol=image.entry_points[0].symbol,
        abi_id=math_abi(info),
        buffers=tuple(BufferBinding(i,name,"output" if i == count-1 else "input",
                                    "fp32" if i == count-1 else {"f32":"fp32","f16":"fp16","bf16":"bf16"}[info["storage"]],
                                    len(shape),"row_major",4 if i == count-1 or info["storage"]=="f32" else 2)
                      for i,name in enumerate(ordered)),
        scalars=tuple(ScalarArgument(count+i,name,"int64") for i,name in enumerate(scalar_names)),
        shape_guards=tuple(ShapeGuard(name,i,"eq",extent) for name in ordered
                           for i,extent in enumerate(shape)),
        geometry=LaunchGeometry(grid=(info["rows"] if scan else (info["elements"]+255)//256,1,1),
                                workgroup=(256,1,1)),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",)),
        provenance={"work_item":"E2E-REAL-6","sync_key":"ROCM-MATH-PACKAGE-2026-10-06",
                    "route":"canonical_native_math_schedule",
                    "native_math":{k:list(v) if isinstance(v,tuple) else v for k,v in info.items()},
                    **lineage})


def package_math_recipe(recipe, *, pipeline_name):
    from .rocm_native import ROCMNativePackage,_compile_shape_free_tile_ir
    recipe.validate()
    info = _info(recipe.tile_ir)
    family = {"unary":"scalar_unary","binary":"scalar_binary","scan":"scan"}[info["family"]]
    target,backend,payload,compiler,toolchain,libraries,state = _compile_shape_free_tile_ir(
        recipe.tile_ir,family=family,architecture=info["architecture"])
    directive = "tessera_rocm."+info["family"]
    from .rocm_native import _directive_symbol
    entry = _directive_symbol(target,directive)
    image = NativeImageArtifact(
        target=recipe.target,architecture=info["architecture"],pipeline_name=pipeline_name,
        compiler_fingerprint=compiler,toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),binary_format="hsaco",
        payload=payload,entry_points=(NativeEntryPoint(entry,math_abi(info)),),
        compile_state=state,device_libraries=libraries)
    lineage = {stage+"_digest":hashlib.sha256(text.encode()).hexdigest()
               for stage,text in (("graph",recipe.graph_ir),("schedule_ir",recipe.schedule_ir),
                                  ("tile_ir",recipe.tile_ir))}
    return ROCMNativePackage(recipe.tile_ir,target,backend,image,_descriptor(image,info,lineage))


def validate_math_runtime_artifact(artifact):
    image, desc = artifact.native_image, artifact.launch_descriptor
    info = _info(artifact.tile_ir)
    lineage = {stage+"_digest":hashlib.sha256(text.encode()).hexdigest()
               for stage,text in (("graph",artifact.graph_ir),("schedule_ir",artifact.schedule_ir),
                                  ("tile_ir",artifact.tile_ir))}
    if (image.target != "rocm_"+info["architecture"] or image.architecture != info["architecture"]
            or image.binary_format != "hsaco" or len(image.entry_points) != 1
            or image.target_ir_digest != hashlib.sha256(artifact.target_ir.encode()).hexdigest()
            or desc != _descriptor(image,info,lineage)):
        raise ValueError("native math artifact stage/descriptor lineage differs")
    # Portable validation uses persisted native metadata, never a compiler subprocess.
    family = info["family"]
    expected_kind = ("cumsum" if info["kind"] == "sum" else "cummax") if family == "scan" else info["kind"]
    directives = re.findall(r'tessera_rocm\.'+family+r' \{([^{}]*)\}', artifact.target_ir)
    if len(directives) != 1:
        raise ValueError("native math requires one Target arithmetic directive")
    attrs = directives[0]
    kinds = re.findall(r'kind = "([^"]+)"', attrs)
    dtypes = re.findall(r'\bdtype = "([^"]+)"', attrs)
    output_dtypes = re.findall(r'\boutput_dtype = "([^"]+)"', attrs)
    # ODS canonical printing omits default-valued attributes.
    actual_kind = kinds[0] if len(kinds) == 1 else {"unary":"exp","binary":"sub","scan":"cumsum"}[family]
    if (len(kinds) > 1 or actual_kind != expected_kind
            or len(dtypes) > 1 or (dtypes[0] if dtypes else "f32") != info["storage"]
            or output_dtypes != ["f32"]
            or f'name = "{desc.entry_symbol}"' not in attrs):
        raise ValueError("native math Target arithmetic differs from Tile")


def runtime_projection(image, desc, buffers, scalars):
    import numpy as np
    info = desc.provenance["native_math"]
    lineage = {key:desc.provenance[key] for key in ("graph_digest","schedule_ir_digest","tile_ir_digest")}
    desc.validate_image(image)
    if (image.target != "rocm_"+info["architecture"] or image.architecture != info["architecture"]
            or image.binary_format != "hsaco" or len(image.entry_points) != 1
            or desc != _descriptor(image,info,lineage)):
        raise ValueError("native math descriptor differs from checked ABI")
    ordered = sorted(desc.buffers,key=lambda b:b.ordinal)
    if set(buffers) != {b.name for b in ordered}:
        raise ValueError("native math needs complete buffer bindings")
    scan = info["family"] == "scan"
    wanted = {"Rows":info["rows"],"Columns":info["columns"]} if scan else {"N":info["elements"]}
    if set(scalars) != set(wanted) or any(type(scalars[k]) is not int or scalars[k] != v for k,v in wanted.items()):
        raise ValueError("native math scalars differ from the compiled shape")
    arrays = [buffers[b.name] for b in ordered]
    input_dtype = np.dtype(np.float32 if info["storage"]=="f32" else np.float16)
    if info["storage"]=="bf16":
        import ml_dtypes
        input_dtype = np.dtype(ml_dtypes.bfloat16)
    if any(not isinstance(a,np.ndarray) or a.dtype != (np.dtype(np.float32) if i==len(arrays)-1 else input_dtype)
           or a.shape != tuple(info["shape"]) or not a.flags.c_contiguous for i,a in enumerate(arrays)):
        raise ValueError("native math requires exact compact input storage and f32 output")
    if not arrays[-1].flags.writeable or any(np.shares_memory(arrays[-1],a) for a in arrays[:-1]):
        raise ValueError("native math output must be writeable and independent of inputs")
    return arrays[:-1],arrays[-1],tuple(wanted.values()),desc.geometry.grid[0]
