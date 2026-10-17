"""Native parent-view compilation and checked program-frame marshalling.

The original Graph crosses registered native Schedule/Tile/Target passes.
Python binds the compiler-exported frame and never constructs kernel IR.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)
from .native_scaled_program import _manifest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

PARENT_VIEW_ABI = "tessera.rocm.parent_view.parent_o_n.f32.v1"
_TARGET = "rocm_gfx1201"


def requests_parent_view(module: GraphIRModule) -> bool:
    if len(module.functions) != 1:
        return False
    fn = module.functions[0]
    return (len(fn.args) == 1 and len(fn.result_types) == 1 and len(fn.body) == 3
            and all(op.op_name in {"tessera.slice", "tessera.dynamic_slice"} for op in fn.body[:2])
            and fn.body[-1].op_name == "tessera.silu_mul")


def supports_parent_view(module: GraphIRModule) -> bool:
    if not requests_parent_view(module):
        return False
    fn = module.functions[0]
    return (fn.args[0].ir_type.dtype == "fp32" and fn.args[0].ir_type.rank == 2
            and fn.result_types[0].dtype == "fp32" and fn.result_types[0].rank == 2)


def project_parent_view_graph(module: GraphIRModule, target: str) -> GraphIRModule:
    if target != _TARGET or not supports_parent_view(module):
        raise ValueError("native parent-view requires the named static gfx1201 f32 Graph")
    source = deepcopy(module)
    for key, value in (("tessera.target", "rocm"), ("tessera.arch", "gfx1201")):
        existing = source.module_attrs.get(key)
        if existing is not None and json.loads(existing) not in ({value, target} if key == "tessera.target" else {value}):
            raise ValueError("parent-view source target/architecture conflicts")
        source.module_attrs[key] = json.dumps(value)
    fn = source.functions[0]
    if len(fn.return_values) != 1 or fn.return_values != ["%" + fn.body[-1].result_names[0]]:
        raise ValueError("parent-view must return its original SiLU result")
    names = [fn.args[0].name, fn.body[-1].result_names[0]]
    if len(set(names)) != 2:
        raise ValueError("parent-view input/output bindings must be distinct")
    existing = source.module_attrs.get("tessera.launch_bindings")
    if existing is not None and json.loads(existing) != names:
        raise ValueError("parent-view source binding aliases conflict")
    marker = source.module_attrs.get("tessera.native.gate_up_view")
    if marker not in (None, "unit"):
        raise ValueError("parent-view source intent conflicts")
    source.module_attrs["tessera.launch_bindings"] = json.dumps(names)
    source.module_attrs["tessera.native.gate_up_view"] = "unit"
    return source


@dataclass(frozen=True)
class ParentViewRecipe:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str = _TARGET

    def validate(self) -> None:
        from .rocm_pass_cache import run
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("native parent-view construction requires matching tessera-opt")
        if self.target != _TARGET:
            raise ValueError("parent-view recipe architecture differs")
        schedule = run(tool, self.graph_ir, "--tessera-graph-to-schedule", execute=run_tessera_opt)
        if schedule != self.schedule_ir:
            raise ValueError("parent-view Schedule differs from original Graph replay")
        tile = run(tool, schedule, "--tessera-schedule-to-tile", execute=run_tessera_opt)
        if tile != self.tile_ir:
            raise ValueError("parent-view Tile differs from native Schedule replay")


def lower_parent_view_graph(module: GraphIRModule, target: str) -> ParentViewRecipe:
    source = project_parent_view_graph(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native parent-view requires production tessera-opt")
    graph = source.to_mlir(target=target, canonical=True)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    recipe = ParentViewRecipe(graph, schedule, tile, target)
    recipe.validate()
    return recipe


def _frame(target_ir: str) -> tuple[dict, dict]:
    program = _manifest(target_ir, "tessera.rocm.parent_view_program_json")
    member = _manifest(target_ir, "tessera.rocm.parent_view_member_json")
    _validate_frame(program, member)
    return program, member


def _validate_frame(program: dict, member: dict) -> None:
    if (program.get("schema") != 1 or program.get("kind") != "parent_view"
            or program.get("argument_count") != 1 or program.get("outputs") != [1]
            or member.get("schema") != 1 or member.get("step") != 0
            or member.get("architecture") != "gfx1201"
            or member.get("profile") != "gfx1201_parent_view_f32_v1"
            or member.get("abi") != "flattened_rank1_memref_then_i64"
            or member.get("inputs") != [0] or member.get("output") != 1):
        raise ValueError("parent-view native frame profile differs")
    shape = member.get("output_shape")
    names = member.get("bindings")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or not 0 < x <= 2**31-1 for x in shape)
            or not isinstance(names, list) or len(names) != 2
            or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != 2):
        raise ValueError("parent-view native shape/binding frame differs")
    rows, width = shape
    count = rows * width
    if (count > 2**31 // 12 or member.get("parent_shape") != [rows, 2*width]
            or member.get("scalars") != [count]
            or member.get("geometry") != [(count+255)//256, 1, 1, 256, 1, 1]):
        raise ValueError("parent-view capacity/count/geometry differs")
    expected_buffers = [
        {"shape": [rows, 2*width], "storage": "f32", "elements": 2*count,
         "bytes": 8*count, "first_write": -1, "last_read": 0, "ownership": 0},
        {"shape": shape, "storage": "f32", "elements": count,
         "bytes": 4*count, "first_write": 0, "last_read": 1, "ownership": 2},
    ]
    if (program.get("buffers") != expected_buffers
            or program.get("steps") != [{"step": 0, "operation": "tessera.silu_mul",
                "lowering": "native_parent_view", "inputs": [0], "output": 1}]):
        raise ValueError("parent-view native byte spans or SSA lifetimes differ")
    root = program.get("root_ir")
    if (not isinstance(root, str) or not root
            or hashlib.sha256(root.encode()).hexdigest() != program.get("root_ir_sha256")
            or json.loads(base64.b64decode(member["program_base64"], validate=True)) != program):
        raise ValueError("parent-view original source witness differs")
    for key in ("source_hash", "artifact_hash", "tile_body_hash", "target_body_hash"):
        value = member.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("parent-view native region seal differs")
    if not isinstance(member.get("entry"), str) or not member["entry"].startswith("tessera_gate_up_view_"):
        raise ValueError("parent-view native entry differs")


@dataclass(frozen=True)
class ParentViewProgram:
    program_json: str
    members_json: tuple[str, ...]
    images: tuple[bytes, ...]
    graph_ir: str

    def validate(self) -> None:
        if len(self.members_json) != 1 or len(self.images) != 1 or not self.graph_ir:
            raise ValueError("parent-view program ownership differs")
        _validate_frame(json.loads(self.program_json), json.loads(self.members_json[0]))
        if not self.images[0].startswith(b"\x7fELF"):
            raise ValueError("parent-view program requires a native HSACO")


def _program(image: NativeImageArtifact, program: dict, member: dict, graph_ir: str) -> ParentViewProgram:
    package = ParentViewProgram(json.dumps(program, sort_keys=True),
        (json.dumps(member, sort_keys=True),), (image.payload,), graph_ir)
    package.validate()
    return package


def _descriptor(image: NativeImageArtifact, program: dict, member: dict, lineage: dict) -> LaunchDescriptor:
    _validate_frame(program, member)
    names = member["bindings"]
    return LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=member["entry"], abi_id=PARENT_VIEW_ABI,
        buffers=tuple(BufferBinding(i, name, "input" if i == 0 else "output", "fp32", 2,
                                    "row_major", 4) for i, name in enumerate(names)),
        scalars=(ScalarArgument(2, "N", "int64"),),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", extent)
            for name, row in zip(names, program["buffers"], strict=True)
            for axis, extent in enumerate(row["shape"])),
        geometry=LaunchGeometry(grid=tuple(member["geometry"][:3]),
                                workgroup=tuple(member["geometry"][3:])),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={"work_item": "ROCM-NVFP4-INGEST-1",
            "sync_key": "ROCM-NATIVE-GATE-UP-CONSUMER-20261016",
            "route": "canonical_native_parent_view_schedule",
            "native_parent_view": {"program_json": json.dumps(program, sort_keys=True),
                                   "member_json": json.dumps(member, sort_keys=True)}, **lineage})


def package_parent_view_recipe(recipe: ParentViewRecipe, *, pipeline_name: str):
    if pipeline_name != "tessera-rocm-executable":
        raise ValueError("parent-view image producer must be tessera-rocm-executable")
    from .rocm_native import ROCMNativePackage, _compile_native_tile_ir
    recipe.validate()
    target, backend, payload, compiler, toolchain, libraries, state = _compile_native_tile_ir(
        recipe.tile_ir, directive="tessera_rocm.row_region_kernel", family="parent_view", architecture="gfx1201")
    program, member = _frame(target)
    # The binary-side native exporter must reproduce the Target frame exactly.
    if _frame(backend) != (program, member):
        raise ValueError("parent-view binary frame differs from native Target export")
    image = NativeImageArtifact(target=_TARGET, architecture="gfx1201", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target.encode()).hexdigest(), binary_format="hsaco",
        payload=payload, entry_points=(NativeEntryPoint(member["entry"], PARENT_VIEW_ABI),),
        compile_state=state, device_libraries=libraries)
    lineage = {key+"_digest": hashlib.sha256(text.encode()).hexdigest() for key, text in
               (("graph", recipe.graph_ir), ("schedule_ir", recipe.schedule_ir), ("tile_ir", recipe.tile_ir))}
    return ROCMNativePackage(recipe.tile_ir, target, backend, image,
                             _descriptor(image, program, member, lineage))


def validate_parent_view_runtime_artifact(artifact) -> ParentViewProgram:
    image, desc = artifact.native_image, artifact.launch_descriptor
    program, member = _frame(artifact.target_ir)
    lineage = {key+"_digest": hashlib.sha256(text.encode()).hexdigest() for key, text in
               (("graph", artifact.graph_ir), ("schedule_ir", artifact.schedule_ir), ("tile_ir", artifact.tile_ir))}
    if (image.target != _TARGET or image.architecture != "gfx1201" or image.binary_format != "hsaco"
            or image.pipeline_name != "tessera-rocm-executable"
            or image.target_ir_digest != hashlib.sha256(artifact.target_ir.encode()).hexdigest()
            or image.entry_points != (NativeEntryPoint(member["entry"], PARENT_VIEW_ABI),)
            or desc != _descriptor(image, program, member, lineage)):
        raise ValueError("parent-view artifact stage/ABI lineage differs")
    return _program(image, program, member, artifact.graph_ir)


def submit_parent_view(image, descriptor, buffers, scalars, stream):
    import numpy as np
    from tessera import runtime as rt
    from .native_scaled_program import PreparedScaledProgram
    if (stream not in (None, 0) or image.target != _TARGET or image.architecture != "gfx1201"
            or image.binary_format != "hsaco" or image.pipeline_name != "tessera-rocm-executable"
            or rt._rocm_live_arch() != "gfx1201"):
        raise ValueError("native parent-view requires owning gfx1201 and its owned completion stream")
    info = descriptor.provenance["native_parent_view"]
    program, member = json.loads(info["program_json"]), json.loads(info["member_json"])
    _validate_frame(program, member)
    if (set(buffers) != set(member["bindings"]) or set(scalars) != {"N"}
            or type(scalars["N"]) is not int or [scalars["N"]] != member["scalars"]):
        raise ValueError("parent-view invocation bindings/count differ")
    parent, output = (buffers[name] for name in member["bindings"])
    for array, row in zip((parent, output), program["buffers"], strict=True):
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float32)
                or list(array.shape) != row["shape"] or not array.flags.c_contiguous
                or array.nbytes != row["bytes"]):
            raise ValueError("parent-view invocation storage/capacity differs")
    if not output.flags.writeable or np.shares_memory(parent, output):
        raise ValueError("parent-view output must be writable and independent")
    library = rt._load_rocm_native_movement_runtime()
    if library is None:
        raise RuntimeError("native parent-view requires the checked HIP program runtime")
    package = _program(image, program, member, program["root_ir"])
    with PreparedScaledProgram(package, [parent], runtime_library=library._name) as owner:
        generation, _ = owner.invoke()
        np.copyto(output, owner.read(generation)[0], casting="no")
    return output
