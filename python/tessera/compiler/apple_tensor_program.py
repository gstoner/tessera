"""APPLE-NTBI-2 slice 2: an apple7 native tensor program with packaged members.

Original Graph -> (requested) native tensor program Schedule/Tile -> one Apple
scheduled package per member.

Each frame stage is bound to the traced Graph op it came from. That op is
lowered on its own through the production Graph -> Schedule -> Tile boundaries
and packaged by the existing Apple scheduled packager
(``package_scheduled_matmul`` / ``package_scheduled_kernel``). The package's
launch-level Tile carrier must equal the carrier the program projects for that
member (``--tessera-schedule-to-tile=tensor-member=<i>``), schedule digest
included. A member that drifts from its standalone package is refused, never
re-derived in Python.

What this slice does not provide: a Metal frame owner. ``run_members_sequentially``
launches each member package on its own and round-trips the private
intermediates through host memory. It is a numerical proof of the member
packages, labelled ``frame_owned = False``; it is not one command buffer and is
not the program's execution route (slices 3 and 4).
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp
from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

REQUEST = "tessera.native.apple_tensor_program_request"
PIPELINE = "tessera-lower-to-apple_gpu-runtime"
FRAME_SCHEMA = "tessera.native.tensor_program_frame.v1"
_FRAME = r'tessera.native.tensor_program_frame_json = "([^"]+)"'
_SOURCE = r'tessera.native.tensor_program_source_sha256 = "([0-9a-f]{64})"'
#: Graph op -> the launch-level Tile carrier its Apple scheduled package consumes.
_CARRIERS = {"tessera.matmul": "tile.matmul_kernel", "tessera.softmax": "tile.softmax_kernel"}
_STORAGE = {"fp16": "f16", "fp32": "f32"}
_NUMPY = {"f16": "float16", "f32": "float32"}


class AppleTensorProgramError(ValueError):
    """A member, its package, or the frame disagrees with the compiler product."""


def _frame(tile_ir: str) -> dict:
    records = re.findall(_FRAME, tile_ir)
    if len(records) != 1:
        raise AppleTensorProgramError("Apple tensor program requires exactly one controller frame")
    frame = json.loads(base64.b64decode(records[0], validate=True))
    keys = {"schema", "target", "architecture", "function", "buffers", "roots", "stages", "output"}
    if (not isinstance(frame, dict) or set(frame) != keys or frame["schema"] != FRAME_SCHEMA
            or frame["target"] != "apple_gpu" or frame["architecture"] != "apple7"):
        raise AppleTensorProgramError("Apple tensor frame schema/target differs")
    buffers = frame["buffers"]
    for index, spec in enumerate(buffers):
        if (not isinstance(spec, dict) or set(spec) != {"id", "role", "shape", "storage", "bytes"}
                or spec["id"] != index or spec["role"] not in {"input", "private", "output"}
                or spec["storage"] not in _NUMPY):
            raise AppleTensorProgramError("Apple tensor frame buffer identity/role differs")
        itemsize = 2 if spec["storage"] == "f16" else 4
        elements = 1
        for extent in spec["shape"]:
            if type(extent) is not int or extent <= 0:
                raise AppleTensorProgramError("Apple tensor frame buffer shape differs")
            elements *= extent
        if spec["bytes"] != elements * itemsize:
            raise AppleTensorProgramError("Apple tensor frame buffer bytes differ")
    if frame["roots"] != [i for i, s in enumerate(buffers) if s["role"] == "input"]:
        raise AppleTensorProgramError("Apple tensor frame source argument roles differ")
    if [i for i, s in enumerate(buffers) if s["role"] == "output"] != [frame["output"]]:
        raise AppleTensorProgramError("Apple tensor frame result role differs")
    for stage in frame["stages"]:
        if (not isinstance(stage, dict)
                or set(stage) != {"member", "entry", "inputs", "output", "carrier"}
                or stage["carrier"] not in _CARRIERS.values()):
            raise AppleTensorProgramError("Apple tensor frame stage record differs")
    return frame


def _carrier(tile_ir: str, carrier: str) -> str:
    """The single carrier op, SSA names erased: attributes and operand types."""
    lines = re.findall(rf"(?m)^\s*{re.escape(carrier)} [^\n]+$", tile_ir)
    if len(lines) != 1:
        raise AppleTensorProgramError(f"Apple member must carry exactly one {carrier}")
    return re.sub(r"%[\w.]+", "%", lines[0].strip())


def _signature(tile_ir: str, name: str) -> tuple[str, ...]:
    lines = re.findall(rf"(?m)^\s*func\.func @{re.escape(name)}\(([^\n]*)$", tile_ir)
    if len(lines) != 1:
        raise AppleTensorProgramError(f"Apple member lacks one @{name} function")
    return tuple(re.findall(r"tensor<[^>]+>", lines[0]))


def _source(module: GraphIRModule) -> str:
    if not isinstance(module, GraphIRModule) or len(module.functions) != 1:
        raise TypeError("Apple tensor program requires one traced Graph function")
    attrs = dict(module.module_attrs)
    for name, value in (("tessera.target", '"apple_gpu"'), ("tessera.arch", '"apple7"')):
        if name in attrs and attrs[name] != value:
            raise ValueError("Apple tensor frontend target or architecture differs")
        attrs[name] = value
    attrs[REQUEST] = "unit"
    targeted = copy.copy(module)
    targeted.module_attrs = attrs
    return targeted.to_mlir(target="apple_gpu", canonical=True)


def _stage_module(function: GraphIRFunction, op: IROp,
                  types: Mapping[str, Any]) -> tuple[GraphIRModule, tuple[str, ...]]:
    """The traced op as its own one-op Graph program (what a standalone call lowers).

    One argument per operand position, as the projected member keeps them: an
    aliased operand (``matmul(p, p)``) gets a distinct argument name per
    position, and the returned bindings say which value feeds each.
    """
    operands = [value.removeprefix("%") for value in op.operands]
    if op.inferred_type is None:
        raise AppleTensorProgramError(f"Apple member {op.op_name} lacks its result type")
    bindings: list[str] = []
    for position, name in enumerate(operands):
        bindings.append(name if name not in bindings else f"{name}__operand{position}")
    staged = copy.deepcopy(op)
    staged.operands = ["%" + name for name in bindings]
    single = GraphIRFunction(
        name=function.name,
        args=[IRArg(binding, types[name]) for binding, name in zip(bindings, operands)],
        result_types=[op.inferred_type], body=[staged],
        return_values=["%" + str(op.result)])
    module = GraphIRModule(functions=[single], module_attrs={"tessera.ir.version": '"1.0"'})
    return module, tuple(bindings)


def _package(op_name: str, stage_module: GraphIRModule):
    from .apple_native import package_scheduled_kernel, package_scheduled_matmul
    from .scheduled_kernel import lower_scheduled_kernel
    from .scheduled_matmul import lower_scheduled_matmul

    if op_name == "tessera.matmul":
        product = lower_scheduled_matmul(stage_module, target="apple_gpu")
        return product, package_scheduled_matmul(product, pipeline_name=PIPELINE)
    kernel = lower_scheduled_kernel(stage_module, target="apple_gpu")
    return kernel, package_scheduled_kernel(kernel, pipeline_name=PIPELINE)


@dataclass(frozen=True)
class AppleTensorStage:
    """One frame stage, the Graph op it lowers, and that op's Apple package."""
    index: int
    op_name: str
    carrier: str
    inputs: tuple[str, ...]
    output: str
    #: The package's input binding name for each operand position.
    bindings: tuple[str, ...]
    member_ir: str
    artifact: Any
    package: Any


def _bind_stages(function: GraphIRFunction, frame: dict, tile_ir: str, compiler: Path):
    """Bind every frame stage to its traced op, package it, and diff the carriers."""
    body = function.body
    if len(body) != len(frame["stages"]):
        raise AppleTensorProgramError("Apple frame stages differ from the traced Graph ops")
    names = {frame["roots"][i]: arg.name for i, arg in enumerate(function.args)}
    if len(names) != len(function.args) or len(function.args) != len(frame["roots"]):
        raise AppleTensorProgramError("Apple frame roots differ from the Graph arguments")
    types = {arg.name: arg.ir_type for arg in function.args}
    stages = []
    for index, (stage, op) in enumerate(zip(frame["stages"], body, strict=True)):
        carrier = _CARRIERS.get(op.op_name)
        if carrier is None or carrier != stage["carrier"] or op.inferred_type is None:
            raise AppleTensorProgramError(
                f"Apple frame stage {index} carrier differs from its Graph op {op.op_name}")
        ids = {value: key for key, value in names.items()}
        operands = [value.removeprefix("%") for value in op.operands]
        if [ids.get(name) for name in operands] != stage["inputs"]:
            raise AppleTensorProgramError(f"Apple frame stage {index} binds different buffers")
        spec = frame["buffers"][stage["output"]]
        if (tuple(spec["shape"]) != tuple(int(d) for d in op.inferred_type.shape)
                or spec["storage"] != _STORAGE.get(str(op.inferred_type.dtype))):
            raise AppleTensorProgramError(f"Apple frame stage {index} result buffer differs")
        if stage["output"] in names:
            raise AppleTensorProgramError("Apple frame stage overwrites a bound buffer")
        names[stage["output"]] = str(op.result)
        types[str(op.result)] = op.inferred_type
        member = run_tessera_opt(compiler, tile_ir, f"--tessera-schedule-to-tile=tensor-member={index}")
        stage_module, bindings = _stage_module(function, op, types)
        artifact, package = _package(op.op_name, stage_module)
        if _carrier(member, carrier) != _carrier(artifact.tile_ir, carrier):
            raise AppleTensorProgramError(
                f"Apple member {index} Tile carrier differs from its standalone package")
        if _signature(member, stage["entry"]) != _signature(artifact.tile_ir, function.name):
            raise AppleTensorProgramError(f"Apple member {index} tensor signature differs")
        bound = [b.name for b in package.descriptor.buffers]
        if bound != list(bindings) + [str(op.result)]:
            raise AppleTensorProgramError(f"Apple member {index} package binds different values")
        stages.append(AppleTensorStage(index, op.op_name, carrier, tuple(operands),
                                       str(op.result), bindings, member, artifact, package))
    if names.get(frame["output"]) != function.return_values[0].removeprefix("%"):
        raise AppleTensorProgramError("Apple frame result differs from the Graph return value")
    return tuple(stages)


@dataclass(frozen=True)
class AppleTensorProgram:
    """Compiler-verified source, Schedule, Tile and per-member Apple packages."""
    graph: GraphIRModule
    source_ir: str
    schedule_ir: str
    tile_ir: str
    stages: tuple[AppleTensorStage, ...]
    compiler: Path
    compiler_identity: str

    @property
    def frame(self) -> dict:
        return _frame(self.tile_ir)

    def validate(self) -> None:
        """Replay the original Graph and every member; refuse any drift."""
        if _tool_digest(self.compiler) != self.compiler_identity:
            raise RuntimeError("Apple tensor compiler changed after packaging")
        if _source(self.graph) != self.source_ir:
            raise AppleTensorProgramError("Apple tensor source differs from its Graph")
        seals = re.findall(_SOURCE, self.tile_ir)
        original = run_tessera_opt(self.compiler, self.source_ir, "--verify-each")
        if seals != [hashlib.sha256(original.strip().encode()).hexdigest()]:
            raise AppleTensorProgramError("Apple tensor Tile binds a different original Graph")
        schedule = run_tessera_opt(self.compiler, original, "--tessera-graph-to-schedule")
        if schedule != self.schedule_ir:
            raise AppleTensorProgramError("Apple tensor Schedule differs from source replay")
        tile = run_tessera_opt(self.compiler, schedule, "--tessera-schedule-to-tile")
        if tile != self.tile_ir:
            raise AppleTensorProgramError("Apple tensor Tile differs from its verified Schedule")
        replayed = _bind_stages(self.graph.functions[0], self.frame, tile, self.compiler)
        for old, new in zip(self.stages, replayed, strict=True):
            if (old.member_ir != new.member_ir or old.artifact.tile_ir != new.artifact.tile_ir
                    or old.package.target_ir != new.package.target_ir
                    or old.package.image.image_digest != new.package.image.image_digest
                    or old.package.descriptor != new.package.descriptor):
                raise AppleTensorProgramError(f"Apple member {old.index} package differs on replay")


def compile_apple_tensor_program(module: GraphIRModule) -> AppleTensorProgram:
    """Compile a traced apple7 matrix-product DAG to packaged native members.

    The module is compiled under the explicit program request, so a DAG the
    owner does not support is refused by the compiler
    (``APPLE_TENSOR_PROGRAM_UNSUPPORTED``) rather than routed elsewhere.
    """
    selected = find_tessera_opt()
    if selected is None:
        raise RuntimeError("Apple tensor program packaging requires the matching compiler")
    compiler = Path(selected).resolve()
    identity = _tool_digest(compiler)
    graph = copy.deepcopy(module)
    source = _source(graph)
    original = run_tessera_opt(compiler, source, "--verify-each")
    schedule = run_tessera_opt(compiler, original, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    if "tessera.native.tensor_program = 1" not in tile:
        raise AppleTensorProgramError("the compiler did not form an Apple native tensor program")
    stages = _bind_stages(graph.functions[0], _frame(tile), tile, compiler)
    if _tool_digest(compiler) != identity:
        raise RuntimeError("Apple tensor compiler changed during packaging")
    return AppleTensorProgram(graph=graph, source_ir=source, schedule_ir=schedule,
                              tile_ir=tile, stages=stages, compiler=compiler,
                              compiler_identity=identity)


def run_members_sequentially(program: AppleTensorProgram, inputs: Mapping[str, Any]):
    """Launch each member package on its own; intermediates round-trip the host.

    A numerical proof of the member packages, not the program's execution
    route: no frame owner, one runtime launch per member. Returns
    ``(output, receipt)``; the receipt says ``frame_owned = False``.
    """
    import numpy as np
    from tessera.runtime import RuntimeArtifact, launch

    if not isinstance(program, AppleTensorProgram):
        raise TypeError("sequential member execution requires a compiled Apple tensor program")
    frame = program.frame
    function = program.graph.functions[0]
    roots = [arg.name for arg in function.args]
    if set(inputs) != set(roots):
        raise ValueError(f"Apple tensor program inputs must be exactly {roots}")
    values: dict[str, Any] = {}
    for index, name in zip(frame["roots"], roots, strict=True):
        spec = frame["buffers"][index]
        array = np.asarray(inputs[name])
        if array.dtype != np.dtype(_NUMPY[spec["storage"]]) or array.shape != tuple(spec["shape"]):
            raise ValueError(f"Apple tensor input {name} differs from its frame buffer")
        values[name] = np.ascontiguousarray(array)
    launches = []
    for stage, record in zip(program.stages, frame["stages"], strict=True):
        package = stage.package
        spec = frame["buffers"][record["output"]]
        buffers = {binding: values[name] for binding, name in zip(stage.bindings, stage.inputs)}
        buffers[stage.output] = np.full(tuple(spec["shape"]), np.nan, _NUMPY[spec["storage"]])
        artifact = RuntimeArtifact(
            metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
            target_ir=package.target_ir, native_image=package.image,
            launch_descriptor=package.descriptor)
        launched = launch(artifact, buffers)
        if not launched.get("ok") or launched.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"Apple member {stage.index} did not execute natively: {launched}")
        values[stage.output] = launched["output"]
        launches.append({"member": record["member"], "entry_symbol": package.descriptor.entry_symbol,
                         "route": package.descriptor.provenance.get("route")})
    receipt = {"schema": "tessera.apple.tensor_program_sequential.v1", "frame_owned": False,
               "member_launches": len(launches), "launches": launches}
    return values[function.return_values[0].removeprefix("%")], receipt


#: The member ABIs that have encode-only entry points in the frame owner.
_FRAME_KINDS = {"tile.softmax_kernel": 1, "tile.matmul_kernel": 2}
_FRAME_ERRORS = {
    -1: "the runtime refused the frame contract",
    -2: "no Metal device",
    -3: "a member pipeline did not compile",
    -4: "a frame buffer could not be allocated",
    -5: "the frame command buffer failed or timed out",
}


def _frame_stage(stage: AppleTensorStage, record: dict, frame: dict):
    """One stage as the owner's flat record, re-checked against its package."""
    from .apple_native import (
        APPLE_SIMDGROUP_GEMM_F16_ABI, APPLE_SIMDGROUP_GEMM_F16_ACCUMULATOR,
        APPLE_SOFTMAX_F16_ABI)

    descriptor = stage.package.descriptor
    buffers = frame["buffers"]
    if stage.carrier == "tile.softmax_kernel":
        if descriptor.abi_id != APPLE_SOFTMAX_F16_ABI:
            raise AppleTensorProgramError(
                f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
        rows, columns = buffers[record["inputs"][0]]["shape"]
        return [1, record["inputs"][0], -1, record["output"], rows, 1, columns, 0, 0, 0], None, None
    if descriptor.abi_id != APPLE_SIMDGROUP_GEMM_F16_ABI:
        raise AppleTensorProgramError(
            f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
    from .apple_target import AppleGPUArch, AppleGPUTargetProfile
    from .msl_gemm_emit import materialize_apple_simdgroup_tile_msl

    artifact = stage.artifact
    (m, k), (k2, n) = (buffers[i]["shape"] for i in record["inputs"])
    if (m, n, k) != (artifact.m, artifact.n, artifact.k) or k2 != k:
        raise AppleTensorProgramError(f"Apple member {stage.index} extents differ from its package")
    bm, bn, bk = (int(value) for value in descriptor.provenance["block"])
    # The same materialization the standalone launch of this package uses.
    msl = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), "fp16", bm, bn, bk,
        accumulator_dtype=APPLE_SIMDGROUP_GEMM_F16_ACCUMULATOR, double_buffer=True)
    threads = int(msl.resources.threadgroup[0])
    a, b = record["inputs"]
    return ([2, a, b, record["output"], m, n, k, msl.shape.m, msl.shape.n, threads],
            msl.msl, msl.entry)


def execute_frame(program: AppleTensorProgram, inputs: Mapping[str, Any]):
    """Run the whole program in one Metal command buffer (the frame owner).

    Every member is encoded by its encode-only entry point into one serial
    compute encoder; program-owned intermediates live in device-only
    (``MTLStorageModePrivate``) buffers. Returns ``(output, receipt)``.
    A failure raises; it never runs a reference path.
    """
    import ctypes

    import numpy as np
    from tessera._apple_gpu_dispatch import apple_gpu_runtime_handle, bind_registered

    if not isinstance(program, AppleTensorProgram):
        raise TypeError("the Apple frame owner requires a compiled Apple tensor program")
    frame = program.frame
    function = program.graph.functions[0]
    roots = [arg.name for arg in function.args]
    if set(inputs) != set(roots):
        raise ValueError(f"Apple tensor program inputs must be exactly {roots}")
    handle, dylib, reason = apple_gpu_runtime_handle()
    # A prebuilt image reports no build path; the loaded handle names its file.
    dylib = dylib or (getattr(handle, "_name", None) if handle is not None else None)
    if not dylib or not Path(dylib).is_file():
        raise RuntimeError(f"the Apple frame owner requires the Apple GPU runtime: {reason}")
    loaded = hashlib.sha256(Path(dylib).read_bytes()).hexdigest()
    for stage in program.stages:
        if hashlib.sha256(stage.package.image.payload).hexdigest() != loaded:
            raise AppleTensorProgramError(
                f"Apple member {stage.index} was packaged against a different runtime image")
    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    if owner is None:
        raise RuntimeError("the Apple GPU runtime lacks tessera_apple_gpu_tensor_frame_execute")

    host: list[Any] = []
    flat: list[int] = []
    keep: list[Any] = []
    output = None
    for spec in frame["buffers"]:
        flat += [{"input": 0, "private": 1, "output": 2}[spec["role"]], spec["bytes"]]
        dtype = np.dtype(_NUMPY[spec["storage"]])
        if spec["role"] == "input":
            name = roots[frame["roots"].index(spec["id"])]
            array = np.asarray(inputs[name])
            if array.dtype != dtype or array.shape != tuple(spec["shape"]):
                raise ValueError(f"Apple tensor input {name} differs from its frame buffer")
            array = np.ascontiguousarray(array)
            keep.append(array)
            host.append(array.ctypes.data_as(ctypes.c_void_p))
        elif spec["role"] == "output":
            output = np.empty(tuple(spec["shape"]), dtype)
            host.append(output.ctypes.data_as(ctypes.c_void_p))
        else:
            host.append(None)
    records, sources, entries = [], [], []
    for stage, record in zip(program.stages, frame["stages"], strict=True):
        row, source, entry = _frame_stage(stage, record, frame)
        records += row
        sources.append(source.encode() if source else None)
        entries.append(entry.encode() if entry else None)
    count, stage_count = len(frame["buffers"]), len(frame["stages"])
    receipt = (ctypes.c_int64 * 4)()
    status = owner(
        (ctypes.c_int64 * len(flat))(*flat), (ctypes.c_void_p * count)(*host), count,
        (ctypes.c_int64 * len(records))(*records), (ctypes.c_char_p * stage_count)(*sources),
        (ctypes.c_char_p * stage_count)(*entries), stage_count, receipt)
    if status != 1:
        raise RuntimeError("Apple tensor frame did not execute: "
                           + _FRAME_ERRORS.get(status, f"status {status}"))
    if receipt[0] != 1 or receipt[1] != stage_count:
        raise RuntimeError("Apple tensor frame receipt differs from the frame")
    return output, {
        "schema": "tessera.apple.tensor_frame_receipt.v1", "frame_owned": True,
        "command_buffers": int(receipt[0]), "dispatches": int(receipt[1]),
        "private_device_buffers": int(receipt[2]),
        "gpu_time_ns": int(receipt[3]) if receipt[3] >= 0 else None,
        "members": [record["member"] for record in frame["stages"]],
    }
