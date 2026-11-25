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
_CARRIERS = {"tessera.matmul": "tile.matmul_kernel", "tessera.softmax": "tile.softmax_kernel",
             "tessera.gelu": "tile.activation_kernel", "tessera.relu": "tile.activation_kernel",
             "tessera.add": "tile.elementwise_kernel", "tessera.rmsnorm": "tile.norm_kernel",
             "tessera.layer_norm": "tile.norm_kernel"}
_STORAGE = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}
_NUMPY = {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}


def _np_dtype(storage: str):
    """The numpy dtype of a frame buffer; bf16 is ml_dtypes' bfloat16."""
    import numpy as np

    if storage == "bf16":
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(_NUMPY[storage])


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
        itemsize = 4 if spec["storage"] == "f32" else 2
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
    # The function name becomes an MLIR symbol (``@name``) at every level; a
    # Python name that is not one (``<lambda>``) would fail the first parse.
    for function in graph.functions:
        if not re.fullmatch(r"[A-Za-z_][\w$.]*", function.name):
            function.name = "_" + re.sub(r"[^\w]", "_", function.name)
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
        if array.dtype != _np_dtype(spec["storage"]) or array.shape != tuple(spec["shape"]):
            raise ValueError(f"Apple tensor input {name} differs from its frame buffer")
        values[name] = np.ascontiguousarray(array)
    launches = []
    for stage, record in zip(program.stages, frame["stages"], strict=True):
        package = stage.package
        spec = frame["buffers"][record["output"]]
        buffers = {binding: values[name] for binding, name in zip(stage.bindings, stage.inputs)}
        buffers[stage.output] = np.full(tuple(spec["shape"]), np.nan, _np_dtype(spec["storage"]))
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


#: Frame-owner stage kinds (``_frame_stage`` selects one per member from its
#: package ABI): 1 softmax, 2 simdgroup GEMM, 3 tanh GELU, 4 synthesized
#: elementwise (an activation, or the two-input add), 5 synthesized row
#: kernel (the f16/bf16 norm, one thread per row).
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
        APPLE_GELU_F16_ABI, APPLE_SIMDGROUP_GEMM_VARIANTS,
        APPLE_SIMDGROUP_GEMM_F16_ACCUMULATOR, APPLE_SOFTMAX_F16_ABI,
        APPLE_SYNTH_ACTIVATION_ABIS, APPLE_SYNTH_BINARY_ABIS, APPLE_SYNTH_NORM_ABIS,
        synthesized_activation_source, synthesized_binary_source, synthesized_norm_source)

    descriptor = stage.package.descriptor
    buffers = frame["buffers"]
    if stage.carrier == "tile.softmax_kernel":
        if descriptor.abi_id != APPLE_SOFTMAX_F16_ABI:
            raise AppleTensorProgramError(
                f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
        rows, columns = buffers[record["inputs"][0]]["shape"]
        return [1, record["inputs"][0], -1, record["output"], rows, 1, columns, 0, 0, 0], None, None
    if stage.carrier == "tile.norm_kernel":
        # The compiler-synthesized row norm (APPLE-NTBI-2 slice 9): the source
        # re-derived from the package's contract and digest-checked.
        provenance = descriptor.provenance
        storages = {abi: storage for storage, abi in APPLE_SYNTH_NORM_ABIS.items()}
        if descriptor.abi_id not in storages:
            raise AppleTensorProgramError(
                f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
        source, entry = synthesized_norm_source(
            provenance["kind"], storages[descriptor.abi_id], provenance["epsilon_bits"])
        if (hashlib.sha256(source.encode()).hexdigest() != provenance["source_sha256"]
                or entry != provenance["entry"]):
            raise AppleTensorProgramError(
                f"Apple member {stage.index} synthesized kernel differs from its package")
        rows, columns = buffers[record["inputs"][0]]["shape"]
        return [5, record["inputs"][0], -1, record["output"], rows, 1, columns, 0, 0, 0], source, entry
    if stage.carrier == "tile.elementwise_kernel":
        # The compiler-synthesized add (APPLE-NTBI-2 slice 7): two inputs,
        # the source re-derived from the package's contract and digest-checked.
        provenance = descriptor.provenance
        storages = {abi: storage for storage, abi in APPLE_SYNTH_BINARY_ABIS.items()}
        if descriptor.abi_id not in storages:
            raise AppleTensorProgramError(
                f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
        source, entry = synthesized_binary_source(
            provenance["kind"], storages[descriptor.abi_id], provenance["nan_mode"])
        if (hashlib.sha256(source.encode()).hexdigest() != provenance["source_sha256"]
                or entry != provenance["entry"]):
            raise AppleTensorProgramError(
                f"Apple member {stage.index} synthesized kernel differs from its package")
        a, b = record["inputs"]
        rows, columns = buffers[a]["shape"]
        return [4, a, b, record["output"], rows, 1, columns, 0, 0, 0], source, entry
    activation_storages = {abi: storage for storage, abi in APPLE_SYNTH_ACTIVATION_ABIS.items()}
    if stage.carrier == "tile.activation_kernel" and descriptor.abi_id in activation_storages:
        # The compiler-synthesized activation (APPLE-NTBI-2 slice 6): the frame
        # carries the same source the standalone package recorded, re-derived
        # from its contract and checked against the package's digest.
        provenance = descriptor.provenance
        source, entry = synthesized_activation_source(
            provenance["kind"], activation_storages[descriptor.abi_id], provenance["nan_mode"],
            provenance["zero_mode"])
        if (hashlib.sha256(source.encode()).hexdigest() != provenance["source_sha256"]
                or entry != provenance["entry"]):
            raise AppleTensorProgramError(
                f"Apple member {stage.index} synthesized kernel differs from its package")
        rows, columns = buffers[record["inputs"][0]]["shape"]
        return [4, record["inputs"][0], -1, record["output"], rows, 1, columns, 0, 0, 0], source, entry
    if stage.carrier == "tile.activation_kernel":
        # The tanh GELU activation's hand-written encode-only entry point; it is
        # elementwise, so the owner launches rows * columns threads.
        if (descriptor.abi_id != APPLE_GELU_F16_ABI
                or descriptor.provenance.get("approximation") != "tanh"):
            raise AppleTensorProgramError(
                f"Apple member {stage.index} has no frame encoder for {descriptor.abi_id}")
        rows, columns = buffers[record["inputs"][0]]["shape"]
        return [3, record["inputs"][0], -1, record["output"], rows, 1, columns, 0, 0, 0], None, None
    gemm = {abi: dtype for _, abi, dtype, _ in APPLE_SIMDGROUP_GEMM_VARIANTS.values()}
    if descriptor.abi_id not in gemm:
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
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), gemm[descriptor.abi_id], bm, bn, bk,
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
        dtype = _np_dtype(spec["storage"])
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


# --------------------------------------------------------------------------
# Slice 4: RuntimeArtifact export and compiler-checked restoration.
# --------------------------------------------------------------------------

ARTIFACT_SCHEMA = "tessera.apple.tensor_program.v1"
COMPILER_PATH = "canonical_apple_tensor_program"
_GRAPH_SCHEMA = "tessera.apple.tensor_program_graph.v1"
_SCALARS = (str, int, float, bool, type(None))


def _type_record(value: Any) -> dict:
    from .graph_ir import IRType

    if not isinstance(value, IRType):
        raise AppleTensorProgramError("Apple tensor Graph type is not an IRType")
    return {"mlir": value.mlir_str, "shape": list(value.shape),
            "dtype": value.dtype, "layout": value.layout}


def _type_from_record(record: Any):
    from .graph_ir import IRType

    if (not isinstance(record, dict) or set(record) != {"mlir", "shape", "dtype", "layout"}
            or not isinstance(record["mlir"], str) or not isinstance(record["shape"], list)
            or any(not isinstance(extent, str) for extent in record["shape"])
            or not isinstance(record["dtype"], (str, type(None)))
            or not isinstance(record["layout"], (str, type(None)))):
        raise AppleTensorProgramError("Apple tensor artifact Graph type record differs")
    return IRType(record["mlir"], tuple(record["shape"]), record["dtype"], record["layout"])


def _scalar_map(value: Any, what: str) -> dict:
    if value is None:
        return {}
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or any(not isinstance(item, _SCALARS) for item in value.values())):
        raise AppleTensorProgramError(f"Apple tensor Graph {what} are not scalar")
    return dict(value)


_SPAN = ("line", "col", "end_line", "end_col", "source_name")


def _span_record(span: Any):
    from .graph_ir import SourceSpan

    if span is None:
        return None
    if not isinstance(span, SourceSpan):
        raise AppleTensorProgramError("Apple tensor Graph source location is not a SourceSpan")
    return {name: getattr(span, name) for name in _SPAN}


def _span_from_record(record: Any):
    from .graph_ir import SourceSpan

    if record is None:
        return None
    if (not isinstance(record, dict) or set(record) != set(_SPAN)
            or any(type(record[name]) is not int for name in ("line", "col"))
            or any(not isinstance(record[name], (int, type(None))) or isinstance(record[name], bool)
                   for name in ("end_line", "end_col"))
            or not isinstance(record["source_name"], (str, type(None)))):
        raise AppleTensorProgramError("Apple tensor artifact source location differs")
    return SourceSpan(**record)


def _optional_text(value: Any, what: str):
    if not isinstance(value, (str, type(None))):
        raise AppleTensorProgramError(f"Apple tensor Graph {what} are not text")
    return value


def _graph_record(graph: GraphIRModule) -> dict:
    """The program's Graph as data: only what the profile's ops carry."""
    (function,) = graph.functions
    ops = []
    for op in function.body:
        if op.op_name not in _CARRIERS or op.result is None:
            raise AppleTensorProgramError(f"Apple tensor artifact cannot record {op.op_name}")
        ops.append({
            "result": op.result, "op_name": op.op_name, "operands": list(op.operands),
            "operand_types": list(op.operand_types), "result_type": op.result_type,
            "attrs": _optional_text(op.attrs, "op attributes"),
            "kwargs": _scalar_map(op.kwargs, "op keywords"),
            "inferred_type": _type_record(op.inferred_type),
            "source_span": _span_record(op.source_span),
        })
    return {
        "schema": _GRAPH_SCHEMA,
        "module_attrs": _scalar_map(graph.module_attrs, "module attributes"),
        "name": function.name,
        "args": [{"name": arg.name, "type": _type_record(arg.ir_type)} for arg in function.args],
        "result_types": [_type_record(value) for value in function.result_types],
        "fn_attrs": _scalar_map(function.fn_attrs, "function attributes"),
        "body": ops, "return_values": list(function.return_values),
    }


def _graph_from_record(record: Any) -> GraphIRModule:
    keys = {"schema", "module_attrs", "name", "args", "result_types", "fn_attrs",
            "body", "return_values"}
    op_keys = {"result", "op_name", "operands", "operand_types", "result_type", "attrs",
               "kwargs", "inferred_type", "source_span"}
    if (not isinstance(record, dict) or set(record) != keys or record["schema"] != _GRAPH_SCHEMA
            or not isinstance(record["name"], str) or not isinstance(record["args"], list)
            or not isinstance(record["body"], list) or not isinstance(record["result_types"], list)
            or not isinstance(record["return_values"], list)
            or any(not isinstance(value, str) for value in record["return_values"])):
        raise AppleTensorProgramError("Apple tensor artifact Graph record differs")
    args = []
    for arg in record["args"]:
        if not isinstance(arg, dict) or set(arg) != {"name", "type"} or not isinstance(arg["name"], str):
            raise AppleTensorProgramError("Apple tensor artifact Graph argument differs")
        args.append(IRArg(arg["name"], _type_from_record(arg["type"])))
    body = []
    for op in record["body"]:
        if (not isinstance(op, dict) or set(op) != op_keys or op["op_name"] not in _CARRIERS
                or not isinstance(op["result"], str) or not isinstance(op["result_type"], str)
                or any(not isinstance(value, str) for value in op["operands"] + op["operand_types"])):
            raise AppleTensorProgramError("Apple tensor artifact Graph op differs")
        inferred = _type_from_record(op["inferred_type"])
        body.append(IROp(
            result=op["result"], op_name=op["op_name"], operands=list(op["operands"]),
            operand_types=list(op["operand_types"]), result_type=op["result_type"],
            attrs=_optional_text(op["attrs"], "op attributes"),
            kwargs=_scalar_map(op["kwargs"], "op keywords"),
            source_span=_span_from_record(op["source_span"]),
            inferred_type=inferred, inferred_types=(inferred,)))
    function = GraphIRFunction(
        name=record["name"], args=args,
        result_types=[_type_from_record(value) for value in record["result_types"]],
        body=body, fn_attrs=_scalar_map(record["fn_attrs"], "function attributes"),
        return_values=list(record["return_values"]))
    return GraphIRModule(functions=[function],
                         module_attrs=_scalar_map(record["module_attrs"], "module attributes"))


def _member_record(stage: AppleTensorStage) -> dict:
    descriptor = stage.package.descriptor
    return {"member": stage.index, "op_name": stage.op_name, "carrier": stage.carrier,
            "abi_id": descriptor.abi_id, "route": descriptor.provenance.get("route"),
            "target_ir_sha256": hashlib.sha256(stage.package.target_ir.encode()).hexdigest(),
            "image_digest": stage.package.image.image_digest}


def runtime_artifact(program: AppleTensorProgram, *, argument_names=()):
    """Export the compiler product; restoration requires the matching compiler.

    The artifact carries the program's Graph as a narrow record, its original
    source/Schedule/Tile text and a digest of every member package. It carries
    no executable payload of its own: restoration recompiles the recorded Graph
    and refuses unless every level and every member package matches.
    """
    from tessera.runtime import RuntimeArtifact

    if not isinstance(program, AppleTensorProgram):
        raise TypeError("Apple tensor artifact requires a compiled Apple tensor program")
    names = tuple(argument_names)
    roots = [arg.name for arg in program.graph.functions[0].args]
    if names and (len(names) != len(roots) or len(set(names)) != len(names)
                  or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("Apple tensor argument names differ from source roots")
    graph = _graph_record(program.graph)
    # The record must rebuild the exact source the program was compiled from,
    # or the artifact could never be restored; refuse it at export instead.
    if _source(_graph_from_record(graph)) != program.source_ir:
        raise AppleTensorProgramError("Apple tensor Graph record does not rebuild its source")
    manifest = {
        "schema": ARTIFACT_SCHEMA, "graph": graph, "source_ir": program.source_ir,
        "schedule_ir": program.schedule_ir, "tile_ir": program.tile_ir,
        "members": [_member_record(stage) for stage in program.stages],
        "compiler_identity": program.compiler_identity,
    }
    return RuntimeArtifact(
        graph_ir=program.source_ir, schedule_ir=program.schedule_ir, tile_ir=program.tile_ir,
        target_ir="\n".join(stage.package.target_ir for stage in program.stages),
        metadata={"target": "apple_gpu", "compiler_path": COMPILER_PATH,
                  "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
                  "arg_names": list(names or roots), "native_program": manifest,
                  "restoration": "matching_compiler_required"})


def restore_runtime_artifact(artifact) -> AppleTensorProgram:
    """Recompile the recorded Graph with the configured compiler; refuse any drift."""
    metadata = getattr(artifact, "metadata", None) or {}
    data = metadata.get("native_program")
    keys = {"schema", "graph", "source_ir", "schedule_ir", "tile_ir", "members",
            "compiler_identity"}
    if (metadata.get("target") != "apple_gpu" or metadata.get("compiler_path") != COMPILER_PATH
            or not isinstance(data, dict) or set(data) != keys
            or data["schema"] != ARTIFACT_SCHEMA or not isinstance(data["members"], list)
            or any(getattr(artifact, field) != data[key] for field, key in (
                ("graph_ir", "source_ir"), ("schedule_ir", "schedule_ir"), ("tile_ir", "tile_ir")))):
        raise AppleTensorProgramError("Apple tensor runtime artifact differs from its compiler product")
    selected = find_tessera_opt()
    if selected is None:
        raise RuntimeError("Apple tensor artifact restoration requires the matching compiler")
    if _tool_digest(Path(selected).resolve()) != data["compiler_identity"]:
        raise AppleTensorProgramError("Apple tensor artifact was compiled by a different compiler")
    graph = _graph_from_record(data["graph"])
    if _source(graph) != data["source_ir"]:
        raise AppleTensorProgramError("Apple tensor artifact Graph does not rebuild its source")
    program = compile_apple_tensor_program(graph)
    if (program.source_ir, program.schedule_ir, program.tile_ir) != (
            data["source_ir"], data["schedule_ir"], data["tile_ir"]):
        raise AppleTensorProgramError("Apple tensor artifact differs from its compiler replay")
    if [_member_record(stage) for stage in program.stages] != data["members"]:
        raise AppleTensorProgramError("Apple tensor artifact member packages differ on replay")
    if artifact.target_ir != "\n".join(stage.package.target_ir for stage in program.stages):
        raise AppleTensorProgramError("Apple tensor artifact Target IR differs on replay")
    return program


def execute_runtime_artifact(artifact, args):
    """Restore a checked product and run it through the Metal frame owner."""
    from collections.abc import Mapping as _Mapping
    from tessera.runtime import NativeProgramExecution

    program = restore_runtime_artifact(artifact)
    roots = [arg.name for arg in program.graph.functions[0].args]
    names = (artifact.metadata or {}).get("arg_names")
    if (not isinstance(names, list) or len(names) != len(roots)
            or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names)):
        raise ValueError("Apple tensor artifact argument names differ from its roots")
    if isinstance(args, _Mapping):
        if set(args) != set(names):
            raise ValueError("Apple tensor named arguments differ")
        ordered = [args[name] for name in names]
    elif isinstance(args, (tuple, list)):
        if len(args) != len(roots):
            raise ValueError("Apple tensor positional arguments differ from its roots")
        ordered = list(args)
    else:
        raise ValueError("Apple tensor execution requires a root sequence or mapping")
    output, receipt = execute_frame(program, dict(zip(roots, ordered)))
    return NativeProgramExecution(output, ({
        "ok": True, "execution_kind": "native_gpu", "compiler_path": COMPILER_PATH, **receipt},))
