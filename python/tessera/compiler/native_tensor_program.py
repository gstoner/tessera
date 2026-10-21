"""Original Graph -> verified native Target program -> retained C++ execution.

This binding transports actual compiler projections. It contains no numerical
operation classification or Graph/Tile construction. Portable deserialization
without the compiler is a separate admission boundary, not provided here.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import re

from .graph_ir import GraphIRModule
from .nvidia_native import _compile_tile_ir
from .prepared_native_tensor_frame import PreparedNativeTensorFrame, validate_frame
from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

_QUERY = "--tessera-graph-to-schedule=tensor-program-query"
_REPLAY = "--tessera-graph-to-schedule=tensor-program-replay"
_FRAME = r'tessera.native.tensor_program_frame_json = "([^"]+)"'
_SOURCE = r'tessera.native.tensor_program_source_sha256 = "([0-9a-f]{64})"'


def _source(source):
    if isinstance(source, GraphIRModule):
        attrs = dict(source.module_attrs)
        for name, value in (("tessera.target", '"nvidia_sm120"'), ("tessera.arch", '"sm_120"')):
            if name in attrs and attrs[name] != value:
                raise ValueError("native tensor frontend target or architecture differs")
            attrs[name] = value
        # Record only the selected target. The actual semantic operations, types,
        # operands, results, policies and argument order belong to the frontend.
        source = replace(source, module_attrs=attrs).to_mlir(target="nvidia_sm120", canonical=True)
    if not isinstance(source, str) or not source.strip():
        raise TypeError("native tensor packaging requires original frontend Graph MLIR")
    return source


def _frame(target):
    import json

    records = re.findall(_FRAME, target)
    if len(records) != 1:
        raise ValueError("native Target program requires exactly one controller frame")
    value = json.loads(base64.b64decode(records[0], validate=True))
    validate_frame(value)
    return value


@dataclass(frozen=True)
class NativeTensorProgram:
    """Compiler-verified immutable source, complete Target and member images.

    Construction and explicit validate() require the matching compiler.
    prepare() and retained calls use the already verified values without tools.
    There is no dictionary/portable loader that can bypass source replay.
    """
    source_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    members: tuple[str, ...]
    images: tuple[bytes, ...]
    compiler: Path
    compiler_identity: str

    def __post_init__(self):
        self.validate()

    @property
    def frame(self):
        return copy.deepcopy(_frame(self.target_ir))

    def validate(self):
        if (not isinstance(self.compiler, Path)
                or _tool_digest(self.compiler) != self.compiler_identity):
            raise RuntimeError("native tensor compiler changed after packaging")
        if (any(not isinstance(stage, str) or not stage.strip()
                for stage in (self.source_ir, self.schedule_ir, self.tile_ir, self.target_ir))
                or type(self.members) is not tuple or type(self.images) is not tuple
                or any(not isinstance(value, str) for value in self.members)
                or any(not isinstance(value, bytes) for value in self.images)):
            raise ValueError("native tensor immutable source/member fields differ")
        seals = re.findall(_SOURCE, self.target_ir)
        if seals != [hashlib.sha256(self.source_ir.strip().encode()).hexdigest()]:
            raise ValueError("native tensor Target binds a different original Graph")
        schedule = run_tessera_opt(self.compiler, self.source_ir, "--tessera-graph-to-schedule")
        if schedule != self.schedule_ir:
            raise ValueError("native tensor Schedule differs from original source projection")
        tile = run_tessera_opt(self.compiler, schedule, "--tessera-schedule-to-tile")
        if tile != self.tile_ir:
            raise ValueError("native tensor Tile differs from its verified Schedule")
        replay = run_tessera_opt(self.compiler, self.target_ir, _REPLAY)
        if replay != self.target_ir:
            raise ValueError("native tensor Target differs after original Graph replay")
        frame = self.frame
        if len(self.members) != len(frame["stages"]) or len(self.images) != len(self.members):
            raise ValueError("native tensor member/image arity differs")
        for index, (stage, member, image) in enumerate(zip(
                frame["stages"], self.members, self.images, strict=True)):
            projected = run_tessera_opt(self.compiler, self.target_ir,
                                       f"--tessera-schedule-to-tile=tensor-member={index}")
            if projected != member:
                raise ValueError("native tensor member differs from verified Target projection")
            _, ptx, *_ = _compile_tile_ir(member, stage["entry"], tessera_opt=self.compiler)
            if image != ptx.encode():
                raise ValueError("native tensor image differs from its native member")
        if _tool_digest(self.compiler) != self.compiler_identity:
            raise RuntimeError("native tensor compiler changed during source replay")

    def prepare(self, *, provider=None):
        """Bind verified bytes to a retained native owner; no compiler invocation."""
        return PreparedNativeTensorFrame(self.frame, self.images, provider=provider)


def compile_native_tensor_program(source, *, tool=None):
    """Return the native program, or None when the compiler declines this profile.

    A declined profile retains its original source. Callers may select an
    independently implemented route; compilation failures after admission
    propagate and are never replaced by eager arithmetic.
    """
    selected = tool if tool is not None else find_tessera_opt()
    if selected is None:
        raise RuntimeError("native tensor packaging requires the matching compiler")
    compiler = Path(selected).resolve()
    identity = _tool_digest(compiler)
    original = run_tessera_opt(compiler, _source(source), "--verify-each")
    queried = run_tessera_opt(compiler, original, _QUERY)
    if "tessera.native.tensor_program_admission = true" not in queried:
        return None
    scheduled = run_tessera_opt(compiler, original, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, scheduled, "--tessera-schedule-to-tile")
    target = run_tessera_opt(compiler, tile, "--tessera-lower-to-nvidia-sm120")
    frame = _frame(target)
    members, images = [], []
    for index, stage in enumerate(frame["stages"]):
        member = run_tessera_opt(compiler, target,
                                f"--tessera-schedule-to-tile=tensor-member={index}")
        _, ptx, *_ = _compile_tile_ir(member, stage["entry"], tessera_opt=compiler)
        members.append(member)
        images.append(ptx.encode())
    return NativeTensorProgram(source_ir=original, schedule_ir=scheduled, tile_ir=tile,
                               target_ir=target, members=tuple(members), images=tuple(images),
                               compiler=compiler, compiler_identity=identity)


def runtime_artifact(program, *, argument_names=()):
    """Expose actual native compiler stages; restoration requires the compiler."""
    from tessera.runtime import RuntimeArtifact

    if not isinstance(program, NativeTensorProgram):
        raise TypeError("native tensor artifact requires a verified compiler product")
    names = tuple(argument_names)
    if names and (len(names) != len(program.frame["roots"])
                  or any(not isinstance(name, str) or not name for name in names)
                  or len(set(names)) != len(names)):
        raise ValueError("native tensor argument names differ from source roots")
    manifest = {
        "schema": "tessera.native.tensor_program.v1",
        "source_ir": program.source_ir, "schedule_ir": program.schedule_ir,
        "tile_ir": program.tile_ir, "target_ir": program.target_ir,
        "members": list(program.members),
        "images": [base64.b64encode(image).decode() for image in program.images],
        "compiler_identity": program.compiler_identity,
    }
    return RuntimeArtifact(
        graph_ir=program.source_ir, schedule_ir=program.schedule_ir,
        tile_ir=program.tile_ir, target_ir=program.target_ir,
        metadata={"target": "nvidia_sm120", "compiler_path": "canonical_native_tensor_program",
                  "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
                  "arg_names": list(names), "native_program": manifest,
                  "restoration": "matching_compiler_required"},
    )


def restore_runtime_artifact(artifact):
    """Recheck source ancestry with a trusted configured tool, never a payload path."""
    metadata = artifact.metadata or {}
    data = metadata.get("native_program")
    keys = {"schema", "source_ir", "schedule_ir", "tile_ir", "target_ir",
            "members", "images", "compiler_identity"}
    if (metadata.get("target") != "nvidia_sm120"
            or metadata.get("compiler_path") != "canonical_native_tensor_program"
            or not isinstance(data, dict) or set(data) != keys
            or data["schema"] != "tessera.native.tensor_program.v1"
            or not isinstance(data["members"], list) or not isinstance(data["images"], list)
            or any(not isinstance(value, str) for value in data["images"])
            or any(getattr(artifact, field) != data[key] for field, key in (
                ("graph_ir", "source_ir"), ("schedule_ir", "schedule_ir"),
                ("tile_ir", "tile_ir"), ("target_ir", "target_ir")))):
        raise ValueError("native tensor runtime artifact differs from its compiler product")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native tensor artifact restoration requires the matching compiler")
    return NativeTensorProgram(
        source_ir=data["source_ir"], schedule_ir=data["schedule_ir"],
        tile_ir=data["tile_ir"], target_ir=data["target_ir"],
        members=tuple(data["members"]),
        images=tuple(base64.b64decode(value, validate=True) for value in data["images"]),
        compiler=Path(tool).resolve(), compiler_identity=data["compiler_identity"],
    )


def execute_runtime_artifact(artifact, args):
    """Restore a checked product and run its complete native frame synchronously."""
    from collections.abc import Mapping
    from tessera.runtime import NativeProgramExecution

    program = restore_runtime_artifact(artifact)
    names = (artifact.metadata or {}).get("arg_names")
    if isinstance(args, Mapping):
        if (not isinstance(names, list) or len(names) != len(program.frame["roots"])
                or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or set(args) != set(names)):
            raise ValueError("native tensor named arguments differ")
        roots = tuple(args[name] for name in names)
    elif isinstance(args, (tuple, list)):
        roots = tuple(args)
    else:
        raise ValueError("native tensor execution requires a root sequence or mapping")
    with program.prepare() as owner:
        output = owner(roots)
    return NativeProgramExecution(output, ({
        "ok": True, "execution_kind": "native_gpu",
        "compiler_path": "canonical_native_tensor_program",
        "completed_native_stages": len(program.members),
    },))
