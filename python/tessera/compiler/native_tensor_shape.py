"""Compile a compiler-owned MLIR shape function; no Python shape evaluation.

The retained native frame consumes the resulting function through its checked
ABI. Compilation is a cold packaging operation, using the selected LLVM tools.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import tempfile

from .llvm_tools import find_llvm_tool
from .native_gpu_storage import _run
from .toolchain_identity import binary_content_digest


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class NativeShapeCompanion:
    image: bytes
    ir_sha256: str
    image_sha256: str
    entry: str
    root_words: int
    plan_words: int
    tools: tuple[tuple[str, str], ...]

    def to_record(self):
        data = asdict(self)
        data["image"] = base64.b64encode(self.image).decode("ascii")
        data["tools"] = [list(value) for value in self.tools]
        return data

    @classmethod
    def from_record(cls, data, program):
        keys = {"image", "ir_sha256", "image_sha256", "entry", "root_words", "plan_words", "tools"}
        if (not isinstance(data, dict) or set(data) != keys
                or not isinstance(data["image"], str) or not isinstance(data["tools"], list)
                or any(not isinstance(value, list) or len(value) != 2
                       or any(not isinstance(field, str) for field in value) for value in data["tools"])):
            raise ValueError("native shape companion record differs")
        companion = cls(**{**data, "image": base64.b64decode(data["image"], validate=True),
                            "tools": tuple(tuple(value) for value in data["tools"])})
        companion.validate(program)
        return companion

    def validate(self, program):
        _validate_program(program)
        if (not isinstance(self.image, bytes) or not self.image
                or _sha(self.image) != self.image_sha256
                or self.ir_sha256 != program["sha256"]
                or self.entry != program["entry"]
                or self.root_words != program["root_words"]
                or self.plan_words != program["plan_words"]
                or not isinstance(self.tools, tuple) or len(self.tools) != 3
                or any(not isinstance(value, tuple) or len(value) != 2
                       or not isinstance(value[0], str) or not isinstance(value[1], str)
                       or value[0] not in {"mlir-opt", "mlir-translate", "clang"}
                       or not re.fullmatch(r"[0-9a-f]{64}", value[1]) for value in self.tools)
                or len({value[0] for value in self.tools}) != 3):
            raise ValueError("native shape companion differs from its compiler product")


def _validate_program(program):
    if (not isinstance(program, dict)
            or set(program) != {"entry", "root_words", "plan_words", "ir", "sha256"}
            or program["entry"] != "__tessera_tensor_shape_plan"
            or type(program["root_words"]) is not int
            or not 2 <= program["root_words"] <= 256
            or type(program["plan_words"]) is not int
            or not 15 <= program["plan_words"] <= 960
            or not isinstance(program["ir"], str) or not program["ir"].strip()
            or _sha(program["ir"].encode()) != program["sha256"]):
        raise ValueError("native shape program transport differs")


@lru_cache(maxsize=64)
def _compile(ir, entry, root_words, plan_words, tools):
    paths = {name: Path(path) for name, path, _ in tools}
    lowered = _run(paths["mlir-opt"], "--convert-to-llvm",
                   "--reconcile-unrealized-casts", source=ir)
    native = _run(paths["mlir-translate"], "--mlir-to-llvmir", source=lowered)
    with tempfile.TemporaryDirectory(prefix="tessera-native-shape-") as tmp:
        library = Path(tmp) / "shape.so"
        _run(paths["clang"], "-x", "ir", "-", "-shared", "-fPIC", "-O2",
             "-o", str(library), source=native)
        image = library.read_bytes()
    return NativeShapeCompanion(image, _sha(ir.encode()), _sha(image), entry,
                                root_words, plan_words,
                                tuple((name, digest) for name, _, digest in tools))


def build_shape_companion(program, *, llvm_bin=None):
    """Compile typed shape MLIR with a content-pinned matching LLVM toolchain."""
    _validate_program(program)
    tools = []
    for name in ("mlir-opt", "mlir-translate", "clang"):
        tool = Path(llvm_bin) / name if llvm_bin is not None else find_llvm_tool(name)
        if tool is None or not tool.is_file():
            raise RuntimeError(f"native shape compilation requires the matching {name}")
        tools.append((name, str(tool.resolve()), binary_content_digest(tool)))
    companion = _compile(program["ir"], program["entry"], program["root_words"],
                         program["plan_words"], tuple(tools))
    companion.validate(program)
    return companion
