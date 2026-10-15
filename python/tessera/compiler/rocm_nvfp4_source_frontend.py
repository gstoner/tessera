"""Portable frontend/replay binding for source-preserving native NVFP4 programs."""
from __future__ import annotations
from dataclasses import dataclass
import inspect

from .native_artifact import LaunchDescriptor, NativeImageArtifact
from .rocm_native import ROCMNativePackage
from .rocm_nvfp4_expand_native import NVFP4ExpansionPackage
from .rocm_nvfp4_expansion_program import (
    NativeNVFP4ExpansionProgram, package_native_nvfp4_expansion_program,
)
from .rocm_nvfp4_resident import _portable_stage, _program_digest

SCHEMA = "tessera.rocm.nvfp4_source_frontend_program.v1"


def supports_source_trace(module):
    return (module is not None and len(module.functions) == 1
            and [op.op_name for op in module.functions[0].body]
            == ["tessera.nvfp4_expand", "tessera.scaled_matmul"])


def _package_record(package):
    package.validate()
    return {
        "program_json": package.program_json, "graph_ir": package.graph_ir,
        "consumer_checked_target_ir": package.consumer_checked_target_ir,
        "binding_digest": package.binding_digest,
        "stages": [
            _portable_stage(package.producer.graph_ir, package.producer.schedule_ir, package.producer.native),
            _portable_stage(package.consumer_graph_ir, package.consumer_schedule_ir, package.consumer),
        ],
    }


def _restore_package(record):
    if not isinstance(record, dict) or set(record) != {
            "program_json", "graph_ir", "consumer_checked_target_ir", "binding_digest", "stages"}:
        raise ValueError("source-preserving program record fields differ")
    if (any(not isinstance(record[key], str) for key in (
            "program_json", "graph_ir", "consumer_checked_target_ir", "binding_digest"))
            or not isinstance(record["stages"], list) or len(record["stages"]) != 2):
        raise ValueError("source-preserving program record types differ")
    stages = []
    for stage in record["stages"]:
        if (not isinstance(stage, dict) or set(stage) != {
                "graph_ir", "schedule_ir", "tile_ir", "target_ir", "image", "descriptor"}
                or any(not isinstance(stage[key], str) for key in (
                    "graph_ir", "schedule_ir", "tile_ir", "target_ir"))):
            raise ValueError("source-preserving stage fields differ")
        native = ROCMNativePackage(stage["tile_ir"], stage["target_ir"], "",
            NativeImageArtifact.from_dict(stage["image"]), LaunchDescriptor.from_dict(stage["descriptor"]))
        stages.append((stage["graph_ir"], stage["schedule_ir"], native))
    package = NativeNVFP4ExpansionProgram(
        record["program_json"], record["graph_ir"], NVFP4ExpansionPackage(*stages[0]),
        stages[1][2], stages[1][0], stages[1][1], record["consumer_checked_target_ir"],
        record["binding_digest"])
    package.validate()
    return package


@dataclass(frozen=True)
class TracedSourceNVFP4Program:
    native: NativeNVFP4ExpansionProgram
    argument_names: tuple[str, ...]
    graph_ir: str

    def validate(self):
        self.native.validate()
        if (self.graph_ir != self.native.graph_ir or len(self.argument_names) != 5
                or any(not isinstance(n, str) or not n.isidentifier() for n in self.argument_names)
                or len(set(self.argument_names)) != 5):
            raise ValueError("source-preserving frontend Graph/argument binding differs")
        for name in self.argument_names:
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def execute(self, *args, **kwargs):
        from tessera import runtime as rt
        self.validate()
        signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                       for name in self.argument_names])
        values = signature.bind(*args, **kwargs)
        ordered = [values.arguments[name] for name in self.argument_names]
        lib = rt._load_rocm_native_movement_runtime()
        if lib is None:
            raise RuntimeError("source-preserving program requires native prepared HIP provider")
        # Ordinary calls consume every current root. Import-once reuse requires
        # the explicit prepared API and must never assume host weights immutable.
        with self.native.prepare(ordered, runtime_library=lib._name) as owner:
            generation, _ = owner.invoke()
            output, = owner.read(generation)
        receipts = tuple({
            "ok": True, "execution_kind": "native_gpu",
            "image_digest": p.image.image_digest, "descriptor_digest": p.descriptor.descriptor_digest,
            "abi_id": p.descriptor.abi_id, "native_call_binding": "native_cpp_multi_output",
        } for p in (self.native.producer.native, self.native.consumer))
        return output, receipts

    def manifest(self):
        self.validate()
        record = {"schema": SCHEMA, "native_program": _package_record(self.native),
                  "argument_names": list(self.argument_names), "graph_ir": self.graph_ir}
        return {**record, "contract_digest": _program_digest(record)}


def source_program_from_manifest(data):
    if not isinstance(data, dict) or set(data) != {
            "schema", "native_program", "argument_names", "graph_ir", "contract_digest"}:
        raise ValueError("source-preserving frontend manifest fields differ")
    record = {key: value for key, value in data.items() if key != "contract_digest"}
    if data["schema"] != SCHEMA or data["contract_digest"] != _program_digest(record):
        raise ValueError("source-preserving frontend manifest identity differs")
    if not isinstance(data["argument_names"], list) or not isinstance(data["graph_ir"], str):
        raise ValueError("source-preserving frontend manifest types differ")
    program = TracedSourceNVFP4Program(_restore_package(data["native_program"]),
                                      tuple(data["argument_names"]), data["graph_ir"])
    program.validate()
    return program


def package_traced_source(module):
    if not supports_source_trace(module):
        raise ValueError("source-preserving frontend requires expansion then scaled matmul")
    package = package_native_nvfp4_expansion_program(module)
    program = TracedSourceNVFP4Program(package, tuple(a.name for a in module.functions[0].args),
                                      package.graph_ir)
    program.validate()
    return program
