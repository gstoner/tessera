"""Typed x86 compilation-pipeline configuration.

The x86 native packager selects one semantic-family plugin and exact artifact
boundary.  The registered C++ ``tessera-x86-executable`` pipeline owns pass
composition; callers never assemble ``tessera-tile-to-x86`` option strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


EXECUTABLE_PIPELINE_SCHEMA_VERSION = "tessera.executable_pipeline.v1"


class X86InputLevel(str, Enum):
    TILE = "tile"


class X86OutputLevel(str, Enum):
    TARGET = "target"


@dataclass(frozen=True)
class X86FamilyPlugin:
    family: str
    tile_producer: str = "content_addressed_tile"
    target_ir_consumer: str = "tessera_x86"
    backend_codegen: str = "prebuilt_native_shared_object"


_FAMILIES = (
    "alibi",
    "argreduce",
    "attention",
    "attention_backward",
    "clifford",
    "deltanet",
    "ebm",
    "elementwise",
    "es_low_rank_correction",
    "fft",
    "kv_cache",
    "linalg",
    "loss",
    "matmul",
    "moe",
    "movement",
    "norm",
    "optimizer",
    "quantization",
    "reduction",
    "rng",
    "rope",
    "scan",
    "softmax",
    "solver_ift",
    "sort",
    "sparse",
    "spectral_backward",
    "ssm",
)

FAMILY_PLUGINS: dict[str, X86FamilyPlugin] = {
    family: X86FamilyPlugin(family=family) for family in _FAMILIES
}


@dataclass(frozen=True)
class X86ExecutablePipeline:
    family: str
    input_level: X86InputLevel = X86InputLevel.TILE
    output_level: X86OutputLevel = X86OutputLevel.TARGET
    architecture: str = "x86_64_avx512"

    def __post_init__(self) -> None:
        if self.family not in FAMILY_PLUGINS:
            raise ValueError(f"unknown x86 family plugin {self.family!r}")
        if self.architecture not in {"x86_64_avx512", "x86_64_base"}:
            raise ValueError(
                f"x86 executable pipeline has no family-plugin profile for "
                f"{self.architecture!r}"
            )
        if (
            self.architecture == "x86_64_base"
            and self.family not in {"softmax", "reduction"}
        ):
            raise ValueError(
                "x86_64_base exposes only the proven softmax/reduction plugins"
            )

    @property
    def plugin(self) -> X86FamilyPlugin:
        return FAMILY_PLUGINS[self.family]

    def pass_pipeline(self) -> str:
        options = (
            f"family={self.family} input={self.input_level.value} "
            f"output={self.output_level.value} arch={self.architecture}"
        )
        return f"builtin.module(tessera-x86-executable{{{options}}})"

    def cache_key(self) -> tuple[str, ...]:
        return (
            self.family,
            self.input_level.value,
            self.output_level.value,
            self.architecture,
        )


__all__ = [
    "EXECUTABLE_PIPELINE_SCHEMA_VERSION",
    "FAMILY_PLUGINS",
    "X86ExecutablePipeline",
    "X86FamilyPlugin",
    "X86InputLevel",
    "X86OutputLevel",
]
