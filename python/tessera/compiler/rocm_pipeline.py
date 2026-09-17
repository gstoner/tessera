"""Typed ROCm compilation-pipeline configuration.

Python selects an explicit semantic-family plugin and artifact boundary.  The
registered C++ ``tessera-rocm-executable`` pipeline owns pass composition and
ordering; Python never assembles a comma-separated sequence of pass names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


EXECUTABLE_PIPELINE_SCHEMA_VERSION = "tessera.executable_pipeline.v1"


class ROCMInputLevel(str, Enum):
    GRAPH = "graph"
    TILE = "tile"
    DIRECTIVE = "directive"


class ROCMOutputLevel(str, Enum):
    TARGET = "target"
    BINARY = "binary"



#: The gfx1201 promotions, by name. Every other RDNA4/CDNA arch has none.
_GFX1201_PROMOTED_FAMILIES = frozenset(
    {"softmax", "reduction", "matmul", "attention", "attention_backward"})


def promoted_families(arch: str) -> frozenset[str]:
    """The family plugins with exact-device proof on ``arch``.

    This is the one place the executable pipeline's fail-closed rule lives:
    gfx1151 has every family, gfx1201 has the five replay-verified ones, and
    gfx1200/gfx1250 have none pending exact-device evidence. It is a function
    rather than an inline condition so a *test* can ask the same question the
    launch path asks, and skip where launch would refuse. Before it existed,
    ~1500 tests in the `test_rocm_*_compiled.py` families ran on the gfx1201 box
    and failed with this module's own refusal — a host reporting "this proof
    does not exist here" as "this proof is broken".
    """
    if arch == "gfx1151":
        return frozenset(FAMILY_PLUGINS)
    if arch == "gfx1201":
        return _GFX1201_PROMOTED_FAMILIES
    return frozenset()

@dataclass(frozen=True)
class ROCMFamilyPlugin:
    family: str
    tile_producer: str
    target_ir_consumer: str
    backend_codegen: str


FAMILY_PLUGINS: dict[str, ROCMFamilyPlugin] = {
    family: ROCMFamilyPlugin(
        family=family,
        tile_producer="content_addressed_tile",
        target_ir_consumer="tessera_rocm",
        backend_codegen="rocdl_hsaco",
    )
    for family in (
        "attention",
        "attention_backward",
        "depth_attention",
        "algebra_clifford",
        "attention_mla_decode",
        "control_state_machine",
        "draft_dspark",
        "ebm_affine_langevin",
        "ebm_decode_init",
        "ebm_ebt_tiny",
        "ebm_energy_quadratic",
        "ebm_langevin",
        "ebm_partition",
        "fused_silu_mul",
        "indexing_gather",
        "indexing_scatter",
        "loss_binary",
        "loss_pointwise",
        "loss_policy",
        "matmul_batched_f32",
        "matmul_f32",
        "normalization",
        "optimizer",
        "ordering_sort",
        "position_alibi",
        "position_rope",
        "quant_dequant_gemm",
        "quant_fp",
        "quant_int4_pack",
        "reduction_arg",
        "rng_philox",
        "scan",
        "spectral_backward",
        "spectral_dft",
        "es_low_rank_correction",
        "scalar_activation",
        "scalar_binary",
        "scalar_bitwise",
        "scalar_compare",
        "scalar_logical",
        "scalar_predicate",
        "scalar_unary",
        "scalar_where",
        "sequence_deltanet",
        "sequence_linear_attention",
        "sequence_recurrent_cell",
        "sequence_selective_ssm",
        "sequence_selective_ssm_backward",
        "solver_cholesky",
        "solver_lu",
        "solver_qr",
        "solver_svd",
        "solver_triangular_solve",
        "solver_ift",
        "sparse_block_attention",
        "sparse_block_topk",
        "sparse_sddmm",
        "sparse_spmm",
        "matmul",
        "moe_dispatch",
        "paged_kv",
        "reduction",
        "softmax",
    )
}


@dataclass(frozen=True)
class ROCMExecutablePipeline:
    family: str
    input_level: ROCMInputLevel = ROCMInputLevel.TILE
    output_level: ROCMOutputLevel = ROCMOutputLevel.BINARY
    arch: str = "gfx1151"
    staging: str = "register"
    tile_q: int = 64
    tile_kv: int = 64
    depth_cooperative: bool = False

    def __post_init__(self) -> None:
        if type(self.depth_cooperative) is not bool or (self.depth_cooperative and self.family!='depth_attention'):
            raise ValueError('cooperative depth reduction requires the depth_attention family')
        if self.family not in FAMILY_PLUGINS:
            raise ValueError(f"unknown ROCm family plugin {self.family!r}")
        if (self.family == "control_state_machine"
                and self.output_level is not ROCMOutputLevel.BINARY):
            # The state-machine family is a host-level per-thread lowering
            # with no tessera_rocm.* Target-IR boundary: output=target would
            # relabel the untouched host program as Target IR (PR #606
            # review, P2). Mirrors the C++ contract-pass rejection.
            raise ValueError(
                "family 'control_state_machine' has no Target-IR boundary; "
                "only output=binary is supported")
        if self.family not in promoted_families(self.arch):
            raise ValueError(
                f"ROCm executable pipeline has no promoted family plugins for {self.arch}; "
                "gfx1200/gfx1250 remain fail-closed pending exact-device evidence"
            )
        if self.staging not in {"register", "lds"}:
            raise ValueError("ROCm matmul staging must be register or lds")
        if self.tile_q <= 0 or self.tile_kv <= 0:
            raise ValueError("ROCm attention tile sizes must be positive")

    @property
    def plugin(self) -> ROCMFamilyPlugin:
        return FAMILY_PLUGINS[self.family]

    def pass_pipeline(self, *, output: ROCMOutputLevel | None = None) -> str:
        terminal = output or self.output_level
        options = (
            f"family={self.family} input={self.input_level.value} "
            f"output={terminal.value} arch={self.arch} staging={self.staging} "
            f"tile-q={self.tile_q} tile-kv={self.tile_kv}"
        )
        if self.depth_cooperative:options += " depth-cooperative=true"
        return f"builtin.module(tessera-rocm-executable{{{options}}})"

    def cache_key(self) -> tuple[str, ...]:
        return (
            self.family,
            self.input_level.value,
            self.output_level.value,
            self.arch,
            self.staging,
            str(self.tile_q),
            str(self.tile_kv),
            str(self.depth_cooperative),
        )
