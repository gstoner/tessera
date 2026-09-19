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
_GFX1201_PROMOTED_FAMILIES = frozenset({
    "softmax", "reduction", "matmul", "attention", "attention_backward",
    # Two scalar per-thread families with no WMMA fragment in them, promoted
    # 2026-09-17 on exact-device evidence from Tajasarus (RX 9070 XT):
    # `test_rocm_state_machine_exec.py` (forward + generated backward of the
    # irreducible and data-dependent machines) and
    # `test_rocm_ebm_geo_langevin_compiled.py` (the affine Langevin core under
    # the bivector and sphere samplers, spied to fire on every chain step).
    # The C++ pass (`Passes.cpp`, tessera-rocm-executable) carries the same
    # two names; keep both lists identical.
    "control_state_machine", "ebm_affine_langevin",
    # GFX1201-PARITY slice 2 (2026-09-17): the scalar and row-program
    # families with no WMMA fragment in them, promoted on the full-sweep
    # measurement on Tajasarus (972 tests skip -> pass, no kernel failure).
    # The optimizer *VJP* certificate lanes still stamp `rocm_gfx1151`; their
    # tests are pinned to that host until slice 2b widens the target name.
    "scalar_unary", "scalar_binary", "scalar_compare", "scalar_logical",
    "scalar_bitwise", "scalar_predicate", "scalar_where", "scalar_activation",
    "loss_binary", "loss_pointwise", "loss_policy", "normalization",
    "rng_philox", "indexing_gather", "indexing_scatter", "position_alibi",
    "position_rope", "quant_dequant_gemm", "quant_fp", "quant_int4_pack",
    "reduction_arg", "scan", "optimizer", "fused_silu_mul",
    # Slices 3-5 of the same program (engineering loops, 2026-09-17): the
    # attention tail, the spectral/solver/EBM/f32-matmul families and the
    # RDNA4-only FP8 matmul contract, each measured on Tajasarus (the per-test
    # diff, then the family files at the loop's head).
    "algebra_clifford",
    "attention_mla_decode",
    "depth_attention",
    "draft_dspark",
    "ebm_decode_init",
    "ebm_ebt_tiny",
    "ebm_energy_quadratic",
    "ebm_langevin",
    "ebm_partition",
    "es_low_rank_correction",
    "matmul_batched_f32",
    "matmul_f32",
    "moe_dispatch",
    "ordering_sort",
    # paged_kv (2026-09-18): the last family. Its generator is a scalar
    # per-thread gather with no WMMA fragment; it had no gfx1201 evidence
    # only because its device tests gated on a probe of the gfx11 WMMA
    # flash-attention kernel, which the runtime built without an `arch`
    # stamp. Measured on Tajasarus once the directive carried the chip.
    "paged_kv",
    "sequence_deltanet",
    "sequence_linear_attention",
    "sequence_recurrent_cell",
    "sequence_selective_ssm",
    "sequence_selective_ssm_backward",
    "solver_cholesky",
    "solver_ift",
    "solver_lu",
    "solver_qr",
    "solver_svd",
    "solver_triangular_solve",
    "sparse_block_attention",
    "sparse_block_topk",
    "sparse_sddmm",
    "sparse_spmm",
    "spectral_backward",
    "spectral_dft",
    # RDNA4-only: the checked 2:4 sparse matmul on SWMMAC (public admission,
    # 2026-09-18). gfx11 has no SWMMAC, so this family never joins gfx1151.
    "sparse_matmul_2to4",
})

#: Families whose ISA contract exists only on RDNA4; they are excluded from
#: the generic compiled lane's "every family" reading on gfx1151.
RDNA4_ONLY_FAMILIES = frozenset({"sparse_matmul_2to4"})


def generic_lane_families() -> frozenset[str]:
    """The families the generic compiled lane (a test that names no family)
    may assume on a fully promoted host."""
    return frozenset(FAMILY_PLUGINS) - RDNA4_ONLY_FAMILIES


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
        return frozenset(FAMILY_PLUGINS) - RDNA4_ONLY_FAMILIES
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
        "sparse_matmul_2to4",  # RDNA4-only (SWMMAC); see RDNA4_ONLY_FAMILIES
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
    #: Waves per workgroup (rows x cols of per-wave panels) for the LDS-staged
    #: typed matmul body; ignored under register staging.
    lds_waves: tuple[int, int] = (2, 2)
    #: Full 16-wide K slabs the typed matmul body issues per loop iteration
    #: (latency hiding; 1 is the established one-slab loop).
    k_unroll: int = 1
    #: rocdl.sched.group.barrier granularity for the WMMA panel.
    #: 0 keeps LLVM's default drained, single-buffered schedule --
    #: the schedule every recorded gfx1201 number was measured
    #: under. Stays 0 until a measurement says otherwise
    #: (ROCM-SCHED-GROUP-1).
    sched_groups: int = 0
    #: Dwords of LDS row padding. 0 is the unpadded historical
    #: layout whose fragment read collides 4 ways on the banks
    #: (ROCM-LDS-BANKPAD-1).
    lds_pad_dwords: int = 0
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
            if self.family in RDNA4_ONLY_FAMILIES:
                raise ValueError(
                    f"ROCm executable pipeline has no promoted family plugins for {self.arch}; "
                    f"{self.family!r} is an RDNA4 (SWMMAC) contract with no gfx11 form"
                )
            raise ValueError(
                f"ROCm executable pipeline has no promoted family plugins for {self.arch}; "
                "gfx1200/gfx1250 remain fail-closed pending exact-device evidence"
            )
        if self.staging not in {"register", "lds"}:
            raise ValueError("ROCm matmul staging must be register or lds")
        if (len(self.lds_waves) != 2 or any(type(w) is not int or w <= 0 or w > 8 for w in self.lds_waves)):
            raise ValueError("ROCm LDS staging takes a positive (waves_m, waves_n) pair of at most 8 each")
        if type(self.k_unroll) is not int or not 1 <= self.k_unroll <= 8:
            raise ValueError("ROCm matmul k_unroll must be an integer in [1, 8]")
        if type(self.sched_groups) is not int or not 0 <= self.sched_groups <= 16:
            raise ValueError(
                "ROCm sched_groups must be an integer in [0, 16]; it is a "
                "scheduling-granularity knob, and past the point where the "
                "pattern asks for more outstanding loads than the hardware "
                "holds it spills rather than overlapping")
        if type(self.lds_pad_dwords) is not int or not 0 <= self.lds_pad_dwords <= 4:
            raise ValueError("ROCm lds_pad_dwords must be an integer in [0, 4]")
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
            f"lds-waves-m={self.lds_waves[0]} lds-waves-n={self.lds_waves[1]} "
            f"k-unroll={self.k_unroll} "
            f"sched-groups={self.sched_groups} "
            f"lds-pad-dwords={self.lds_pad_dwords} "
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
            f"{self.lds_waves[0]}x{self.lds_waves[1]}",
            str(self.k_unroll),
            str(self.tile_q),
            str(self.tile_kv),
            str(self.depth_cooperative),
        )
