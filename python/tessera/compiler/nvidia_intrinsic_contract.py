"""SM120 CUDA integer, cast, and packed-SIMD API coverage.

CUDA header availability is not a Tessera execution claim.  These rows retain
compile proof separately from Target-IR and runtime readiness so a CUDA API
intrinsic cannot become selectable merely because ``nvcc`` accepts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IntrinsicCategory = Literal[
    "integer_math", "integer_bits", "integer_dot", "cast", "packed_simd",
]
Readiness = Literal["ready", "planned", "not_applicable"]


@dataclass(frozen=True)
class SM120IntrinsicContract:
    key: str
    category: IntrinsicCategory
    representative_symbols: tuple[str, ...]
    semantics: str
    cuda_compile_state: Readiness
    target_ir_state: Readiness
    runtime_state: Readiness
    invalid_contract: str
    ptx_operand_storage: str

    @property
    def selectable(self) -> bool:
        return self.target_ir_state == self.runtime_state == "ready"


SM120_INTRINSIC_CONTRACTS: tuple[SM120IntrinsicContract, ...] = (
    SM120IntrinsicContract(
        "integer_min_max_abs", "integer_math", ("abs", "min", "max"),
        "signed/unsigned scalar integer arithmetic with C++ promotions",
        "ready", "planned", "planned",
        "signed minimum passed to abs is undefined",
        ".s32/.u32/.s64/.u64 typed registers",
    ),
    SM120IntrinsicContract(
        "integer_bit_ops", "integer_bits",
        ("__brev", "__byte_perm", "__clz", "__ffs", "__popc", "__funnelshift_l"),
        "bit-exact scalar operations; funnel shifts distinguish wrap from clamp",
        "ready", "planned", "planned",
        "__fns base outside 0..31 is undefined",
        ".b32/.b64 or same-width integer registers",
    ),
    SM120IntrinsicContract(
        "integer_packed_dot", "integer_dot", ("__dp2a_lo", "__dp4a"),
        "signedness-specific packed i16/i8 or four-way i8 dot plus i32 accumulate",
        "ready", "planned", "planned",
        "signedness-specific overloads are not interchangeable",
        ".b32 packed operands and .s32/.u32 accumulator",
    ),
    SM120IntrinsicContract(
        "numeric_cast_rn_rd_ru_rz", "cast",
        ("__float2int_rn", "__float2int_rd", "__float2int_ru", "__float2int_rz"),
        "numeric conversion with suffix-selected rounding mode",
        "ready", "planned", "planned",
        "floating input outside destination integer range is undefined",
        "explicitly typed source and destination registers",
    ),
    SM120IntrinsicContract(
        "bit_reinterpret_cast", "cast", ("__float_as_int", "__int_as_float"),
        "bit-preserving reinterpretation, not numeric conversion",
        "ready", "planned", "planned",
        "source and destination widths must match",
        "same-width bit-size register compatibility",
    ),
    SM120IntrinsicContract(
        "packed_simd_2x16_4x8", "packed_simd",
        ("__vadd2", "__vadd4", "__vaddss4", "__vabsdiffs4", "__vcmpeq4"),
        "independent packed halfword/byte lanes with operation-specific wrap or saturation",
        "ready", "planned", "planned",
        "lane width, signedness, saturation, and predicate-result form are semantic",
        ".b32 containing .s16x2/.u16x2/.s8x4/.u8x4 instruction formats",
    ),
)

_BY_KEY = {row.key: row for row in SM120_INTRINSIC_CONTRACTS}


def sm120_intrinsic_contract(key: str) -> SM120IntrinsicContract:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"no SM120 intrinsic contract for {key!r}") from exc


__all__ = [
    "IntrinsicCategory",
    "Readiness",
    "SM120IntrinsicContract",
    "SM120_INTRINSIC_CONTRACTS",
    "sm120_intrinsic_contract",
]
