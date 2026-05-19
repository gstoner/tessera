"""Shared compiler legality checks for Python lowering and diagnostics.

The rule names and diagnostic codes are intentionally stable. C++ MLIR passes
should mirror these rule names as the native verifier grows coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .capabilities import CAPABILITY_REGISTRY_VERSION, canonical_op, normalize_target, supports_op


@dataclass(frozen=True)
class TensorContract:
    shape: tuple[object, ...] = ()
    dtype: Optional[str] = None
    layout: Optional[str] = None
    memory_space: Optional[str] = "global"

    @property
    def rank(self) -> Optional[int]:
        if not self.shape or "*" in self.shape or "?" in self.shape:
            return None
        return len(self.shape)


@dataclass(frozen=True)
class LegalityDiagnostic:
    code: str
    message: str
    rule: str
    severity: str = "error"

    def format(self) -> str:
        return f"{self.severity.upper()} {self.code} [{self.rule}]: {self.message}"


@dataclass(frozen=True)
class LegalityResult:
    op_name: str
    target: str
    diagnostics: tuple[LegalityDiagnostic, ...] = ()
    capability_version: str = CAPABILITY_REGISTRY_VERSION

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)


_ALLOWED_LAYOUTS = {"row_major", "col_major", "nhwc", "nchw", "blocked", "contiguous", None}
_ALLOWED_MEMORY_SPACES = {"global", "shared", "local", "constant", "host", None}
_SIDE_EFFECT_FREE = {
    "tessera.matmul",
    "tessera.gemm",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
    "tessera.rope",
    "tessera.flash_attn",
}
_COLLECTIVES = {
    "tessera.all_reduce",
    "tessera.all_gather",
    "tessera.reduce_scatter",
    "tessera.broadcast",
}


def check_op_legality(
    op_name: str,
    operands: Sequence[TensorContract] = (),
    *,
    target: object = "cpu",
    result: TensorContract | None = None,
    effects: Iterable[str] = (),
    attrs: Mapping[str, object] | None = None,
) -> LegalityResult:
    target_name = normalize_target(target)
    op = canonical_op(op_name)
    diagnostics: list[LegalityDiagnostic] = []
    attrs = attrs or {}

    _check_tensor_contracts(op, operands, result, diagnostics)
    _check_capability(op, target_name, operands, diagnostics)
    if op in {"tessera.matmul", "tessera.gemm"}:
        _check_matmul(op, operands, diagnostics)
    elif op in {"tessera.softmax", "tessera.softmax_safe"}:
        _check_min_rank(op, operands, 1, "LEGALITY_SOFTMAX_RANK", "softmax_rank", diagnostics)
    elif op == "tessera.gelu":
        _check_min_rank(op, operands, 1, "LEGALITY_GELU_RANK", "gelu_rank", diagnostics)
    elif op == "tessera.rope":
        _check_min_rank(op, operands, 2, "LEGALITY_ROPE_RANK", "rope_rank", diagnostics)
    elif op == "tessera.flash_attn":
        _check_flash_attention(operands, diagnostics)
    elif "kv_cache" in op:
        _check_kv_cache(op, operands, attrs, diagnostics)
    elif op in _COLLECTIVES:
        _check_collective(op, operands, effects, diagnostics)

    if op in _SIDE_EFFECT_FREE and any(effect not in {"read", "pure"} for effect in effects):
        diagnostics.append(LegalityDiagnostic(
            "LEGALITY_EFFECT_UNSUPPORTED",
            f"{op} is modeled as side-effect-free but effects {tuple(effects)!r} were requested",
            "effect_contract",
        ))

    return LegalityResult(op_name=op, target=target_name, diagnostics=tuple(diagnostics))


def _check_tensor_contracts(
    op: str,
    operands: Sequence[TensorContract],
    result: TensorContract | None,
    diagnostics: list[LegalityDiagnostic],
) -> None:
    for index, tensor in enumerate((*operands, *(() if result is None else (result,)))):
        role = "result" if result is not None and index == len(operands) else f"operand {index}"
        if tensor.layout not in _ALLOWED_LAYOUTS:
            diagnostics.append(LegalityDiagnostic(
                "LEGALITY_LAYOUT_UNSUPPORTED",
                f"{op} {role} uses unsupported layout {tensor.layout!r}",
                "layout_contract",
            ))
        if tensor.memory_space not in _ALLOWED_MEMORY_SPACES:
            diagnostics.append(LegalityDiagnostic(
                "LEGALITY_MEMORY_SPACE_UNSUPPORTED",
                f"{op} {role} uses unsupported memory space {tensor.memory_space!r}",
                "memory_space_contract",
            ))


def _check_capability(
    op: str,
    target: str,
    operands: Sequence[TensorContract],
    diagnostics: list[LegalityDiagnostic],
) -> None:
    dtype = next((operand.dtype for operand in operands if operand.dtype), None)
    rank = next((operand.rank for operand in operands if operand.rank is not None), None)
    cap = supports_op(target, op, dtype=dtype, rank=rank)
    if not cap.supported:
        diagnostics.append(LegalityDiagnostic(
            "LEGALITY_TARGET_CAPABILITY",
            cap.reason or f"{op} is not supported on {target}",
            "target_capability",
        ))


def _check_matmul(
    op: str,
    operands: Sequence[TensorContract],
    diagnostics: list[LegalityDiagnostic],
) -> None:
    if len(operands) < 2:
        diagnostics.append(LegalityDiagnostic("LEGALITY_OPERAND_ARITY", f"{op} requires two operands", "operand_arity"))
        return
    lhs, rhs = operands[0], operands[1]
    if lhs.rank is not None and lhs.rank != 2:
        diagnostics.append(LegalityDiagnostic("LEGALITY_MATMUL_RANK", "matmul lhs must be rank-2", "matmul_rank"))
    if rhs.rank is not None and rhs.rank != 2:
        diagnostics.append(LegalityDiagnostic("LEGALITY_MATMUL_RANK", "matmul rhs must be rank-2", "matmul_rank"))
    if lhs.rank == 2 and rhs.rank == 2:
        lhs_k, rhs_k = lhs.shape[1], rhs.shape[0]
        if _static_dim(lhs_k) and _static_dim(rhs_k) and str(lhs_k) != str(rhs_k):
            diagnostics.append(LegalityDiagnostic(
                "LEGALITY_MATMUL_K_MISMATCH",
                f"matmul K dimension mismatch: lhs has {lhs_k}, rhs has {rhs_k}",
                "matmul_shape",
            ))


def _check_min_rank(
    op: str,
    operands: Sequence[TensorContract],
    rank: int,
    code: str,
    rule: str,
    diagnostics: list[LegalityDiagnostic],
) -> None:
    if operands and operands[0].rank is not None and operands[0].rank < rank:
        diagnostics.append(LegalityDiagnostic(code, f"{op} requires rank >= {rank}", rule))


def _check_flash_attention(operands: Sequence[TensorContract], diagnostics: list[LegalityDiagnostic]) -> None:
    if len(operands) < 3:
        diagnostics.append(LegalityDiagnostic("LEGALITY_OPERAND_ARITY", "flash_attn requires q, k, and v", "operand_arity"))
        return
    for name, tensor in zip(("q", "k", "v"), operands[:3]):
        if tensor.rank is not None and tensor.rank not in {3, 4}:
            diagnostics.append(LegalityDiagnostic(
                "LEGALITY_FLASH_ATTN_RANK",
                f"flash_attn {name} must be rank-3 or rank-4",
                "flash_attn_rank",
            ))


def _check_kv_cache(
    op: str,
    operands: Sequence[TensorContract],
    attrs: Mapping[str, object],
    diagnostics: list[LegalityDiagnostic],
) -> None:
    if not operands and op != "tessera.kv_cache_create":
        diagnostics.append(LegalityDiagnostic("LEGALITY_OPERAND_ARITY", f"{op} requires a cache operand", "operand_arity"))
    if "page_size" in attrs and int(attrs["page_size"]) <= 0:
        diagnostics.append(LegalityDiagnostic("LEGALITY_KV_CACHE_PAGE", "kv_cache page_size must be positive", "kv_cache_page"))


def _check_collective(
    op: str,
    operands: Sequence[TensorContract],
    effects: Iterable[str],
    diagnostics: list[LegalityDiagnostic],
) -> None:
    if not operands:
        diagnostics.append(LegalityDiagnostic("LEGALITY_OPERAND_ARITY", f"{op} requires at least one tensor", "operand_arity"))
    if "communication" not in set(effects):
        diagnostics.append(LegalityDiagnostic(
            "LEGALITY_COLLECTIVE_EFFECT",
            f"{op} must carry the communication effect",
            "collective_effect",
        ))


def _static_dim(dim: object) -> bool:
    return str(dim) not in {"*", "?", "None", ""}


__all__ = [
    "LegalityDiagnostic",
    "LegalityResult",
    "TensorContract",
    "check_op_legality",
]
