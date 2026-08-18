"""Digest-bound physical validation products for structured reverse mode.

The general W4 lowering lives in the C++ paired-autodiff pipeline.  This module
provides the deliberately small architecture gate used to prove that its
residual boundary can be consumed without re-entering Graph IR: one dynamic
branch product and one bounded HYBRID while product.  The arithmetic children
are the production AVX-512 or gfx1151 elementwise packages; control and tape
ownership remain in this content-addressed parent artifact.

This is not a second control-flow compiler.  It accepts only products already
identified by paired IR/CFG/residual digests and rejects every other target or
product shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Sequence

from .residual_evaluator import (
    RegionResidualABI,
    TreeverseCandidate,
    TreeverseExecution,
    build_region_residual_abi,
    capture_treeverse_forward,
    execute_treeverse_region_adjoint,
)


W4_REGION_PRODUCT_SCHEMA = "tessera.w4_region_product.v1"
RegionProductKind = Literal["dynamic_if_saved_activation", "hybrid_while_scale"]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class W4RegionProduct:
    """One target-owned consumer of a paired structured-region artifact."""

    target: Literal["x86", "rocm_gfx1151"]
    kind: RegionProductKind
    paired_ir_digest: str
    cfg_digest: str
    state_shape_bound: tuple[int, ...]
    residual_abi: RegionResidualABI | None = None
    schema: str = W4_REGION_PRODUCT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != W4_REGION_PRODUCT_SCHEMA:
            raise ValueError("unsupported W4 region-product schema")
        if self.target not in {"x86", "rocm_gfx1151"}:
            raise ValueError("W4 physical products support x86 and gfx1151 only")
        _require_digest("paired IR digest", self.paired_ir_digest)
        _require_digest("CFG digest", self.cfg_digest)
        if not self.state_shape_bound or any(dim <= 0 for dim in self.state_shape_bound):
            raise ValueError("W4 region products require positive state-shape bounds")
        if self.kind == "hybrid_while_scale":
            if self.residual_abi is None or self.residual_abi.policy != "hybrid":
                raise ValueError("HYBRID while products require a HYBRID residual ABI")
            if self.residual_abi.cfg_digest != self.cfg_digest:
                raise ValueError("residual and structured-CFG identities disagree")
        elif self.residual_abi is not None:
            raise ValueError("dynamic if products carry branch residuals, not loop tapes")

    def contract(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target": self.target,
            "kind": self.kind,
            "paired_ir_digest": self.paired_ir_digest,
            "cfg_digest": self.cfg_digest,
            "state_shape_bound": list(self.state_shape_bound),
            "residual_abi": self.residual_abi.as_dict() if self.residual_abi else None,
            "residual_digest": self.residual_digest,
            "consumer": (
                "avx512_native_children"
                if self.target == "x86"
                else "gfx1151_native_children"
            ),
            "graph_reentry": False,
            "predicate_replay": False,
        }

    @property
    def residual_digest(self) -> str:
        if self.residual_abi is not None:
            return self.residual_abi.digest
        payload = json.dumps(
            {
                "schema": "tessera.branch_residual_abi.v1",
                "cfg_digest": self.cfg_digest,
                "state_shape_bound": list(self.state_shape_bound),
                "inactive_arm": "initialized_zero_extent_sentinel",
                "selection": "forward_predicate",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha256_text(payload)

    @property
    def artifact_digest(self) -> str:
        payload = json.dumps(self.contract(), sort_keys=True, separators=(",", ":"))
        return _sha256_text(payload)


def build_dynamic_if_product(
    *, target: str, paired_ir: str, cfg_digest: str, state_shape_bound: Sequence[int]
) -> W4RegionProduct:
    """Bind a dynamic-extent saved-activation branch product."""

    return W4RegionProduct(
        target=target,  # type: ignore[arg-type]
        kind="dynamic_if_saved_activation",
        paired_ir_digest=_sha256_text(paired_ir),
        cfg_digest=cfg_digest,
        state_shape_bound=tuple(int(dim) for dim in state_shape_bound),
    )


def build_hybrid_while_product(
    *,
    target: str,
    paired_ir: str,
    cfg_digest: str,
    state_shape_bound: Sequence[int],
    steps: int,
    checkpoint_indices: Sequence[int],
) -> W4RegionProduct:
    """Bind a sparse HYBRID state tape to one paired while product."""

    shape = tuple(int(dim) for dim in state_shape_bound)
    state_bytes = 4
    for dim in shape:
        state_bytes *= dim
    indices = tuple(int(index) for index in checkpoint_indices)
    candidate = TreeverseCandidate(
        steps=steps,
        checkpoint_slots=len(indices),
        checkpoint_indices=indices,
        retained_residual_bytes=state_bytes * len(indices),
        estimated_replayed_steps=_hybrid_replay_count(steps, indices),
        estimated_backward_work_ns=0,
    )
    abi = build_region_residual_abi(
        candidate, cfg_digest=cfg_digest, state_dtype="f32", state_shape=shape
    )
    return W4RegionProduct(
        target=target,  # type: ignore[arg-type]
        kind="hybrid_while_scale",
        paired_ir_digest=_sha256_text(paired_ir),
        cfg_digest=cfg_digest,
        state_shape_bound=shape,
        residual_abi=abi,
    )


def _hybrid_replay_count(steps: int, checkpoints: Sequence[int]) -> int:
    boundaries = (0, *checkpoints, steps)
    return sum(
        length * (length - 1) // 2
        for length in (
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        )
    )


def _native_binary(target: str, op_name: str, left: Any, right: Any) -> Any:
    # Runtime is imported lazily to keep compiler construction independent of
    # driver discovery and to make the physical boundary explicit at execute.
    from tessera import runtime

    if target == "x86":
        if not runtime._x86_elementwise_available():
            raise RuntimeError("W4 x86 product requires the AVX-512 production image")
        return runtime._x86_binary_exec(op_name, left, right)
    if target != "rocm_gfx1151" or runtime._rocm_chip() != "gfx1151":
        raise RuntimeError("W4 ROCm product requires the exact gfx1151 device")
    return runtime._rocm_binary_exec(op_name, left, right)


def _check_runtime_shape(product: W4RegionProduct, value: Any) -> tuple[int, ...]:
    import numpy as np

    array = np.asarray(value)
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != len(product.state_shape_bound) or any(
        actual > bound for actual, bound in zip(shape, product.state_shape_bound)
    ):
        raise ValueError(
            f"runtime state shape {shape} exceeds bound {product.state_shape_bound}"
        )
    if array.dtype != np.float32:
        raise ValueError("W4 physical validation products require f32 state")
    return shape


def execute_dynamic_if_product(
    product: W4RegionProduct,
    *,
    predicate: bool,
    saved_true: Any,
    saved_false: Any,
    cotangent: Any,
) -> Any:
    """Consume only the selected saved activation; never replay the branch.

    The paired fixture computes ``tanh(x)^2`` in the true arm and
    ``sigmoid(x)^2`` in the false arm.  Both derivatives are expressed solely
    through the selected saved activation, which makes accidental branch
    recomputation or inactive-placeholder consumption observable.
    """

    import numpy as np

    if product.kind != "dynamic_if_saved_activation":
        raise ValueError("not a dynamic-if W4 product")
    selected = np.asarray(saved_true if predicate else saved_false)
    shape = _check_runtime_shape(product, selected)
    ct = np.asarray(cotangent, dtype=np.float32)
    if ct.shape != shape:
        raise ValueError("branch cotangent and selected residual shapes disagree")
    one: Any = np.ones(shape, dtype=np.float32)
    two: Any = np.full(shape, 2.0, dtype=np.float32)
    square = _native_binary(product.target, "tessera.mul", selected, selected)
    if predicate:
        local = _native_binary(product.target, "tessera.sub", one, square)
        local = _native_binary(product.target, "tessera.mul", selected, local)
    else:
        complement = _native_binary(product.target, "tessera.sub", one, selected)
        local = _native_binary(product.target, "tessera.mul", square, complement)
    local = _native_binary(product.target, "tessera.mul", two, local)
    return _native_binary(product.target, "tessera.mul", ct, local)


def execute_hybrid_while_product(
    product: W4RegionProduct,
    *,
    initial_state: Any,
    scales: Sequence[float],
    cotangent: Any,
) -> TreeverseExecution:
    """Execute the exact sparse tape and bounded replay carried by the ABI."""

    import numpy as np

    if product.kind != "hybrid_while_scale" or product.residual_abi is None:
        raise ValueError("not a HYBRID while W4 product")
    shape = _check_runtime_shape(product, initial_state)
    if len(scales) != product.residual_abi.steps:
        raise ValueError("while scale count disagrees with the residual ABI")
    ct = np.asarray(cotangent, dtype=np.float32)
    if ct.shape != shape:
        raise ValueError("while cotangent and state shapes disagree")
    candidate = TreeverseCandidate(
        steps=product.residual_abi.steps,
        checkpoint_slots=len(product.residual_abi.slots),
        checkpoint_indices=product.residual_abi.checkpoint_indices,
        retained_residual_bytes=product.residual_abi.retained_bytes,
        estimated_replayed_steps=_hybrid_replay_count(
            product.residual_abi.steps, product.residual_abi.checkpoint_indices
        ),
        estimated_backward_work_ns=0,
    )

    def multiply(index: int, value: Any) -> Any:
        scale: Any = np.full(shape, np.float32(scales[index]), dtype=np.float32)
        return _native_binary(product.target, "tessera.mul", value, scale)

    capture = capture_treeverse_forward(
        candidate=candidate,
        initial_state=np.asarray(initial_state, dtype=np.float32),
        forward_step=multiply,
        residual_abi=product.residual_abi,
    )
    return execute_treeverse_region_adjoint(
        cotangent=ct,
        residuals=capture.residuals,
        forward_step=multiply,
        backward_step=lambda index, before, after, adjoint: multiply(index, adjoint),
    )


__all__ = [
    "W4_REGION_PRODUCT_SCHEMA",
    "W4RegionProduct",
    "build_dynamic_if_product",
    "build_hybrid_while_product",
    "execute_dynamic_if_product",
    "execute_hybrid_while_product",
]
