"""Public transform intent for compiler-owned JIT differentiation.

This module binds an existing semantic program to a separate native AD owner.
It does not evaluate the primal, construct derivative arithmetic, or emit Tile IR.
"""
from __future__ import annotations

import copy
from typing import Any, cast

from .autodiff_request import DifferentiationRequest
from .jit import JitFn
from .matmul_pipeline import normalize_target_kind


def requires_native_jvp(fn: JitFn) -> bool:
    """Explicit native targets must never lose their derivative on a Python tape."""
    return fn.native_required or normalize_target_kind(fn.target) not in {"cpu", "reference_cpu"}


def _source_witness(fn: JitFn):
    graph = fn._legacy_graph_ir or fn.graph_ir
    maps = tuple(getattr(fn, name, None) for name in (
        "_frontend_batch_axes", "_frontend_batch_depth",
        "_frontend_batch_policies", "_frontend_output_permutation"))
    return (graph, tuple(fn.constraints.constraints), fn._constraint_ir_args,
            maps, fn.target, fn.deterministic, fn.seed, fn.cpu_tile, fn.source_origin,
            fn._frontend_source_text, fn._fn)


def native_public_jvp(fn: JitFn, primals: tuple[Any, ...], tangents: tuple[Any, ...]):
    """Project public JVP activity into native Graph AD without changing the caller."""
    from tessera.autodiff.tape import TesseraAutodiffError

    fn.last_jvp_execution = None
    if len(primals) != len(fn.arg_names):
        raise ValueError("native public JVP requires one primal per frontend argument")
    active = tuple(index for index, tangent in enumerate(tangents) if tangent is not None)
    if not active:
        raise ValueError("native public JVP requires at least one active tangent")
    if getattr(fn, "_bounded_lhs", None) is not None:
        raise TesseraAutodiffError("bounded tensor JVP requires a native bounded AD projection")

    witness = _source_witness(fn)
    cache = getattr(fn, "_native_public_jvp_owners", None)
    if cache is None:
        cache = {}
        fn._native_public_jvp_owners = cache
    retained = cache.get(active)
    if retained is None or retained[0] != witness:
        if retained is not None:
            retained[1].close_native_storage()
        request = DifferentiationRequest(
            mode="forward", wrt=tuple(fn.arg_names[index] for index in active),
            wrt_indices=active, native_required=True)
        owner = JitFn(
            fn._fn, copy.deepcopy(witness[0]), fn.inferred_effect,
            copy.deepcopy(fn.constraints), deterministic=fn.deterministic,
            seed=fn.seed, target=fn.target, attn_config=copy.deepcopy(fn.attn_config),
            cpu_tile=cast(tuple[int, int, int], fn.cpu_tile), source_origin=fn.source_origin,
            source_text=fn._frontend_source_text, native_required=True,
            differentiation_request=request)
        owner._constraint_ir_args = copy.deepcopy(fn._constraint_ir_args)
        for name in ("_frontend_batch_axes", "_frontend_batch_depth",
                     "_frontend_batch_policies", "_frontend_output_permutation"):
            if hasattr(fn, name):
                setattr(owner, name, copy.deepcopy(getattr(fn, name)))
        if len(cache) >= 8 and active not in cache:
            retired = next(iter(cache))
            cache.pop(retired)[1].close_native_storage()
        retained = (copy.deepcopy(witness[:-1]) + (witness[-1],), owner)
        cache[active] = retained
    owner = retained[1]
    result = owner.native_jvp(*primals, tangents=tuple(tangents[index] for index in active))
    receipt = owner.last_jvp_execution
    if not isinstance(receipt, dict) or receipt.get("execution_kind") not in {"native_cpu", "native_gpu"}:
        raise TesseraAutodiffError("native public JVP lacks a native execution receipt")
    fn.last_jvp_execution = dict(receipt, public_transform="jvp", wrt_indices=active)
    return result
