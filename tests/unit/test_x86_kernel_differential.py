"""E2E-REAL-6, x86 elementwise / cohort-2 / breadth: retired constructors vs the compiled route.

``x86_native.package_elementwise`` / ``package_cohort2`` and
``x86_breadth.package_graph_breadth`` used to read the Python Graph object to
decide admission (``_elementwise_contract`` / ``_cohort2_contract`` /
``graph_breadth_contract``) and authored Tile IR text beside the compiled
route. Admission and packaging now belong to the native Graph -> Schedule ->
Tile route (``NativeX86Kernel.h``, ``NativeAbsolute.h``, ``schedule.norm``) and
``x86_native.package_scheduled_kernel``. The frozen constructors in
``tests/_support/x86_kernel_baseline.py`` are the declared oracle
(Decision #31(a)); these tests are its differential.

Host-free half (any host with ``tessera-opt``): over the enumerated envelope
the old contracts admitted, the compiled route admits the same cases and
reproduces every descriptor / ABI field, the Tile launch-op attributes, and
the Target IR ``tessera-x86-executable`` produces from each (the C-ABI call
and its kind constant); what the old route refused is still refused; each
case it admitted that the compiled route refuses is pinned as intentional.
Device half (an x86 host with the AVX-512 shared image; skips elsewhere):
both packages run on the same inputs and must agree bit-for-bit.

ALiBi is not in the envelope: it keeps its retired constructor in production
(``x86_native._alibi_contract``) because its Graph operand list is not
decodable by position, so the cohort-2 family stays a bootstrap-prune gap.
"""

from __future__ import annotations

import math
import re
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import native_x86_kernel, scheduled_kernel, x86_breadth, x86_native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.x86_pipeline import X86ExecutablePipeline
from tests._support import x86_kernel_baseline as baseline

PIPELINE = "tessera-lower-to-x86"
AVX512 = x86_native.X86_AVX512_ARCHITECTURE
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")
_ELEMENT = {"fp32": "f32", "bool": "i1", "int32": "i32", "int64": "i64"}


def _type(shape: tuple[int, ...], dtype: str) -> IRType:
    dims = "x".join(map(str, shape))
    return IRType(f"tensor<{dims + 'x' if dims else ''}{_ELEMENT[dtype]}>", tuple(map(str, shape)), dtype)


def _module(op_name, inputs, output, kwargs=None, name="x86_kernel") -> GraphIRModule:
    """``inputs``: (name, shape, dtype) per operand; ``output``: (shape, dtype)."""
    unique = {}
    for arg_name, shape, dtype in inputs:
        unique.setdefault(arg_name, IRArg(arg_name, _type(shape, dtype)))
    result = _type(*output)
    return GraphIRModule(functions=[GraphIRFunction(
        name=name, args=list(unique.values()), result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=[f"%{n}" for n, _, _ in inputs],
                   operand_types=[str(_type(s, d)) for _, s, d in inputs],
                   result_type=str(result), kwargs=dict(kwargs or {}))],
        return_values=["%o"],
    )])


# ---------------------------------------------------------------------------
# The envelope the retired constructors served.
# ---------------------------------------------------------------------------

_SHAPES = [(1,), (17,), (3, 17), (2, 3, 5)]


def _elementwise_cases():
    tables = [
        ("unary", x86_native.X86_UNARY_KINDS, "fp32", "fp32"),
        ("binary", x86_native.X86_BINARY_KINDS, "fp32", "fp32"),
        ("predicate", x86_native.X86_PREDICATE_KINDS, "fp32", "bool"),
        ("compare", x86_native.X86_COMPARE_KINDS, "fp32", "bool"),
        ("logical", x86_native.X86_LOGICAL_KINDS, "bool", "bool"),
        ("bitwise", x86_native.X86_BITWISE_KINDS, "int32", "int32"),
        ("transcendental", x86_native.X86_TRANSCENDENTAL_KINDS, "fp32", "fp32"),
        ("binary_math", x86_native.X86_BINARY_MATH_KINDS, "fp32", "fp32"),
    ]
    for family, kinds, dtype, out in tables:
        for op, kind in kinds.items():
            arity = (2 if family in {"binary", "compare", "binary_math"} else
                     (1 if kind in {"not", "popcount"} else 2) if family in {"logical", "bitwise"} else 1)
            for shape in _SHAPES:
                yield (f"{op}{shape}", _module(op, [(n, shape, dtype) for n in "ab"[:arity]], (shape, out)))
    for shape in _SHAPES:
        yield (f"tessera.where{shape}", _module(
            "tessera.where", [("c", shape, "bool"), ("a", shape, "fp32"), ("b", shape, "fp32")], (shape, "fp32")))


def _cohort2_cases():
    for op in x86_native.X86_ARGREDUCE_KINDS:
        for shape in _SHAPES:
            for axis in sorted({-1, len(shape) - 1}) + [None, "missing"]:
                for keepdims in (False, True):
                    if axis is None and len(shape) > 1:
                        continue  # pinned below (keepdims) / the flattened-descriptor correction
                    if axis is None:
                        out = (1,) if keepdims else ()
                    else:
                        out = shape[:-1] + ((1,) if keepdims else ())
                    kwargs = {"keepdims": keepdims}
                    if axis != "missing":
                        kwargs["axis"] = axis
                    yield (f"{op}{shape}axis={axis}keep={keepdims}",
                           _module(op, [("x", shape, "fp32")], (out, "int32"), kwargs))
    for op in x86_native.X86_SCAN_KINDS:
        for shape in _SHAPES:
            for axis in sorted({-1, len(shape) - 1}) + ["missing"]:
                kwargs = {} if axis == "missing" else {"axis": axis}
                yield (f"{op}{shape}axis={axis}", _module(op, [("x", shape, "fp32")], (shape, "fp32"), kwargs))
        # axis=None over a rank-1 operand is the last axis. cumsum is excluded:
        # the retired route admitted it and then failed to package it (below).
        if op != "tessera.cumsum":
            yield (f"{op}(9,)axis=None", _module(op, [("x", (9,), "fp32")], ((9,), "fp32"), {"axis": None}))
    for op in x86_native.X86_NORM_KINDS:
        for shape in _SHAPES:
            for eps in ("default", 1e-5, 1e-3, 0.25):
                kwargs = {} if eps == "default" else {"eps": eps}
                yield (f"{op}{shape}eps={eps}", _module(op, [("x", shape, "fp32")], (shape, "fp32"), kwargs))
    for shape in [(1, 2), (3, 18), (2, 3, 4)]:
        yield (f"tessera.rope{shape}", _module(
            "tessera.rope", [("x", shape, "fp32"), ("theta", shape, "fp32")], (shape, "fp32")))


def _breadth_cases():
    for n, m in [(20, 7), (1, 1), (64, 300)]:
        for kwargs in ({}, {"axis": 0}, {"axis": -1}):
            yield (f"gather{n},{m}{kwargs}", _module(
                "tessera.gather", [("source", (n,), "fp32"), ("idx", (m,), "int64")], ((m,), "fp32"), kwargs))
    for op, parameter in [("tessera.mse_loss", None), ("tessera.loss.mse", None), ("tessera.mae_loss", None),
                          ("tessera.loss.mae", None), ("tessera.huber_loss", "delta"),
                          ("tessera.loss.huber", "delta"), ("tessera.smooth_l1_loss", "beta"),
                          ("tessera.loss.smooth_l1", "beta"), ("tessera.log_cosh_loss", None),
                          ("tessera.loss.log_cosh", None)]:
        for shape in [(17,), (3, 17)]:
            values = [None] if parameter is None else [None, 0.5, 2.0]
            for value in values:
                kwargs = {"reduction": "none", **({parameter: value} if value is not None else {})}
                yield (f"{op}{shape}{kwargs}", _module(
                    op, [("pred", shape, "fp32"), ("target", shape, "fp32")], (shape, "fp32"), kwargs))
    for shape in [(3, 3), (5, 5)]:
        yield (f"cholesky{shape}", _module("tessera.cholesky", [("matrix", shape, "fp32")], (shape, "fp32")))
        yield (f"cholesky{shape}lower", _module(
            "tessera.cholesky", [("matrix", shape, "fp32")], (shape, "fp32"), {"lower": True}))
    for matrix, rhs in [((4, 4), (4, 3)), ((6, 6), (6, 1))]:
        for kwargs in ({}, {"lower": True}, {"lower": False}):
            yield (f"tri_solve{matrix}{rhs}{kwargs}", _module(
                "tessera.tri_solve", [("matrix", matrix, "fp32"), ("rhs", rhs, "fp32")], (rhs, "fp32"), kwargs))


def _batched_linalg_cases():
    """The retained constructor's envelope (rank-3 cholesky / tri_solve)."""
    yield ("cholesky(2, 4, 4)", _module("tessera.cholesky", [("matrix", (2, 4, 4), "fp32")], ((2, 4, 4), "fp32")))
    for kwargs in ({}, {"lower": True}, {"lower": False}):
        yield (f"tri_solve(2, 4, 4){kwargs}", _module(
            "tessera.tri_solve", [("matrix", (2, 4, 4), "fp32"), ("rhs", (2, 4, 5), "fp32")],
            ((2, 4, 5), "fp32"), kwargs))


_CASES = {
    "elementwise": list(_elementwise_cases()),
    "cohort2": list(_cohort2_cases()),
    "breadth": list(_breadth_cases()),
}
_ALL = [(family, case_id, module) for family, cases in _CASES.items() for case_id, module in cases]

_OLD = {
    "elementwise": (baseline.supports_elementwise, baseline.package_elementwise),
    "cohort2": (baseline.supports_cohort2, baseline.package_cohort2),
    "breadth": (baseline.supports_graph_breadth, baseline.package_graph_breadth),
}
_NEW = {
    "elementwise": (x86_native.supports_elementwise, x86_native.package_elementwise),
    "cohort2": (x86_native.supports_cohort2, x86_native.package_cohort2),
    "breadth": (x86_breadth.supports_graph_breadth, x86_breadth.package_graph_breadth),
}


def _stub_lower(monkeypatch):
    def fake(tile, symbol, family, architecture=AVX512):
        return f"call @{symbol}", b"image", "c", "t"
    monkeypatch.setattr(x86_native, "_lower", fake)
    monkeypatch.setattr(x86_breadth, "_lower", fake)


def _kernel_op(tile_ir: str) -> tuple[str, dict[str, str]]:
    match = re.search(r"(?m)^\s*(tile\.\w+) [^{]*\{([^{}]*)\}", tile_ir)
    assert match, tile_ir
    pairs = re.findall(r'([\w.]+) = ("[^"]*"|-?\d+ : i64|true|false)', match[2])
    return match[1], {key: value for key, value in pairs if not key.startswith("tessera.")}


def _target_calls(tile_ir: str, family: str) -> list[str]:
    """The C-ABI calls and constants ``tessera-x86-executable`` makes of *tile_ir*.

    Normalization rides ``schedule.norm``, whose Tile launch binds its
    rows/cols as constants where the retired constructor took them as entry
    arguments, so for norm the comparison is the called symbol and the f32
    epsilon constant the kernel receives.
    """
    pipeline = X86ExecutablePipeline(family=family, architecture=AVX512).pass_pipeline()
    result = subprocess.run([str(find_tessera_opt()), "-", f"--pass-pipeline={pipeline}"],
                            input=tile_ir, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    if family == "norm":
        symbols = re.findall(r"(?:func\.)?call @(\w+)\(", result.stdout)
        epsilons = [float(v) for v in re.findall(r"arith\.constant ([-+0-9.eE]+) : f32", result.stdout)]
        return [*symbols, *(repr(np.float32(v)) for v in epsilons)]
    return [line.strip() for line in result.stdout.splitlines()
            if "func.call" in line or "arith.constant" in line or "func.func private" in line]


def _pipeline_family(package) -> str:
    provenance = package.descriptor.provenance
    if provenance.get("graph_level"):
        return str(provenance["family"])
    family = str(provenance["family"])
    return "elementwise" if family in {"unary", "binary", "predicate", "compare", "logical",
                                        "bitwise", "where", "transcendental", "binary_math"} else family


_SHARED_SKIP = {"route", "work_item", "eps"}


@_needs_compiler
@pytest.mark.parametrize("family,case_id,module", _ALL, ids=[c[1] for c in _ALL])
def test_compiled_route_reproduces_retired_descriptor_tile_and_target(monkeypatch, family, case_id, module):
    _stub_lower(monkeypatch)
    old_supports, old_package = _OLD[family]
    new_supports, new_package = _NEW[family]
    assert old_supports(module), case_id
    assert new_supports(module), case_id
    assert x86_native.native_package_kind(module) == family
    old = old_package(module, pipeline_name=PIPELINE)
    new = new_package(module, pipeline_name=PIPELINE)
    o, n = old.descriptor, new.descriptor
    assert (o.entry_symbol, o.abi_id, o.buffers, o.scalars, o.shape_guards, o.geometry, o.ordering) == (
        n.entry_symbol, n.abi_id, n.buffers, n.scalars, n.shape_guards, n.geometry, n.ordering)
    for key, value in o.provenance.items():
        if key not in _SHARED_SKIP:
            assert n.provenance[key] == value, key
    if "eps" in o.provenance:  # the Graph double vs the f32 the kernel receives
        assert np.float32(o.provenance["eps"]) == np.float32(n.provenance["eps"])
        assert n.provenance["eps"] == float(np.float32(o.provenance["eps"]))
    assert n.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert n.provenance["work_item"] == "E2E-REAL-6"
    assert len(n.provenance["schedule_digest"]) == 64
    assert _kernel_op(old.tile_ir) == _kernel_op(new.tile_ir)
    pipeline_family = _pipeline_family(old)
    assert _target_calls(old.tile_ir, pipeline_family) == _target_calls(new.tile_ir, pipeline_family)
    assert (old.image.architecture, old.image.entry_points) == (new.image.architecture, new.image.entry_points)


def _flatten_scan_rank2():
    return _module("tessera.cumprod", [("x", (3, 17), "fp32")], ((51,), "fp32"), {"axis": None})


_OLD_ADMITTED_NEW_REFUSED = {
    # The retired contract admitted these and then could not package them: a
    # launch descriptor names each buffer once (E_LAUNCH_DESCRIPTOR_SCHEMA).
    "repeated_operand": ("elementwise", _module(
        "tessera.add", [("x", (3, 17), "fp32"), ("x", (3, 17), "fp32")], ((3, 17), "fp32"))),
    # NumPy keeps every axis for a flattened keepdims argmax; the ABI result is
    # rank-1 and the retired constructor declared (1,).
    "argmax_flatten_keepdims_rank2": ("cohort2", _module(
        "tessera.argmax", [("x", (3, 17), "fp32")], ((1,), "int32"), {"axis": None, "keepdims": True})),
    # The Graph verifier requires rank >= 2 for rope (LEGALITY_ROPE_RANK).
    "rope_rank1": ("cohort2", _module(
        "tessera.rope", [("x", (6,), "fp32"), ("t", (6,), "fp32")], ((6,), "fp32"))),
    # Keyword attributes the C ABI does not implement were ignored (#21a).
    "elementwise_unknown_keyword": ("elementwise", _module(
        "tessera.add", [("a", (3, 17), "fp32"), ("b", (3, 17), "fp32")], ((3, 17), "fp32"), {"alpha": 2.0})),
    "compare_signedness": ("elementwise", _module(
        "tessera.lt", [("a", (3, 17), "fp32"), ("b", (3, 17), "fp32")], ((3, 17), "bool"),
        {"signedness": "signed"})),
    "argmax_integer_keepdims": ("cohort2", _module(
        "tessera.argmax", [("x", (3, 17), "fp32")], ((3, 1), "int32"), {"axis": -1, "keepdims": 1})),
    "scan_flatten_rank2": ("cohort2", _flatten_scan_rank2()),
    "norm_numeric_policy": ("cohort2", _module(
        "tessera.rmsnorm", [("x", (3, 17), "fp32")], ((3, 17), "fp32"), {"numeric_policy": "fast"})),
    "norm_non_last_axis": ("cohort2", _module(
        "tessera.layer_norm", [("x", (3, 17), "fp32")], ((3, 17), "fp32"), {"axis": 0})),
    "cholesky_upper": ("breadth", _module(
        "tessera.cholesky", [("matrix", (3, 3), "fp32")], ((3, 3), "fp32"), {"lower": False})),
    "tri_solve_transpose": ("breadth", _module(
        "tessera.tri_solve", [("matrix", (4, 4), "fp32"), ("rhs", (4, 3), "fp32")], ((4, 3), "fp32"),
        {"trans": True})),
    "tri_solve_unit_diagonal": ("breadth", _module(
        "tessera.tri_solve", [("matrix", (4, 4), "fp32"), ("rhs", (4, 3), "fp32")], ((4, 3), "fp32"),
        {"unit_diag": True})),
    "huber_zero_delta": ("breadth", _module(
        "tessera.huber_loss", [("p", (8,), "fp32"), ("t", (8,), "fp32")], ((8,), "fp32"),
        {"reduction": "none", "delta": 0.0})),
    "loss_unknown_keyword": ("breadth", _module(
        "tessera.mse_loss", [("p", (8,), "fp32"), ("t", (8,), "fp32")], ((8,), "fp32"),
        {"reduction": "none", "weight": 2.0})),
    "gather_unknown_keyword": ("breadth", _module(
        "tessera.gather", [("s", (8,), "fp32"), ("i", (4,), "int64")], ((4,), "fp32"),
        {"axis": 0, "mode": "clip"})),
}


_BATCHED = list(_batched_linalg_cases())


@_needs_compiler
@pytest.mark.parametrize("case_id,module", _BATCHED, ids=[c[0] for c in _BATCHED])
def test_batched_linalg_keeps_the_retained_constructor(monkeypatch, case_id, module):
    """Rank-3 cholesky / tri_solve cannot be Graph IR yet (the ODS ops are
    rank-2), so they stay on the retained constructor -- identical packages,
    no scheduled route claimed."""
    _stub_lower(monkeypatch)
    assert baseline.supports_graph_breadth(module) and x86_breadth.supports_graph_breadth(module)
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="x86")
    old = baseline.package_graph_breadth(module, pipeline_name=PIPELINE)
    new = x86_breadth.package_graph_breadth(module, pipeline_name=PIPELINE)
    assert old.descriptor == new.descriptor and old.tile_ir == new.tile_ir


@_needs_compiler
@pytest.mark.parametrize("op", sorted(x86_native.X86_ARGREDUCE_KINDS))
@pytest.mark.parametrize("shape", [(3, 17), (2, 3, 5)])
def test_flattened_argreduce_describes_its_real_operand(monkeypatch, op, shape):
    """``argmax(axis=None)`` over a rank >= 2 operand: the retired constructor
    described the operand as rank-1 ``(N,)``, a shape the caller never passes,
    so the runtime refused its descriptor; the compiled route describes the
    real operand and reads it as one row of ``N`` contiguous elements."""
    _stub_lower(monkeypatch)
    module = _module(op, [("x", shape, "fp32")], ((), "int32"), {"axis": None})
    old = baseline.package_cohort2(module, pipeline_name=PIPELINE)
    new = x86_native.package_cohort2(module, pipeline_name=PIPELINE)
    n = math.prod(shape)
    assert old.descriptor.buffers[0].rank == 1 and old.descriptor.provenance["shape"] == [n]
    assert new.descriptor.buffers[0].rank == len(shape)
    assert new.descriptor.provenance["shape"] == list(shape)
    assert (new.descriptor.provenance["rows"], new.descriptor.provenance["cols"]) == (1, n)
    assert (old.descriptor.scalars, old.descriptor.buffers[1]) == (new.descriptor.scalars, new.descriptor.buffers[1])


@_needs_compiler
@pytest.mark.parametrize("op", sorted(x86_native.X86_ARGREDUCE_KINDS))
def test_device_flattened_argreduce_executes_where_the_retired_descriptor_was_refused(op):
    if not x86_native.tools_available_for_architecture(AVX512):
        pytest.skip(f"{AVX512} shared image not available on this host")
    module = _module(op, [("x", (3, 17), "fp32")], ((), "int32"), {"axis": None})
    x = np.random.default_rng(7).standard_normal((3, 17)).astype(np.float32)
    old = baseline.package_cohort2(module, pipeline_name=PIPELINE)
    with pytest.raises(AssertionError):
        _run(old, {"x": x})
    got = _run(x86_native.package_cohort2(module, pipeline_name=PIPELINE), {"x": x})
    expected = np.argmax(x) if op == "tessera.argmax" else np.argmin(x)
    assert int(got) == int(expected)


@_needs_compiler
def test_compiled_route_serves_the_rank1_cumsum_the_retired_route_could_not(monkeypatch):
    """``cumsum(axis=None)`` over a rank-1 operand is its last axis. The retired
    route admitted it and then emitted ``axis = none`` (not MLIR); the compiled
    route normalizes the spelling and packages it."""
    _stub_lower(monkeypatch)
    module = _module("tessera.cumsum", [("x", (9,), "fp32")], ((9,), "fp32"), {"axis": None})
    assert baseline.supports_cohort2(module)
    with pytest.raises(RuntimeError):
        baseline.package_cohort2(module, pipeline_name=PIPELINE)
    package = x86_native.package_cohort2(module, pipeline_name=PIPELINE)
    assert package.descriptor.abi_id == x86_native.X86_SCAN_F32_ABI
    assert package.descriptor.provenance["kind"] == "sum"


@pytest.mark.parametrize("name", sorted(_OLD_ADMITTED_NEW_REFUSED))
def test_cases_the_retired_route_admitted_are_refused_on_purpose(monkeypatch, name):
    family, module = _OLD_ADMITTED_NEW_REFUSED[name]
    assert _OLD[family][0](module)
    assert not _NEW[family][0](module)
    assert not x86_native.supports_native_package(module)
    monkeypatch.setattr(x86_native, "_lower", lambda *a, **k: pytest.fail("refused case reached lowering"))
    with pytest.raises(ValueError):
        _NEW[family][1](module, pipeline_name=PIPELINE)


@_needs_compiler
def test_retired_cumsum_flatten_admitted_but_never_packaged():
    """The retired route admitted a flattened rank-2 cumsum and then failed to
    package it (``axis = none`` is not MLIR); the compiled route refuses it at
    admission instead."""
    module = _module("tessera.cumsum", [("x", (3, 17), "fp32")], ((51,), "fp32"), {"axis": None})
    assert baseline.supports_cohort2(module)
    with pytest.raises((RuntimeError, ValueError)):
        baseline.package_cohort2(module, pipeline_name=PIPELINE)
    assert not x86_native.supports_cohort2(module)


def test_norm_epsilon_keyword_is_honoured_not_dropped():
    """The retired x86 norm read only ``eps`` and silently used 1e-5 for an
    ``epsilon=`` keyword; the scheduled contract reads both (the NVIDIA norm
    contract's spelling), so the requested epsilon reaches the kernel."""
    module = _module("tessera.rmsnorm", [("x", (3, 17), "fp32")], ((3, 17), "fp32"), {"epsilon": 1e-3})
    assert baseline.supports_cohort2(module) and x86_native.supports_cohort2(module)
    assert baseline._cohort2_contract(module)["eps"] == 1e-5
    if find_tessera_opt() is not None:
        artifact = scheduled_kernel.lower_scheduled_kernel(module, target="x86")
        assert artifact.epsilon == float(np.float32(1e-3))


_BOTH_REFUSE = {
    "dynamic_shape": ("elementwise", _module("tessera.exp", [("x", (3, 17), "fp32")], ((3, 17), "fp32"))),
    "f16_add": ("elementwise", _module(
        "tessera.add", [("a", (3, 17), "fp32"), ("b", (3, 17), "fp32")], ((3, 17), "fp32"))),
    "mismatched_shapes": ("elementwise", _module(
        "tessera.add", [("a", (3, 17), "fp32"), ("b", (3, 16), "fp32")], ((3, 17), "fp32"))),
    "wrong_arity": ("elementwise", _module("tessera.add", [("a", (3, 17), "fp32")], ((3, 17), "fp32"))),
    "predicate_f32_result": ("elementwise", _module("tessera.isnan", [("a", (4,), "fp32")], ((4,), "fp32"))),
    "argmax_non_last_axis": ("cohort2", _module(
        "tessera.argmax", [("x", (3, 17), "fp32")], ((17,), "int32"), {"axis": 0})),
    "argmax_f32_result": ("cohort2", _module(
        "tessera.argmax", [("x", (3, 17), "fp32")], ((3,), "fp32"), {"axis": -1})),
    "odd_rope": ("cohort2", _module(
        "tessera.rope", [("x", (3, 17), "fp32"), ("t", (3, 17), "fp32")], ((3, 17), "fp32"))),
    "negative_eps": ("cohort2", _module(
        "tessera.rmsnorm", [("x", (3, 17), "fp32")], ((3, 17), "fp32"), {"eps": -1.0})),
    "gather_rank2": ("breadth", _module(
        "tessera.gather", [("s", (4, 4), "fp32"), ("i", (3,), "int64")], ((3,), "fp32"))),
    "gather_int32_indices": ("breadth", _module(
        "tessera.gather", [("s", (8,), "fp32"), ("i", (3,), "int32")], ((3,), "fp32"))),
    "loss_mean_reduction": ("breadth", _module(
        "tessera.mse_loss", [("p", (8,), "fp32"), ("t", (8,), "fp32")], ((8,), "fp32"), {"reduction": "mean"})),
    "cholesky_rectangular": ("breadth", _module(
        "tessera.cholesky", [("m", (3, 4), "fp32")], ((3, 4), "fp32"))),
}
_BOTH_REFUSE["dynamic_shape"][1].functions[0].args[0].ir_type = IRType("tensor<?x17xf32>", ("?", "17"), "fp32")
_f16 = _BOTH_REFUSE["f16_add"][1].functions[0]
_f16.args[1].ir_type = IRType("tensor<3x17xf16>", ("3", "17"), "fp16")


@pytest.mark.parametrize("name", sorted(_BOTH_REFUSE))
def test_refusals_are_unchanged(name):
    family, module = _BOTH_REFUSE[name]
    assert not _OLD[family][0](module)
    assert not _NEW[family][0](module)


def test_retired_constructors_left_production():
    for name in ("emit_elementwise_tile_ir", "emit_cohort2_tile_ir", "_elementwise_contract",
                 "_cohort2_contract"):
        assert not hasattr(x86_native, name), name
    assert not hasattr(x86_breadth, "graph_breadth_contract")


def test_only_alibi_keeps_a_graph_owned_constructor():
    alibi = _module("tessera.alibi", [("slopes", (4,), "fp32")], ((4, 7, 7), "fp32"),
                    {"num_heads": 4, "seq_len": 7})
    assert x86_native._alibi_contract(alibi) is not None
    assert x86_native.supports_cohort2(alibi)
    assert not scheduled_kernel.supports_scheduled_kernel(alibi, target="x86")


def test_only_the_avx512_image_exists_for_these_families():
    module = _module("tessera.exp", [("x", (3, 17), "fp32")], ((3, 17), "fp32"))
    with pytest.raises(ValueError, match="zen5-avx512"):
        scheduled_kernel.lower_scheduled_kernel(module, target="x86", architecture="x86_64_base")


def test_native_contract_names_the_breadth_abi_registry():
    """``NativeX86Kernel.h`` names the breadth C ABI (symbol, ABI id, family);
    it must agree with ``x86_breadth.X86_BREADTH_ABIS``, which the runtime reads."""
    from pathlib import Path

    header = (Path(__file__).resolve().parents[2]
              / "src/compiler/programming_model/lib/NativeX86Kernel.h").read_text()
    for key in ("gather_f32", "pointwise_loss_f32", "cholesky_f32", "tri_solve_f32"):
        spec = x86_breadth.X86_BREADTH_ABIS[key]
        row = re.search(r'\{"' + key + r'", "([^"]+)",\s*"([^"]+)",\s*"([^"]+)", "([^"]+)"', header)
        assert row is not None, key
        assert row.groups() == (spec.symbol, spec.abi_id, spec.family, spec.effects), key


def test_python_admission_names_the_native_op_table():
    """The host-free admission (``native_x86_kernel.X86_KERNEL_OPS``) and the
    native table (``kX86KernelSpecs``) own the same ops with the same kinds."""
    from pathlib import Path

    header = (Path(__file__).resolve().parents[2]
              / "src/compiler/programming_model/lib/NativeX86Kernel.h").read_text()
    native = {op: (family, sub, kind) for op, family, sub, kind in re.findall(
        r'\{"(tessera\.[\w.]+)", "(\w+)", "(\w+)", "(\w+)"\}', header)}
    python = {op: ("abi" if family == "x86_abi" else family, sub, kind)
              for op, (family, sub, kind) in native_x86_kernel.X86_KERNEL_OPS.items()}
    assert native == python


@_needs_compiler
def test_forged_contract_fails_closed(monkeypatch):
    _stub_lower(monkeypatch)
    from dataclasses import replace

    module = _module("tessera.sub", [("a", (3, 17), "fp32"), ("b", (3, 17), "fp32")], ((3, 17), "fp32"))
    artifact = scheduled_kernel.lower_scheduled_kernel(module, target="x86")
    swapped = replace(artifact, tile_ir=artifact.tile_ir.replace('kind = "sub"', 'kind = "add"'))
    with pytest.raises(ValueError):
        x86_native.package_scheduled_kernel(swapped, pipeline_name=PIPELINE)
    rebound = replace(artifact, input_names=("b", "a"), input_name="b")
    with pytest.raises(ValueError):
        x86_native.package_scheduled_kernel(rebound, pipeline_name=PIPELINE)
    altered = artifact.schedule_ir.replace('kind = "sub"', 'kind = "add"', 1)
    assert altered != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="altered"):
        from tessera.compiler.scheduled_matmul import run_tessera_opt
        run_tessera_opt(find_tessera_opt(), altered, "--tessera-schedule-to-tile")


# ---------------------------------------------------------------------------
# Device differential: the AVX-512 image, the same inputs, bitwise.
# ---------------------------------------------------------------------------

def _array(shape, dtype, rng, special=False):
    if dtype == "bool":
        return rng.random(shape) < 0.5
    if dtype == "int32":
        return rng.integers(-2**31, 2**31 - 1, size=shape, dtype=np.int64).astype(np.int32)
    if dtype == "int64":
        return rng.integers(0, 1 << 30, size=shape).astype(np.int64)
    x = (rng.standard_normal(shape) * 4).astype(np.float32)
    flat = x.reshape(-1)
    if special and flat.size >= 6:  # elementwise only: a row op would be all-NaN
        flat[:6] = [np.nan, np.inf, -np.inf, 0.0, -0.0, 88.5]
    return x


def _inputs(case_id, module, rng):
    function, op = module.functions[0], module.functions[0].body[0]
    args = {arg.name: arg for arg in function.args}
    values = {}
    for index, operand in enumerate(op.operands):
        name = operand.removeprefix("%")
        if name in values:
            continue
        shape = tuple(int(d) for d in args[name].ir_type.shape)
        dtype = str(args[name].ir_type.dtype)
        values[name] = _array(shape, dtype, rng, special=x86_native.requests_elementwise(module))
        if op.op_name == "tessera.gather" and index == 1:
            source = tuple(int(d) for d in args[op.operands[0].removeprefix("%")].ir_type.shape)
            values[name] = rng.integers(0, source[0], size=shape).astype(np.int64)
        if op.op_name in {"tessera.cholesky", "tessera.tri_solve"} and index == 0:
            n = shape[-1]
            raw = rng.standard_normal(shape).astype(np.float32)
            spd = raw @ np.swapaxes(raw, -1, -2) + n * np.eye(n, dtype=np.float32)
            values[name] = np.ascontiguousarray(spd if op.op_name == "tessera.cholesky"
                                                else np.tril(spd), dtype=np.float32)
    return values


def _run(package, inputs):
    d = package.descriptor
    args = dict(inputs)
    output = next(b for b in d.buffers if b.direction == "output")
    shape = tuple(g.value for g in d.shape_guards if g.binding == output.name)
    dtype = {"fp32": np.float32, "bool": np.bool_, "int32": np.int32}[output.dtype]
    result = np.zeros(shape, dtype)
    args[output.name] = result
    provenance = d.provenance
    if provenance.get("graph_level"):
        args.update(provenance["graph_scalars"])
    elif "elements" in provenance:
        args["N"] = provenance["elements"]
    else:
        args.update(Rows=provenance["rows"], Cols=provenance["cols"])
        if "eps" in provenance:
            args["Epsilon"] = provenance["eps"]
    artifact = rt.RuntimeArtifact(metadata={"target": "x86"}, native_image=package.image,
                                  launch_descriptor=d, tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    launched = rt.launch(artifact, args)
    assert launched["ok"], launched
    return result


@_needs_compiler
@pytest.mark.parametrize("family,case_id,module", _ALL, ids=[c[1] for c in _ALL])
def test_device_retired_and_compiled_images_agree_bitwise(family, case_id, module):
    if not x86_native.tools_available_for_architecture(AVX512):
        pytest.skip(f"{AVX512} shared image not available on this host")
    old = _OLD[family][1](module, pipeline_name=PIPELINE)
    new = _NEW[family][1](module, pipeline_name=PIPELINE)
    assert old.image.payload == new.image.payload
    for seed in (0, 1):
        inputs = _inputs(case_id, module, np.random.default_rng(seed))
        got_old, got_new = _run(old, inputs), _run(new, inputs)
        assert got_old.dtype == got_new.dtype and got_old.shape == got_new.shape
        assert got_old.tobytes() == got_new.tobytes(), case_id
        if (got_new.dtype == np.float32 and math.prod(got_new.shape) > 8
                and not x86_native.requests_elementwise(module)):
            assert np.isfinite(got_new).all(), case_id
