"""Owning gfx1201 public replay of the bounded-row native quantized MLP.

GFX1201-PUBLIC-QUANTIZED-ROW-BOUND-20261027. One explicit
``compile_native_quantized_program(..., m_bound=B)`` call; every later public
call at 1 <= M <= B binds that capacity program without retracing or compiling.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import provider, static_rows
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

SHAPES = [((32, 48, 19), 8, (3, 8, 1, 5)),
          ((16, 16, 1), 1, (1,)),
          ((128, 12288, 256), 256, (37, 256, 1))]


@pytest.mark.parametrize("dims,bound,actives", SHAPES)
def test_public_bound_replays_every_active_m_without_compiling(dims, bound, actives, monkeypatch):
    lib = provider()
    h, k, n = dims
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(actives[0], h, k, n))
    program = fn.compile_native_quantized_program(*inputs(actives[0], h, k, n), m_bound=bound)
    assert program.bounded
    capacity = package_quantized_product_graph(source(bound, h, k, n))

    oracles = {m: package_quantized_product_graph(source(m, h, k, n)) for m in actives if m <= 64}

    def forbidden(*a, **kw):
        pytest.fail("compiler or tracer ran during bounded public replay")
    digest = program.native.binding_digest
    for index, m in enumerate(actives):
        roots = inputs(m, h, k, n, seed=600 + index)
        before = [x.copy() for x in roots]
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", forbidden)
            patch.setattr(fn, "_trace_frontend_capture", forbidden)
            patch.setattr(fn, "_fn", forbidden)
            actual = fn(*roots)
        assert actual.shape == (m, n)
        assert fn.execution_kind == "native_gpu"
        assert fn._rocm_quantized_product_last_program is program
        assert len(fn._rocm_quantized_product_last_receipts) == 7
        for root, held in zip(roots, before, strict=True):
            np.testing.assert_array_equal(root, held)
        expected = static_rows(capacity, roots, bound, lib, seed=900 + index)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        if m <= 64:
            # Independent staged math at the active shape.
            assert_compiled_chain(oracles[m], roots, actual, lib)
    assert program.native.binding_digest == digest
    report = fn.compile_report()
    assert report.target_decision["rocm_gfx1201"].startswith(
        "canonical_rocm_quantized_bounded_mlp_program; 7 ordered native packages")
    with pytest.raises(ValueError, match="no portable record"):
        fn.runtime_artifact()
    with pytest.raises(ValueError, match="capacity"):
        fn(*inputs(bound + 1, h, k, n, seed=17))


def test_public_bound_refuses_changed_weight_shapes():
    provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(3, 32, 48, 19))
    fn.compile_native_quantized_program(*inputs(3, 32, 48, 19), m_bound=8)
    # The bound rebinds rows only: N, K and the weight roots stay fixed.
    with pytest.raises(ValueError, match="projection boundaries"):
        fn(*inputs(3, 32, 48, 21))
    roots = inputs(3, 32, 48, 19)
    roots[0] = inputs(3, 32, 64, 19)[0]
    with pytest.raises(ValueError, match="storage/shape"):
        fn(*roots)
