"""Public explicit row-bound compilation of the native seven-member quantized MLP.

GFX1201-PUBLIC-QUANTIZED-ROW-BOUND-20261027. Packaging only: the owning-device
replay lives in tests/device/rocm/test_public_quantized_mlp_row_bound_device.py.
"""
import json

import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.benchmark_public_quantized_gate_up import make_gate_up
from tessera.compiler.rocm_quantized_product_frontend import _compiler_path, runtime_artifact


def public(m=3, h=32, k=48, n=19):
    return ts.jit(target="rocm_gfx1201")(make_mlp(m, h, k, n)), inputs(m, h, k, n)


def test_explicit_bound_compiles_the_bounded_v5_program(rocm_image_toolchain):
    fn, roots = public()
    program = fn.compile_native_quantized_program(*roots, m_bound=8)
    plan = json.loads(program.native.program_json)
    assert program.bounded and plan["schema"] == "tessera.native.quantized_product.v5"
    assert (plan["active_m"], plan["m_bound"]) == (3, 8)
    assert program.argument_names == tuple(fn.arg_names)
    assert _compiler_path(program.native) == "canonical_rocm_quantized_bounded_mlp_program"
    assert fn._rocm_quantized_product_bounded_program is program
    assert "tessera.native.quantized_m_bound" in fn.graph_ir.module_attrs
    for refuse in (program.manifest, lambda: runtime_artifact(program)):
        with pytest.raises(ValueError, match="no portable record"):
            refuse()


def test_unbounded_compile_keeps_the_static_v4_program(rocm_image_toolchain):
    fn, roots = public()
    program = fn.compile_native_quantized_program(*roots)
    assert not program.bounded
    assert program.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v4"
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None


@pytest.mark.parametrize("bound", [0, -1, 2.0, True, "8"])
def test_malformed_bound_is_refused_before_tracing(bound, monkeypatch):
    fn, roots = public()
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a malformed bound"))
    with pytest.raises(ValueError, match="positive integer"):
        fn.compile_native_quantized_program(*roots, m_bound=bound)


def test_bound_below_the_example_rows_is_refused(rocm_image_toolchain):
    fn, roots = public()
    # The native export owns admission; its diagnostic crosses the compiler boundary.
    with pytest.raises(RuntimeError, match="exceed their checked capacity"):
        fn.compile_native_quantized_program(*roots, m_bound=2)
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None


def test_bound_on_the_gate_up_program_is_refused(rocm_image_toolchain):
    from benchmarks.rocm.benchmark_public_quantized_gate_up import inputs as gate_inputs
    fn = ts.jit(target="rocm_gfx1201")(make_gate_up(3, 32, 48))
    with pytest.raises(RuntimeError, match="require the complete seven-member MLP"):
        fn.compile_native_quantized_program(*gate_inputs(3, 32, 48), m_bound=8)


def test_non_gfx1201_target_is_refused(monkeypatch):
    # Other targets already refuse this Graph at decoration; the compile
    # entry point still refuses on its own.
    fn, roots = public()
    monkeypatch.setattr(fn, "target", "rocm_gfx1151")
    with pytest.raises(ValueError, match="gfx1201"):
        fn.compile_native_quantized_program(*roots, m_bound=8)


def _cached_bounds(fn):
    return [m.module_attrs.get("tessera.native.quantized_m_bound")
            for m in fn._traced_frontend_specializations.values()]


def test_bound_request_never_reaches_the_cached_trace(rocm_image_toolchain):
    fn, roots = public()
    with pytest.raises(RuntimeError, match="exceed their checked capacity"):
        fn.compile_native_quantized_program(*roots, m_bound=2)
    assert _cached_bounds(fn) == [None]
    bounded = fn.compile_native_quantized_program(*roots, m_bound=8)
    assert bounded.bounded and _cached_bounds(fn) == [None]
    static = fn.compile_native_quantized_program(*roots)
    assert not static.bounded
    assert static.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v4"
