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
    assert program.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v5"


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


def _bounded_record(fn, roots):
    program = fn.compile_native_quantized_program(*roots, m_bound=8)
    return program, program.manifest()


def test_bounded_record_round_trips_with_both_arbitrary_m_products(rocm_image_toolchain):
    """GFX1201-PORTABLE-BOUNDED-QUANTIZED-RECORD-20261028."""
    from tessera.compiler.rocm_quantized_product_frontend import _resolve
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    from tessera.runtime import RuntimeArtifact
    fn, roots = public()
    program, record = _bounded_record(fn, roots)
    native = record["native_program"]
    assert native["schema"] == "tessera.rocm.quantized_product_package.v5"
    assert native["gate"]["arbitrary_m"] is True and native["down"]["arbitrary_m"] is True
    replay = NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(native)))
    assert replay.bounded and replay.gate.arbitrary_m and replay.down.arbitrary_m
    assert replay.binding_digest == program.native.binding_digest
    assert replay.program_json == program.native.program_json
    artifact = RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    assert artifact.metadata["compiler_path"] == "canonical_rocm_quantized_bounded_mlp_program"
    resolved = _resolve(json.dumps(artifact.metadata["native_program"], sort_keys=True, allow_nan=False),
                        artifact.graph_ir, json.dumps(artifact.metadata["arg_names"]))
    assert resolved.bounded and resolved.native.binding_digest == program.native.binding_digest


def _tampered(record, edit):
    copy = json.loads(json.dumps(record["native_program"]))
    edit(copy)
    return copy


def _set_plan(key, value):
    def edit(record):
        plan = json.loads(record["program_json"])
        plan[key] = value
        record["program_json"] = json.dumps(plan, sort_keys=True)
    return edit


@pytest.mark.parametrize("edit,message", [
    # The package schema and the plan schema are bound to each other.
    (lambda r: r.update(schema="tessera.rocm.quantized_product_package.v4"), "schema/frame differs"),
    # A dropped claim breaks the component's binding digest, which covers it.
    (lambda r: r["gate"].pop("arbitrary_m"), "member binding differs"),
    (lambda r: r["down"].pop("arbitrary_m"), "member binding differs"),
    (lambda r: r["gate"].update(arbitrary_m=False), "arbitrary-M claim differs"),
    (lambda r: r["gate"].update(arbitrary_m=1), "arbitrary-M claim differs"),
    # The row frame is re-derived, so capacity and rows cannot be restated.
    (_set_plan("m_bound", 16), "row slot capacity differs"),
    (_set_plan("active_m", 9), "bounded rows differ"),
])
def test_tampered_bounded_record_is_refused(rocm_image_toolchain, edit, message):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    _, record = _bounded_record(fn, roots)
    with pytest.raises(ValueError, match=message):
        NativeQuantizedProductProgram.from_manifest(_tampered(record, edit))


def test_static_record_cannot_carry_arbitrary_m_products(rocm_image_toolchain):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    program, record = _bounded_record(fn, roots)
    gate = record["native_program"]["gate"]
    # Alone, a shape-free product image has no row policy.
    with pytest.raises(ValueError, match="arbitrary-M claim differs"):
        NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(gate)))
    with pytest.raises(ValueError, match="no standalone portable record"):
        program.native.gate.to_manifest()
    static = public()[0].compile_native_quantized_program(*roots).manifest()["native_program"]
    static["gate"]["arbitrary_m"] = True
    with pytest.raises(ValueError, match="arbitrary-M claim differs"):
        NativeQuantizedProductProgram.from_manifest(static)


@pytest.mark.parametrize("member", ["gate", "down"])
def test_rehashed_static_claim_on_a_shape_free_image_is_refused(rocm_image_toolchain, member):
    """A forger who also recomputes the digest still meets the image check:
    a static claim requires the image to be the checked static Target."""
    import dataclasses
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    _, record = _bounded_record(fn, roots)
    replay = NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(record["native_program"])))
    component = dataclasses.replace(getattr(replay, member), arbitrary_m=False)
    component = dataclasses.replace(component, binding_digest=component._identity())
    with pytest.raises(ValueError, match="checked Target differs"):
        component.validate()
