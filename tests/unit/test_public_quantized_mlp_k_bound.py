"""Public explicit contraction-bound compilation of the native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-K-20261028. Packaging only: the owning-device replay
lives in tests/device/rocm/test_public_quantized_mlp_k_bound_device.py.
"""
import json

import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.benchmark_public_quantized_gate_up import make_gate_up
from tessera.compiler.rocm_quantized_product_frontend import _compiler_path, runtime_artifact

V6 = "tessera.native.quantized_product.v6"


def public(m=3, h=32, k=48, n=19):
    return ts.jit(target="rocm_gfx1201")(make_mlp(m, h, k, n)), inputs(m, h, k, n)


def _k_bounded(fn, roots, *, m_bound=8, k_bound=96):
    return fn.compile_native_quantized_program(*roots, m_bound=m_bound, k_bound=k_bound)


def test_explicit_k_bound_compiles_the_v6_program(rocm_image_toolchain):
    from tessera.compiler.rocm_nvfp4_expand_native import NVFP4_EXPAND_ABI, NVFP4_EXPAND_RUNTIME_K_ABI
    fn, roots = public()
    program = _k_bounded(fn, roots)
    plan = json.loads(program.native.program_json)
    assert program.bounded and program.native.k_bounded and plan["schema"] == V6
    assert (plan["active_m"], plan["m_bound"], plan["active_k"], plan["k_bound"]) == (3, 8, 48, 96)
    # Activation, both gate weight planes, the gate quantizer pair and the
    # gate expansion; the gate product (step 2) contracts K away.
    assert plan["k_frame"] == {"slots": [0, 1, 2, 7, 8, 9, 10], "k_steps": [0, 1, 2]}
    assert [plan["buffers"][s]["shape"] for s in (0, 1, 2, 7, 8, 9, 10)] == [
        [8, 96], [64, 48], [64, 6], [8, 96], [8, 6], [64, 96], [6, 64]]
    assert "tessera.native.quantized_k_bound" in plan["original_graph_ir"]
    assert "tessera.native.quantized_k_bound" not in plan["source_graph_ir"]
    native = program.native
    assert native.gate.runtime_k and not native.down.runtime_k
    assert native.gate.expansion.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_K_ABI
    assert native.down.expansion.native.descriptor.abi_id == NVFP4_EXPAND_ABI
    assert native.gate.product.descriptor.provenance["runtime_shape_image"] is True
    assert _compiler_path(native) == "canonical_rocm_quantized_k_bounded_mlp_program"
    assert program.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v6"
    assert fn._rocm_quantized_product_bounded_program is program
    assert fn.graph_ir.module_attrs["tessera.native.quantized_k_bound"] == "96 : i64"


def test_k_bound_alone_keeps_the_traced_rows_as_capacity(rocm_image_toolchain):
    fn, roots = public()
    program = fn.compile_native_quantized_program(*roots, k_bound=64)
    plan = json.loads(program.native.program_json)
    assert plan["schema"] == V6 and (plan["m_bound"], plan["k_bound"]) == (3, 64)


def test_row_bound_alone_keeps_the_v5_program(rocm_image_toolchain):
    fn, roots = public()
    program = fn.compile_native_quantized_program(*roots, m_bound=8)
    assert json.loads(program.native.program_json)["schema"] == "tessera.native.quantized_product.v5"
    assert not program.native.k_bounded and not program.native.gate.runtime_k


@pytest.mark.parametrize("bound", [0, -16, 24, 2.0, True, "64"])
def test_malformed_k_bound_is_refused_before_tracing(bound, monkeypatch):
    fn, roots = public()
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a malformed bound"))
    with pytest.raises(ValueError, match="positive multiple of 16"):
        fn.compile_native_quantized_program(*roots, m_bound=8, k_bound=bound)


def test_k_bound_below_the_traced_k_is_refused(rocm_image_toolchain):
    fn, roots = public()
    with pytest.raises(RuntimeError, match="exceeds its checked K16 capacity"):
        _k_bounded(fn, roots, k_bound=32)
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None
    # The request belonged to that compile only; the cached trace never saw it.
    assert [m.module_attrs.get("tessera.native.quantized_k_bound")
            for m in fn._traced_frontend_specializations.values()] == [None]


def test_k_bound_on_the_gate_up_program_is_refused(rocm_image_toolchain):
    from benchmarks.rocm.benchmark_public_quantized_gate_up import inputs as gate_inputs
    fn = ts.jit(target="rocm_gfx1201")(make_gate_up(3, 32, 48))
    with pytest.raises(RuntimeError, match="require the complete seven-member MLP"):
        fn.compile_native_quantized_program(*gate_inputs(3, 32, 48), m_bound=8, k_bound=96)


def test_k_bound_without_a_row_bound_is_refused_by_the_export(rocm_image_toolchain):
    from tests.unit.test_native_quantized_mlp_program import source
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    text = source(3, 32, 48, 19).replace(
        "tessera.native.quantized_product}",
        "tessera.native.quantized_product, tessera.native.quantized_k_bound = 96 : i64}")
    with pytest.raises(RuntimeError, match="bounded K requires an M bound"):
        package_quantized_product_graph(text)


def test_k_bounded_record_round_trips(rocm_image_toolchain):
    from tessera.compiler.rocm_quantized_product_frontend import _resolve
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    from tessera.runtime import RuntimeArtifact
    fn, roots = public()
    program = _k_bounded(fn, roots)
    native = program.manifest()["native_program"]
    assert native["schema"] == "tessera.rocm.quantized_product_package.v6"
    assert native["gate"]["runtime_k"] is True and "runtime_k" not in native["down"]
    replay = NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(native)))
    assert replay.k_bounded and replay.gate.runtime_k and not replay.down.runtime_k
    assert replay.binding_digest == program.native.binding_digest
    artifact = RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    assert artifact.metadata["compiler_path"] == "canonical_rocm_quantized_k_bounded_mlp_program"
    resolved = _resolve(json.dumps(artifact.metadata["native_program"], sort_keys=True, allow_nan=False),
                        artifact.graph_ir, json.dumps(artifact.metadata["arg_names"]))
    assert resolved.native.k_bounded


def _set_plan(key, value):
    def edit(record):
        plan = json.loads(record["program_json"])
        plan[key] = value
        record["program_json"] = json.dumps(plan, sort_keys=True)
    return edit


@pytest.mark.parametrize("edit,message", [
    # The package schema and the plan schema are bound to each other.
    (lambda r: r.update(schema="tessera.rocm.quantized_product_package.v5"), "schema/frame differs"),
    # A dropped or restated claim breaks the component digest that covers it.
    (lambda r: r["gate"].pop("runtime_k"), "member binding differs"),
    (lambda r: r["down"].update(runtime_k=True), "member binding differs"),
    (lambda r: r["gate"].update(runtime_k=False), "runtime-K claim differs"),
    (lambda r: r["gate"].update(runtime_k=1), "runtime-K claim differs"),
    # The contraction frame is re-derived, never restated.
    (_set_plan("k_bound", 192), "contraction slot capacity differs"),
    (_set_plan("active_k", 40), "bounded K differs"),
    (_set_plan("active_k", 112), "bounded K differs"),
    (_set_plan("k_frame", {"slots": [0, 7, 8], "k_steps": [0, 1, 2]}), "contraction frame differs"),
])
def test_tampered_k_bounded_record_is_refused(rocm_image_toolchain, edit, message):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    record = json.loads(json.dumps(_k_bounded(fn, roots).manifest()["native_program"]))
    edit(record)
    with pytest.raises(ValueError, match=message):
        NativeQuantizedProductProgram.from_manifest(record)


def test_runtime_k_claim_requires_its_expansion_image(rocm_image_toolchain):
    """A forger who rehashes the component still meets the image policy check."""
    import dataclasses
    fn, roots = public()
    gate = _k_bounded(fn, roots).native.gate
    forged = dataclasses.replace(gate, runtime_k=False)
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="runtime-K expansion policy differs"):
        forged.validate()
    static = public()[0].compile_native_quantized_program(*roots, m_bound=8).native.gate
    forged = dataclasses.replace(static, runtime_k=True)
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="runtime-K expansion policy differs"):
        forged.validate()


def _expansion(n=64, k=48, *, runtime_k):
    from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph, package_nvfp4_expansion
    graph = build_nvfp4_expansion_graph(n, k, [0, n // 2, n], numeric_policy=nvfp4_expansion_policy())
    return package_nvfp4_expansion(graph.to_mlir(target="rocm_gfx1201", canonical=True), runtime_k=runtime_k)


def test_runtime_k_expansion_is_the_projection_of_its_checked_target(rocm_image_toolchain):
    import dataclasses
    from tessera.compiler.native_artifact import ScalarArgument
    package = _expansion(runtime_k=True)
    desc = package.native.descriptor
    assert package.runtime_k and package.contract()[1:4] == (64, 48, (0, 32, 64))
    assert desc.scalars == (ScalarArgument(5, "groups", "int64"),)
    assert desc.geometry.grid == (1, 64, 1) and desc.provenance["runtime_k"] is True
    assert "k = 0 : i64" in package.native.target_ir and "runtime_k" in package.native.target_ir
    # The static image keeps its flat grid and no scalar.
    static = _expansion(runtime_k=False)
    assert not static.runtime_k and static.native.descriptor.scalars == ()
    assert static.native.descriptor.geometry.grid == ((64 * 3 + 255) // 256, 1, 1)
    # A runtime-K image cannot be relabelled static, nor carry another Target.
    swapped = dataclasses.replace(package, native=dataclasses.replace(
        package.native, descriptor=dataclasses.replace(desc, abi_id=static.native.descriptor.abi_id)))
    with pytest.raises(ValueError):
        swapped.validate()
    other = dataclasses.replace(package, native=dataclasses.replace(
        package.native, target_ir=package.native.target_ir.replace("n = 64", "n = 32")))
    with pytest.raises(ValueError):
        other.validate()


def test_kernel_identity_projects_only_a_static_expansion(rocm_image_toolchain):
    from tessera.compiler.rocm_native import _shape_free_target_ir
    from tessera.compiler.rocm_nvfp4_expand_native import _checked_target
    package = _expansion(runtime_k=True)
    checked = _checked_target(package.native.tile_ir)
    with pytest.raises(ValueError, match="only its runtime-K or runtime-N projection"):
        _shape_free_target_ir(checked, family="nvfp4_expand", directive="tessera_rocm.nvfp4_expand")
    # The projection is not idempotent by design: a runtime-K image has no static K.
    with pytest.raises(RuntimeError, match="static verified expansion and exactly one of runtime-k"):
        _shape_free_target_ir(package.native.target_ir, family="nvfp4_expand",
                              directive="tessera_rocm.nvfp4_expand", runtime_k=True)
    # The verifier refuses a runtime-K directive that still records a K.
    with pytest.raises(RuntimeError, match="records K as zero"):
        _shape_free_target_ir(package.native.target_ir.replace("k = 0 : i64", "k = 48 : i64"),
                              family="nvfp4_expand", directive="tessera_rocm.nvfp4_expand", runtime_k=True)
