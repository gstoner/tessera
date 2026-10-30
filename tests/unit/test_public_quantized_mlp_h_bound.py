"""Public explicit hidden-width-bound compilation of the native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-H-20261029. Packaging only: the owning-device replay
lives in tests/device/rocm/test_public_quantized_mlp_h_bound_device.py.
"""
import json

import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera.compiler.rocm_quantized_product_frontend import _compiler_path, runtime_artifact

V8 = "tessera.native.quantized_product.v8"
H_SLOTS = [1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16]


def public(m=3, h=32, k=48, n=19):
    return ts.jit(target="rocm_gfx1201")(make_mlp(m, h, k, n)), inputs(m, h, k, n)


def _h_bounded(fn, roots, **bounds):
    return fn.compile_native_quantized_program(*roots, **{"m_bound": 8, "h_bound": 64, **bounds})


def test_explicit_h_bound_compiles_the_v8_program(rocm_image_toolchain):
    from tessera.compiler.rocm_nvfp4_expand_native import (
        NVFP4_EXPAND_RUNTIME_K_ABI, NVFP4_EXPAND_RUNTIME_N_ABI)
    from tessera.compiler.rocm_parent_view_native import PARENT_VIEW_BF16_RUNTIME_WIDTH_ABI
    fn, roots = public()
    program = _h_bounded(fn, roots)
    plan = json.loads(program.native.program_json)
    native = program.native
    assert plan["schema"] == V8 and native.h_bounded
    assert not native.k_bounded and not native.n_bounded
    assert (plan["active_h"], plan["h_bound"]) == (32, 64)
    # Both weight-plane pairs, both expansions' B/SB, the parent, the SiLU
    # output and the down quantizer pair; the original activation and the
    # output never depend on H.
    assert plan["h_frame"] == {"slots": H_SLOTS, "h_steps": [1, 2, 3, 4, 5, 6]}
    assert [plan["buffers"][s]["shape"] for s in H_SLOTS] == [
        [128, 24], [128, 3], [19, 32], [19, 4], [128, 48], [3, 128],
        [8, 128], [8, 64], [8, 64], [8, 4], [19, 64], [4, 19]]
    assert "tessera.native.quantized_h_bound" in plan["original_graph_ir"]
    assert "tessera.native.quantized_h_bound" not in plan["source_graph_ir"]
    # The gate/up weights expand through the two-projection runtime-N image
    # and the SiLU reads its runtime width; the down projection contracts H.
    assert native.gate.runtime_n and not native.gate.runtime_k
    assert native.down.runtime_k and not native.down.runtime_n
    assert native.gate.expansion.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_N_ABI
    assert native.gate.expansion.contract()[3] == (0, 64, 128)
    assert native.down.expansion.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_K_ABI
    assert native.gate.activation.descriptor.abi_id == PARENT_VIEW_BF16_RUNTIME_WIDTH_ABI
    for component in (native.gate, native.down):
        assert component.product.descriptor.provenance["runtime_shape_image"] is True
    assert _compiler_path(native) == "canonical_rocm_quantized_h_bounded_mlp_program"
    assert program.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v8"
    assert fn.graph_ir.module_attrs["tessera.native.quantized_h_bound"] == "64 : i64"


def test_runtime_width_silu_is_marked_and_reads_its_width(rocm_image_toolchain):
    from tessera.compiler.rocm_parent_view_native import _frame
    fn, roots = public()
    gate = _h_bounded(fn, roots).native.gate
    assert "tessera.native.gate_up_runtime_width" in gate.activation_recipe.graph_ir
    assert "runtime_width" in gate.activation.target_ir
    _, member = _frame(gate.activation.target_ir)
    assert member["profile"] == "gfx1201_parent_view_bf16_f32_runtime_width_v1"
    # The member binds (count, width) at its checked capacity.
    assert member["scalars"] == [8 * 64, 64] and member["output_shape"] == [8, 64]


def test_h_bound_alone_keeps_the_traced_rows_as_capacity(rocm_image_toolchain):
    fn, roots = public()
    plan = json.loads(fn.compile_native_quantized_program(*roots, h_bound=48).native.program_json)
    assert plan["schema"] == V8 and (plan["m_bound"], plan["h_bound"]) == (3, 48)


@pytest.mark.parametrize("bound", [0, -16, 24, 32.0, True, "64"])
def test_malformed_h_bound_is_refused_before_tracing(bound, monkeypatch):
    fn, roots = public()
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a malformed bound"))
    with pytest.raises(ValueError, match="hidden bound must be a positive multiple of 16"):
        fn.compile_native_quantized_program(*roots, m_bound=8, h_bound=bound)


@pytest.mark.parametrize("other", [{"k_bound": 96}, {"n_bound": 40}])
def test_h_bound_with_k_or_n_is_refused_before_tracing(other, monkeypatch):
    fn, roots = public()
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a refused combination"))
    with pytest.raises(ValueError, match="does not compose with a K or N bound"):
        fn.compile_native_quantized_program(*roots, m_bound=8, h_bound=64, **other)


def test_h_bound_below_the_traced_h_is_refused(rocm_image_toolchain):
    fn, roots = public()
    with pytest.raises(RuntimeError, match="exceeds its checked H16 capacity"):
        _h_bounded(fn, roots, h_bound=16)
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None
    assert [m.module_attrs.get("tessera.native.quantized_h_bound")
            for m in fn._traced_frontend_specializations.values()] == [None]


@pytest.mark.parametrize("attrs,message", [
    ("tessera.native.quantized_h_bound = 64 : i64", "bounded H requires an M bound"),
    ("tessera.native.quantized_m_bound = 8 : i64, tessera.native.quantized_h_bound = 64 : i64, "
     "tessera.native.quantized_n_bound = 40 : i64", "does not compose with a K or N bound"),
])
def test_h_bound_export_refusals(rocm_image_toolchain, attrs, message):
    from tests.unit.test_native_quantized_mlp_program import source
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    text = source(3, 32, 48, 19).replace(
        "tessera.native.quantized_product}", f"tessera.native.quantized_product, {attrs}}}")
    with pytest.raises(RuntimeError, match=message):
        package_quantized_product_graph(text)


def test_h_bounded_record_round_trips(rocm_image_toolchain):
    from tessera.compiler.rocm_quantized_product_frontend import _resolve
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    from tessera.runtime import RuntimeArtifact
    fn, roots = public()
    program = _h_bounded(fn, roots)
    native = program.manifest()["native_program"]
    assert native["schema"] == "tessera.rocm.quantized_product_package.v8"
    assert native["gate"]["runtime_n"] is True and "runtime_k" not in native["gate"]
    assert native["down"]["runtime_k"] is True and "runtime_n" not in native["down"]
    replay = NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(native)))
    assert replay.h_bounded and replay.gate.runtime_n and replay.down.runtime_k
    assert replay.binding_digest == program.native.binding_digest
    artifact = RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    assert artifact.metadata["compiler_path"] == "canonical_rocm_quantized_h_bounded_mlp_program"
    resolved = _resolve(json.dumps(artifact.metadata["native_program"], sort_keys=True, allow_nan=False),
                        artifact.graph_ir, json.dumps(artifact.metadata["arg_names"]))
    assert resolved.native.h_bounded


def _set_plan(key, value):
    def edit(record):
        plan = json.loads(record["program_json"])
        plan[key] = value
        record["program_json"] = json.dumps(plan, sort_keys=True)
    return edit


def _drop_plan(key):
    def edit(record):
        plan = json.loads(record["program_json"])
        plan.pop(key)
        record["program_json"] = json.dumps(plan, sort_keys=True)
    return edit


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(schema="tessera.rocm.quantized_product_package.v5"), "schema/frame differs"),
    (lambda r: r["gate"].pop("runtime_n"), "member binding differs"),
    (lambda r: r["down"].pop("runtime_k"), "member binding differs"),
    (lambda r: r["down"].update(runtime_n=True), "member binding differs"),
    (lambda r: r["gate"].update(runtime_n=False), "runtime-N claim differs"),
    (lambda r: r["down"].update(runtime_k=1), "runtime-K claim differs"),
    (_set_plan("h_bound", 80), "hidden slot capacity differs"),
    (_set_plan("active_h", 40), "bounded H differs"),
    (_set_plan("active_h", 80), "bounded H differs"),
    (_set_plan("h_frame", {"slots": H_SLOTS[4:], "h_steps": [1, 2, 3, 4, 5, 6]}), "hidden frame differs"),
    (_set_plan("h_frame", {"slots": H_SLOTS, "h_steps": [1, 2, 3, 4, 5]}), "hidden frame differs"),
    # The hidden keys are required together; a partial frame is not a v8 plan.
    (_drop_plan("h_frame"), "component encoding differs"),
])
def test_tampered_h_bounded_record_is_refused(rocm_image_toolchain, edit, message):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    record = json.loads(json.dumps(_h_bounded(fn, roots).manifest()["native_program"]))
    edit(record)
    with pytest.raises(ValueError, match=message):
        NativeQuantizedProductProgram.from_manifest(record)


def test_gate_runtime_n_requires_the_runtime_width_silu(rocm_image_toolchain):
    """A runtime-N gate/up product binds only the runtime-width parent view."""
    import dataclasses
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    fn, roots = public()
    gate = _h_bounded(fn, roots).native.gate
    static = package_quantized_product_graph(gate.graph_ir)
    forged = dataclasses.replace(gate, activation_recipe=static.activation_recipe,
                                 activation=static.activation)
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="projected Graph differs|width policy differs"):
        forged.validate()


def test_runtime_n_expansion_takes_equally_spaced_projections(rocm_image_toolchain):
    from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
    from tessera.compiler.native_artifact import ScalarArgument
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph, package_nvfp4_expansion
    graph = build_nvfp4_expansion_graph(64, 32, [0, 32, 64], numeric_policy=nvfp4_expansion_policy())
    package = package_nvfp4_expansion(graph.to_mlir(target="rocm_gfx1201", canonical=True), runtime_n=True)
    assert package.runtime_n and package.contract()[1:4] == (64, 32, (0, 32, 64))
    assert package.native.descriptor.scalars == (ScalarArgument(5, "n", "int64"),)
    assert "n = 0 : i64" in package.native.target_ir


def test_runtime_width_marker_is_a_single_request():
    from tessera.compiler.rocm_parent_view_native import runtime_width_graph
    text = 'module attributes {tessera.native.gate_up_view, tessera.target = "rocm"} {}'
    marked = runtime_width_graph(text)
    assert "tessera.native.gate_up_runtime_width, tessera.native.gate_up_view" in marked
    for bad in (marked, "module {}", text.replace("}", ", tessera.native.gate_up_view}", 1)):
        with pytest.raises(ValueError, match="one unmarked native gate/up Graph"):
            runtime_width_graph(bad)


def test_shape_free_product_panel_is_chosen_for_the_admitted_rows(rocm_image_toolchain):
    """At capacity 256 x 2*512 the static rule takes the 32-row panel; a
    shape-free product runs M down to 1 and takes the 16-row panel."""
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    from tests.unit.test_native_quantized_mlp_program import source
    fn, roots = public(16, 128, 4096, 256)
    bounded = fn.compile_native_quantized_program(*roots, m_bound=256, h_bound=512).native
    static = package_quantized_product_graph(source(256, 512, 4096, 256))
    tile = lambda program: program.native_stages[2].descriptor.provenance["macro_tile"]
    assert tile(static) == [32, 32] and tile(bounded) == [16, 32]
    assert "tessera.rocm.blockscale_min_m = 1 : i64" in bounded.gate.product_recipe.graph_ir
    assert "blockscale_min_m" not in static.gate.product_recipe.graph_ir


def test_admitted_rows_oracle_and_request():
    from tessera.compiler.rocm_fp8_blockscale import (
        BlockScalePanel, BlockScaleShape, admitted_rows_graph, blockscale_panel_oracle)
    shape = BlockScaleShape(256, 1024, 4096, 16, 1, "nk")
    assert blockscale_panel_oracle(shape) == BlockScalePanel("global", 32, 32, 1)
    assert blockscale_panel_oracle(shape, min_m=1) == BlockScalePanel("global", 16, 32, 1)
    assert blockscale_panel_oracle(shape, min_m=256) == BlockScalePanel("global", 32, 32, 1)
    text = 'module attributes {tessera.target = "rocm"} {}'
    assert admitted_rows_graph(text).startswith("module attributes {tessera.rocm.blockscale_min_m = 1 : i64, ")
    for bad, min_m in ((admitted_rows_graph(text), 1), ("module {}", 1), (text, 0), (text, True)):
        with pytest.raises(ValueError, match="admitted-rows request"):
            admitted_rows_graph(bad, min_m)


@pytest.mark.parametrize("value", ["0 : i64", "300 : i64", "1 : i32", '"1"'])
def test_native_schedule_refuses_an_out_of_range_admitted_minimum(rocm_image_toolchain, value):
    from tessera.compiler.rocm_quantized_product_program import _export
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
    from tests.unit.test_native_quantized_mlp_program import source
    tool = find_tessera_opt()
    product = _export(tool, source(256, 512, 4096, 256))["member_graphs"][2]
    marked = product.replace("module attributes {", f"module attributes {{tessera.rocm.blockscale_min_m = {value}, ", 1)
    with pytest.raises(RuntimeError, match="blockscale_min_m must be an i64 in"):
        run_tessera_opt(tool, marked, "--tessera-graph-to-schedule")


@pytest.mark.parametrize("bounds,traced", [
    ({"m_bound": 256, "h_bound": 2048}, (16, 512, 4096, 256)),
    ({"m_bound": 256, "k_bound": 8192}, (16, 2048, 4096, 256)),
])
def test_runtime_shape_products_request_the_global_panel(rocm_image_toolchain, bounds, traced):
    """GFX1201-BOUNDED-GLOBAL-PANEL-20261030: at capacity 256 x 4096 the static
    rule takes an LDS gate panel, which no runtime-K/N image has; the request
    selects the global register panel instead of refusing the program."""
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    from tests.unit.test_native_quantized_mlp_program import source
    fn = ts.jit(target="rocm_gfx1201")(make_mlp(*traced))
    native = fn.compile_native_quantized_program(*inputs(*traced), **bounds).native
    gate = native.gate.product
    assert gate.descriptor.provenance["macro_tile"] == [16, 32]
    assert gate.descriptor.provenance["runtime_shape_image"] is True
    assert 'tessera.rocm.blockscale_staging = "global"' in native.gate.product_recipe.graph_ir
    static = package_quantized_product_graph(source(256, 2048, 4096, 256))
    assert static.native_stages[2].descriptor.provenance["macro_tile"] == [128, 128]


def test_m_only_bounded_products_keep_their_lds_panel(rocm_image_toolchain):
    fn, roots = public(256, 2048, 4096, 256)
    native = fn.compile_native_quantized_program(*roots, m_bound=256).native
    assert native.gate.product.descriptor.provenance["macro_tile"] == [128, 128]
    assert "blockscale_staging" not in native.gate.product_recipe.graph_ir


def test_global_staging_oracle_and_request():
    from tessera.compiler.rocm_fp8_blockscale import (
        BlockScalePanel, BlockScaleShape, admitted_rows_graph, blockscale_panel_oracle)
    shape = BlockScaleShape(256, 4096, 4096, 16, 1, "nk")
    assert blockscale_panel_oracle(shape) == BlockScalePanel("lds", 128, 128, 8)
    assert blockscale_panel_oracle(shape, min_m=1, staging="global") == BlockScalePanel("global", 16, 32, 1)
    with pytest.raises(ValueError, match="only"):
        blockscale_panel_oracle(shape, staging="lds")
    text = 'module attributes {tessera.target = "rocm"} {}'
    assert 'blockscale_staging = "global", ' in admitted_rows_graph(text, global_staging=True)
    with pytest.raises(ValueError, match="admitted-rows request"):
        admitted_rows_graph(admitted_rows_graph(text, global_staging=True), global_staging=True)


@pytest.mark.parametrize("value", ['"lds"', "1 : i64"])
def test_native_schedule_refuses_another_staging_request(rocm_image_toolchain, value):
    from tessera.compiler.rocm_quantized_product_program import _export
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
    from tests.unit.test_native_quantized_mlp_program import source
    tool = find_tessera_opt()
    product = _export(tool, source(256, 512, 4096, 256))["member_graphs"][2]
    marked = product.replace("module attributes {", f"module attributes {{tessera.rocm.blockscale_staging = {value}, ", 1)
    with pytest.raises(RuntimeError, match="blockscale_staging must be the string"):
        run_tessera_opt(tool, marked, "--tessera-graph-to-schedule")
