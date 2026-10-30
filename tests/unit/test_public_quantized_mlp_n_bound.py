"""Public explicit column-bound compilation of the native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-N-20261029. Packaging only: the owning-device replay
lives in tests/device/rocm/test_public_quantized_mlp_n_bound_device.py.
"""
import json

import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera.compiler.rocm_quantized_product_frontend import _compiler_path, runtime_artifact

V7 = "tessera.native.quantized_product.v7"


def public(m=3, h=32, k=48, n=19):
    return ts.jit(target="rocm_gfx1201")(make_mlp(m, h, k, n)), inputs(m, h, k, n)


def _n_bounded(fn, roots, **bounds):
    return fn.compile_native_quantized_program(*roots, **{"m_bound": 8, "n_bound": 40, **bounds})


def test_explicit_n_bound_compiles_the_v7_program(rocm_image_toolchain):
    from tessera.compiler.rocm_nvfp4_expand_native import NVFP4_EXPAND_ABI, NVFP4_EXPAND_RUNTIME_N_ABI
    fn, roots = public()
    program = _n_bounded(fn, roots)
    plan = json.loads(program.native.program_json)
    assert plan["schema"] == V7 and program.native.n_bounded and not program.native.k_bounded
    assert (plan["active_n"], plan["n_bound"]) == (19, 40)
    # Both down weight planes, the down expansion's B/SB and the output.
    assert plan["n_frame"] == {"slots": [4, 5, 15, 16, 17], "n_steps": [5, 6]}
    assert [plan["buffers"][s]["shape"] for s in (4, 5, 15, 16, 17)] == [
        [40, 16], [40, 2], [40, 32], [2, 40], [8, 40]]
    assert "tessera.native.quantized_n_bound" in plan["original_graph_ir"]
    assert "tessera.native.quantized_n_bound" not in plan["source_graph_ir"]
    native = program.native
    assert native.down.runtime_n and not native.gate.runtime_n and not native.gate.runtime_k
    assert native.down.expansion.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_N_ABI
    assert native.gate.expansion.native.descriptor.abi_id == NVFP4_EXPAND_ABI
    assert native.down.product.descriptor.provenance["runtime_shape_image"] is True
    assert _compiler_path(native) == "canonical_rocm_quantized_n_bounded_mlp_program"
    assert program.manifest()["schema"] == "tessera.rocm.quantized_product_frontend.v7"
    assert fn.graph_ir.module_attrs["tessera.native.quantized_n_bound"] == "40 : i64"


def test_k_and_n_bounds_share_the_v7_program(rocm_image_toolchain):
    fn, roots = public()
    program = _n_bounded(fn, roots, k_bound=96)
    plan = json.loads(program.native.program_json)
    assert plan["schema"] == V7 and program.native.k_bounded and program.native.n_bounded
    assert plan["k_frame"] == {"slots": [0, 1, 2, 7, 8, 9, 10], "k_steps": [0, 1, 2]}
    assert program.native.gate.runtime_k and program.native.down.runtime_n


def test_n_bound_alone_keeps_the_traced_rows_as_capacity(rocm_image_toolchain):
    fn, roots = public()
    plan = json.loads(fn.compile_native_quantized_program(*roots, n_bound=24).native.program_json)
    assert plan["schema"] == V7 and (plan["m_bound"], plan["n_bound"]) == (3, 24)


@pytest.mark.parametrize("bound", [0, -1, 2.0, True, "40"])
def test_malformed_n_bound_is_refused_before_tracing(bound, monkeypatch):
    fn, roots = public()
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a malformed bound"))
    with pytest.raises(ValueError, match="column bound must be a positive integer"):
        fn.compile_native_quantized_program(*roots, m_bound=8, n_bound=bound)


def test_n_bound_below_the_traced_n_is_refused(rocm_image_toolchain):
    fn, roots = public()
    with pytest.raises(RuntimeError, match="exceeds its checked column capacity"):
        _n_bounded(fn, roots, n_bound=16)
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None
    assert [m.module_attrs.get("tessera.native.quantized_n_bound")
            for m in fn._traced_frontend_specializations.values()] == [None]


def test_n_bound_without_a_row_bound_is_refused_by_the_export(rocm_image_toolchain):
    from tests.unit.test_native_quantized_mlp_program import source
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    text = source(3, 32, 48, 19).replace(
        "tessera.native.quantized_product}",
        "tessera.native.quantized_product, tessera.native.quantized_n_bound = 40 : i64}")
    with pytest.raises(RuntimeError, match="bounded N requires an M bound"):
        package_quantized_product_graph(text)


@pytest.mark.parametrize("bounds", [{}, {"k_bound": 96}])
def test_n_bounded_record_round_trips(rocm_image_toolchain, bounds):
    from tessera.compiler.rocm_quantized_product_frontend import _resolve
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    from tessera.runtime import RuntimeArtifact
    fn, roots = public()
    program = _n_bounded(fn, roots, **bounds)
    native = program.manifest()["native_program"]
    assert native["schema"] == "tessera.rocm.quantized_product_package.v7"
    assert native["down"]["runtime_n"] is True and "runtime_n" not in native["gate"]
    assert ("runtime_k" in native["gate"]) == bool(bounds)
    replay = NativeQuantizedProductProgram.from_manifest(json.loads(json.dumps(native)))
    assert replay.n_bounded and replay.down.runtime_n
    assert replay.binding_digest == program.native.binding_digest
    artifact = RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    assert artifact.metadata["compiler_path"] == "canonical_rocm_quantized_n_bounded_mlp_program"
    resolved = _resolve(json.dumps(artifact.metadata["native_program"], sort_keys=True, allow_nan=False),
                        artifact.graph_ir, json.dumps(artifact.metadata["arg_names"]))
    assert resolved.native.n_bounded


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
    (lambda r: r["down"].pop("runtime_n"), "member binding differs"),
    (lambda r: r["gate"].update(runtime_n=True), "member binding differs"),
    (lambda r: r["down"].update(runtime_n=False), "runtime-N claim differs"),
    (lambda r: r["down"].update(runtime_n=1), "runtime-N claim differs"),
    (_set_plan("n_bound", 48), "column slot capacity differs"),
    (_set_plan("active_n", 41), "bounded N differs"),
    (_set_plan("active_n", 0), "bounded N differs"),
    (_set_plan("n_frame", {"slots": [15, 16, 17], "n_steps": [5, 6]}), "column frame differs"),
    # The column keys are required together; a partial frame is not a v7 plan.
    (_drop_plan("n_frame"), "component encoding differs"),
])
def test_tampered_n_bounded_record_is_refused(rocm_image_toolchain, edit, message):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    fn, roots = public()
    record = json.loads(json.dumps(_n_bounded(fn, roots).manifest()["native_program"]))
    edit(record)
    with pytest.raises(ValueError, match=message):
        NativeQuantizedProductProgram.from_manifest(record)


def test_runtime_n_claim_requires_its_expansion_image(rocm_image_toolchain):
    import dataclasses
    fn, roots = public()
    down = _n_bounded(fn, roots).native.down
    forged = dataclasses.replace(down, runtime_n=False)
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="runtime-N expansion policy differs"):
        forged.validate()


def _expansion(n, k, offsets, **policy):
    from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph, package_nvfp4_expansion
    graph = build_nvfp4_expansion_graph(n, k, offsets, numeric_policy=nvfp4_expansion_policy())
    return package_nvfp4_expansion(graph.to_mlir(target="rocm_gfx1201", canonical=True), **policy)


def test_runtime_n_expansion_is_the_projection_of_its_checked_target(rocm_image_toolchain):
    import dataclasses
    from tessera.compiler.native_artifact import ScalarArgument
    package = _expansion(40, 32, [0, 40], runtime_n=True)
    desc = package.native.descriptor
    assert package.runtime_n and not package.runtime_k and package.contract()[1:4] == (40, 32, (0, 40))
    assert desc.scalars == (ScalarArgument(5, "n", "int64"),)
    assert desc.geometry.grid == ((40 * 2 + 255) // 256, 1, 1) and desc.provenance["runtime_n"] is True
    assert "n = 0 : i64" in package.native.target_ir and "runtime_n" in package.native.target_ir
    other = dataclasses.replace(package, native=dataclasses.replace(
        package.native, target_ir=package.native.target_ir.replace("k = 32", "k = 16")))
    with pytest.raises(ValueError):
        other.validate()


def test_runtime_n_requires_equally_spaced_projections(rocm_image_toolchain):
    # The image derives every runtime boundary as p*N/P from its one N scalar.
    with pytest.raises(ValueError, match="equally spaced projections"):
        _expansion(64, 32, [0, 16, 64], runtime_n=True)
    with pytest.raises(ValueError, match="at most one runtime extent"):
        _expansion(40, 32, [0, 40], runtime_n=True, runtime_k=True)


def test_kernel_identity_runtime_n_refusals(rocm_image_toolchain):
    from tessera.compiler.rocm_native import _shape_free_target_ir
    from tessera.compiler.rocm_nvfp4_expand_native import _checked_target
    package = _expansion(40, 32, [0, 40], runtime_n=True)
    checked = _checked_target(package.native.tile_ir)
    with pytest.raises(ValueError, match="runtime-K or runtime-N projection"):
        _shape_free_target_ir(checked, family="nvfp4_expand", directive="tessera_rocm.nvfp4_expand",
                              runtime_k=True, runtime_n=True)
    with pytest.raises(ValueError, match="NVFP4 expansion family"):
        _shape_free_target_ir(checked, family="scaled_matmul", directive="tessera_rocm.scaled_wmma_gemm",
                              runtime_n=True)
    with pytest.raises(RuntimeError, match="records N as zero over equally spaced projections"):
        _shape_free_target_ir(package.native.target_ir.replace("n = 0 : i64", "n = 40 : i64"),
                              family="nvfp4_expand", directive="tessera_rocm.nvfp4_expand", runtime_k=True)
