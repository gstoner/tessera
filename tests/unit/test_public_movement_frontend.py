"""Public movement frontend shapes, references and owning-device JIT proof."""
import os

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type
from tessera.compiler.trace import trace, to_graph_ir_module


def paged(pages, table):
    return ts.ops.kv_cache_read(pages, 1, 6, page_table=table)


def paged_default(table, pages):
    return ts.ops.kv_cache_read(pages, 3, page_table=table)


def dispatched(token, x):
    return ts.ops.moe_dispatch(x, token, transport=None)


def inputs(family, large=False):
    rng = np.random.default_rng(51005)
    if family.startswith("paged"):
        p, ps, h, d = (8, 8, 4, 32) if large else (4, 4, 3, 8)
        x = rng.normal(size=(p, ps, h, d)).astype(np.float32)
        table = np.array([2, 0, 2, 1], np.int32)
        if family == "paged_default":
            return (table, x), x[table].reshape(-1, h, d)[3:4]
        return (x, table), x[table].reshape(-1, h, d)[1:6]
    t, s, h = (64, 128, 256) if large else (7, 9, 13)
    x = rng.normal(size=(t, h)).astype(np.float32)
    token = rng.integers(0, t, s, dtype=np.int32)
    return (token, x), x[token]


@pytest.mark.parametrize("family", ["paged", "paged_default", "dispatched"])
def test_reference_and_tracing_preserve_tensor_shape_and_operand_roles(family):
    fn = globals()[family]
    args, expected = inputs(family)
    np.testing.assert_array_equal(fn(*args), expected)
    module = to_graph_ir_module(trace(fn, *args), target="rocm_gfx1151")
    assert str(module.functions[0].result_types[0]) == str(tensor_ir_type(expected.shape, "fp32"))
    assert len(module.functions[0].result_types) == 1
    op = module.functions[0].body[0]
    assert len(op.operands) == 2
    assert "page_table" not in op.kwargs
    if family == "paged_default":
        assert op.kwargs["end"] == 4


def test_catalog_infers_slots_and_preserves_opaque_cache_pair():
    result = _infer_result_types("tessera.moe_dispatch",
        [tensor_ir_type((7, 13), "fp32"), tensor_ir_type((9,), "int32")])
    assert str(result[0]) == "tensor<9x13xf32>"
    assert len(_infer_result_types("tessera.kv_cache.read",
        [tensor_ir_type()], {"start": 0, "end": 1})) == 2


@pytest.mark.parametrize("start,end", [(True, 3), (-1, 3), (3, 3), (3, 17), (1.5, 3)])
def test_paged_reference_rejects_invalid_bounds(start, end):
    (x, table), _ = inputs("paged")
    with pytest.raises(ValueError):
        ts.ops.kv_cache_read(x, start, end, page_table=table)


@pytest.mark.parametrize("bad", [-1, 4])
def test_paged_reference_rejects_invalid_physical_page(bad):
    (x, table), _ = inputs("paged")
    table[2] = bad
    with pytest.raises(ValueError, match="physical"):
        paged(x, table)


def test_dispatch_reference_rejects_invalid_slots_and_transport():
    (token, x), _ = inputs("dispatched")
    with pytest.raises(ValueError, match="transport"):
        ts.ops.moe_dispatch(x, token, transport="all_to_all")
    for bad in (-1, x.shape[0]):
        token[0] = bad
        with pytest.raises(ValueError, match="range"):
            ts.ops.moe_dispatch(x, token)
    with pytest.raises(ValueError, match="int32"):
        ts.ops.moe_dispatch(x, token.astype(np.int64))


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("arch,family", [
    ("gfx1151", "paged"), ("gfx1151", "paged_default"), ("gfx1151", "dispatched"),
    ("gfx1201", "paged"), ("gfx1201", "paged_default"),
])
def test_public_jit_movement_executes_and_specializes_on_owning_gpu(arch, family, monkeypatch):
    from tessera import runtime as rt
    if os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1":
        pytest.skip("explicit owning-device movement gate")
    if rt._rocm_live_arch() != arch:
        pytest.skip("different owning architecture")
    fn = ts.jit(target="rocm_" + arch, native_required=True)(globals()[family])
    def forbidden(*args, **kwargs):
        pytest.fail("native movement executed the eager reference")
    from tessera.autodiff.tape import _make_wrapper
    name = "kv_cache_read" if family.startswith("paged") else "moe_dispatch"
    monkeypatch.setattr(ts.ops, name, _make_wrapper(name, forbidden))
    seen = {}
    for iteration, large in enumerate((False, True, False, False)):
        args, expected = inputs(family, large)
        if iteration == 3:
            source = args[0] if family == "paged" else args[1]
            bits = source.view(np.uint32).reshape(-1)
            bits[::7] = 0x7FC12345  # Preserve NaN payloads, infinities and signed zero.
            bits[1::7] = 0x80000000
            bits[2::7] = 0x7F800000
            if family == "dispatched":
                expected = source[args[0]]
            else:
                table = args[1] if family == "paged" else args[0]
                logical = source[table].reshape(-1, source.shape[2], source.shape[3])
                expected = logical[1:6] if family == "paged" else logical[3:4]
        actual = fn(*args)
        artifact = fn.runtime_artifact()
        if large in seen:
            assert artifact is seen[large]
        seen[large] = artifact
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert fn.execution_kind == "native_gpu"
        assert fn.last_fallback_reason is None
        receipt = fn._native_descriptor_last_receipt
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu"
        bundle = fn.compile_bundle
        assert bundle.schedule.producer == "tessera-opt.tessera-graph-to-schedule"
        assert bundle.schedule.input_digest == bundle.graph.output_digest
        assert bundle.tile.input_digest == bundle.schedule.output_digest
        assert bundle.target_ir.input_digest == bundle.tile.output_digest
        assert bundle.backend.input_digest == bundle.target_ir.output_digest
    assert len(fn._native_descriptor_specializations) == 2
    assert len(fn._native_descriptor_artifacts) == 2
    rt._clear_rocm_native_image_cache()

@pytest.mark.hardware_rocm
@pytest.mark.parametrize("arch,family", [
    ("gfx1151", "paged"), ("gfx1151", "paged_default"), ("gfx1151", "dispatched"),
    ("gfx1201", "paged"), ("gfx1201", "paged_default"),
])
def test_prepared_native_binding_avoids_graph_serialization_and_rejects_views(arch, family, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.graph_ir import GraphIRModule
    if os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1":
        pytest.skip("explicit owning-device movement gate")
    if rt._rocm_live_arch() != arch:
        pytest.skip("different owning architecture")
    monkeypatch.setenv("TESSERA_ROCM_PREPARED_MOVEMENT", "1")
    lib = rt._load_rocm_native_movement_runtime()
    assert lib is not None and hasattr(lib, "tessera_rocm_movement_prepare")
    fn = ts.jit(target="rocm_" + arch, native_required=True)(globals()[family])
    args, expected = inputs(family)
    fn(*args)
    before = rt._rocm_native_movement_stats()
    def forbidden(*a, **kw):
        pytest.fail("prepared call serialized or compiled Graph IR")
    with monkeypatch.context() as guard:
        guard.setattr(GraphIRModule, "to_mlir", forbidden)
        actual = fn(*args)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert fn._native_descriptor_last_receipt["native_call_binding"] == "prepared_cpp_movement"
    after = rt._rocm_native_movement_stats()
    assert after["launches"] - before["launches"] == 1
    assert after["allocations"] == before["allocations"]
    source_index = 0 if family == "paged" else 1
    bad = list(args)
    # Same shape/dtype and byte count: metadata must reject a non-contiguous view.
    expanded = np.repeat(args[source_index], 2, axis=-1)
    bad[source_index] = expanded[..., ::2]
    before = rt._rocm_native_movement_stats()
    with pytest.raises(RuntimeError, match="prepared movement invocation"):
        fn(*bad)
    assert rt._rocm_native_movement_stats()["launches"] == before["launches"]
    assert fn._native_descriptor_last_receipt is None
    # A changed typed Graph cannot reuse the sealed prepared call.
    module, _ = fn._trace_frontend_capture(args, {})
    call = next(iter(fn._native_prepared_movement_calls.values()))
    module.functions[0].body[0].kwargs["unsupported_policy"] = True
    before = rt._rocm_native_movement_stats()
    try:
        with pytest.raises(RuntimeError, match="unsupported policy"):
            fn(*args)
        assert rt._rocm_native_movement_stats()["launches"] == before["launches"]
        assert fn._native_descriptor_last_receipt is None
    finally:
        module.functions[0].body[0].kwargs.pop("unsupported_policy")
    fn.close_native_storage()
    assert not call.matches(module)
    with pytest.raises(ValueError, match="closed"):
        call(args)
    np.testing.assert_array_equal(fn(*args).view(np.uint32), expected.view(np.uint32))
    fn.close_native_storage()
    rt._clear_rocm_native_image_cache()

@pytest.mark.parametrize("plan", [False, True])
def test_moe_reference_ad_matches_gather_scatter_adjoint_and_finite_difference(plan):
    from tessera.autodiff.jvp import jvp, jvp_moe_dispatch
    from tessera.autodiff.vjp import vjp_moe_dispatch
    from tessera.stdlib.moe import plan_dispatch
    rng = np.random.default_rng(105)
    x = rng.normal(size=(7, 5))
    tangent = rng.normal(size=x.shape)
    token = np.array([6, 0, 2, 6, 0, 6, 3, 3, 1], np.int32)
    if plan:
        ids = rng.integers(0, 3, (7, 2))
        route = plan_dispatch(ids, np.ones((7, 2)), 3)
        token = route.sort_perm // route.top_k
    else:
        route = token
    out, dy = jvp(lambda value: ts.ops.moe_dispatch(value, route), (x,), (tangent,))
    np.testing.assert_array_equal(out, x[token])
    np.testing.assert_array_equal(dy, tangent[token])
    cotangent = rng.normal(size=out.shape)
    dx, nondiff = vjp_moe_dispatch(cotangent, x, route)
    assert nondiff is None
    expected = np.zeros_like(x)
    for slot, index in enumerate(token):
        expected[index] += cotangent[slot]
    np.testing.assert_allclose(dx, expected, rtol=0, atol=0)
    np.testing.assert_allclose(np.sum(dy * cotangent), np.sum(tangent * dx), rtol=1e-13)
    step = 1e-5
    difference = (ts.ops.moe_dispatch(x + step*tangent, route)
                  - ts.ops.moe_dispatch(x - step*tangent, route)) / (2*step)
    np.testing.assert_allclose(dy, difference, rtol=1e-9, atol=1e-10)
    from tessera.autodiff import grad
    reverse = grad(lambda value: ts.ops.sum(ts.ops.moe_dispatch(value, route)))(x)
    expected_reverse = np.zeros_like(x)
    for index in token:
        expected_reverse[index] += 1
    np.testing.assert_array_equal(reverse, expected_reverse)
    _, zero = jvp_moe_dispatch((x, route), (None, None))
    np.testing.assert_array_equal(zero, np.zeros_like(out))
