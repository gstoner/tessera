"""MXFP4 source for the native FP4 expansion and the seven-member MLP.

ROCM-MXFP4-NATIVE-MLP-20261102. Host and packaging checks only; the owning
gfx1201 execution lives in tests/device/rocm/test_native_mxfp4_mlp.py.
"""
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.nvfp4_expansion import (
    checked_expansion_inputs, expansion_scale_group, expansion_source, mxfp4_expansion_policy,
    nvfp4_expansion_policy, reference_nvfp4_expand,
)


def _mx_weights(n=48, k=64, seed=3):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    exponents = rng.integers(100, 150, size=(k // 32, n), dtype=np.uint8)
    return codes, exponents


def test_policies_name_their_source_and_scale_group():
    assert expansion_source(nvfp4_expansion_policy()) == "nvfp4"
    assert expansion_source(mxfp4_expansion_policy()) == "mxfp4"
    assert expansion_scale_group(nvfp4_expansion_policy()) == 16
    assert expansion_scale_group(mxfp4_expansion_policy()) == 32
    mixed = dict(mxfp4_expansion_policy(), source_scale_application="projection_global_times_e4m3")
    with pytest.raises(ValueError, match="numeric policy"):
        expansion_source(mixed)


@pytest.mark.parametrize("globals_", [(1.0, 1.0), (0.375, 3.0)])
def test_mxfp4_reference_expansion_is_the_exact_decode(globals_):
    codes, exponents = _mx_weights()
    values, plane = reference_nvfp4_expand(
        mx.pack_e2m1_codes(codes), np.ascontiguousarray(exponents.T), np.array(globals_),
        row_offsets=[0, 24, 48], numeric_policy=mxfp4_expansion_policy())
    assert values.dtype == np.dtype(ml_dtypes.float8_e4m3fn) and values.shape == (48, 64)
    assert plane.dtype == np.float32 and plane.shape == (4, 48)
    # Both K16 groups of a K32 block carry the block's one rounded scale.
    np.testing.assert_array_equal(plane[0::2], plane[1::2])
    row_global = np.repeat(np.array(globals_), 24)[:, None]
    decoded = values.astype(np.float64) * np.repeat(plane.T.astype(np.float64), 16, axis=1)
    np.testing.assert_array_equal(decoded, mx.exact_weights(codes, exponents, dtype=np.float64) * row_global)


def test_mxfp4_boundary_refuses_reserved_code_shape_and_range():
    codes, exponents = _mx_weights()
    packed, scales = mx.pack_e2m1_codes(codes), np.ascontiguousarray(exponents.T)
    check = lambda s, g=np.ones(2), p=None: checked_expansion_inputs(
        packed, s, g, row_offsets=[0, 24, 48], numeric_policy=p or mxfp4_expansion_policy())
    check(scales)
    reserved = scales.copy()
    reserved[3, 1] = 255
    with pytest.raises(ValueError, match="255"):
        check(reserved)
    with pytest.raises(ValueError, match=r"\[N,K/32\]"):
        check(np.ascontiguousarray(np.repeat(scales, 2, axis=1)))
    with pytest.raises(ValueError, match=r"\[N,K/32\]"):
        check(scales.view(ml_dtypes.float8_e4m3fn))
    with pytest.raises(ValueError, match="overflows or underflows"):
        check(np.full_like(scales, 254), np.full(2, 1e30))
    with pytest.raises(ValueError, match="E4M3"):
        check(scales, p=nvfp4_expansion_policy())
    with pytest.raises(ValueError, match="K32"):
        checked_expansion_inputs(packed[:, :8], scales[:, :1], np.ones(2), row_offsets=[0, 24, 48],
                                 numeric_policy=mxfp4_expansion_policy())


def test_mxfp4_graph_requires_k32_and_byte_scales():
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph
    graph = build_nvfp4_expansion_graph(48, 64, [0, 24, 48], numeric_policy=mxfp4_expansion_policy())
    text = graph.to_mlir(target="rocm_gfx1201", canonical=True)
    assert "tensor<48x2xui8>" in text and "mxfp4_e2m1_e8m0_k32" in text
    with pytest.raises(ValueError, match="N/K16"):
        build_nvfp4_expansion_graph(48, 48, [0, 24, 48], numeric_policy=mxfp4_expansion_policy())


@pytest.mark.compiler_route
@pytest.mark.parametrize("mutation,match", [
    (("tensor<48x2xui8>", "tensor<48x2xf8E4M3FN>"), "E8M0 bytes"),
    (('"mxfp4_e2m1_e8m0_k32"', '"nvfp4_e2m1_e4m3_k16"'), "conflicting numeric policy field"),
])
def test_mxfp4_graph_verifier_fails_closed(production_compiler, mutation, match):
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph
    text = build_nvfp4_expansion_graph(
        48, 64, [0, 24, 48], numeric_policy=mxfp4_expansion_policy()).to_mlir(
            target="rocm_gfx1201", canonical=True)
    assert text.count(mutation[0]) >= 1
    run = subprocess.run([str(production_compiler), "--tessera-graph-to-schedule"],
                         input=text.replace(*mutation), text=True, capture_output=True)
    assert run.returncode != 0 and match in run.stderr


@pytest.mark.compiler_route
def test_mxfp4_schedule_records_e8m0_scale_storage(production_compiler):
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    text = build_nvfp4_expansion_graph(
        48, 64, [0, 24, 48], numeric_policy=mxfp4_expansion_policy()).to_mlir(
            target="rocm_gfx1201", canonical=True)
    tile = run_tessera_opt(production_compiler, run_tessera_opt(
        production_compiler, text, "--tessera-graph-to-schedule"), "--tessera-schedule-to-tile")
    kernel = next(line for line in tile.splitlines() if "tile.nvfp4_expand_kernel" in line)
    assert 'scale_storage = "e8m0_bits"' in kernel and "mxfp4_e2m1_e8m0_k32" in kernel


def _public_mxfp4(m=3, h=32, k=64, n=19):
    import tessera as ts
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp, mxfp4_inputs
    return ts.jit(target="rocm_gfx1201")(make_mlp(m, h, k, n, source="mxfp4")), mxfp4_inputs(m, h, k, n)


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_public_mxfp4_mlp_packages_two_mxfp4_expansions(rocm_image_toolchain):
    from tessera.compiler.rocm_nvfp4_expand_native import MXFP4_EXPAND_ABI
    from tessera.compiler.rocm_quantized_product_frontend import _compiler_path
    fn, roots = _public_mxfp4()
    program = fn.compile_native_quantized_program(*roots)
    native = program.native
    assert not program.bounded
    assert _compiler_path(native) == "canonical_rocm_quantized_mlp_program"
    for component in (native.gate, native.down):
        assert component.expansion.mxfp4
        assert component.expansion.native.descriptor.abi_id == MXFP4_EXPAND_ABI
        assert component.expansion.numeric_policy() == mxfp4_expansion_policy()
        assert "mxfp4_e2m1_e8m0_k32" in component.expansion.native.target_ir
    plan = json.loads(native.program_json)
    # Gate/up E8M0 plane [2H,K/32] and down plane [N,H/32] are byte roots.
    assert [plan["buffers"][s]["shape"] for s in (2, 5)] == [[64, 2], [19, 1]]
    assert {plan["buffers"][s]["storage"] for s in (2, 5)} <= {"i8", "ui8"}


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_bounded_mxfp4_mlp_is_refused(rocm_image_toolchain):
    fn, roots = _public_mxfp4()
    with pytest.raises(Exception, match="quantized bounded|refuse|MXFP4"):
        fn.compile_native_quantized_program(*roots, m_bound=8, k_bound=128)


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_expansion_execution_requires_the_images_own_architecture(rocm_image_toolchain, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_expand_native import (
        build_nvfp4_expansion_graph, execute_nvfp4_expansion, package_nvfp4_expansion)
    package = package_nvfp4_expansion(build_nvfp4_expansion_graph(
        48, 64, [0, 24, 48], numeric_policy=mxfp4_expansion_policy()))
    assert package.native.image.architecture == "gfx1201"
    codes, exponents = _mx_weights()
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1151")
    with pytest.raises(ValueError, match="requires live gfx1201, not gfx1151"):
        execute_nvfp4_expansion(package, mx.pack_e2m1_codes(codes),
                                np.ascontiguousarray(exponents.T), np.ones(2))
