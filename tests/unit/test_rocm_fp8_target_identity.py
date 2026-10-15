"""Native FP8 Target projection must retain both physical operand formats."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from tessera.compiler import rocm_native as native, scheduled_matmul
from tests.unit.test_rocm_matmul_shape_key import _target as _half_target
from tests.unit.test_scheduled_matmul_consumers import _module


@pytest.fixture(autouse=True)
def native_compiler():
    tool = native._tessera_opt()
    if tool is None:
        pytest.skip("requires ROCm-enabled tessera-opt")
    help_text = subprocess.run([str(tool), "--help"], check=True, capture_output=True, text=True).stdout
    if "tessera-rocm-project-kernel-identity" not in help_text:
        pytest.skip("requires native Target image projection")


def _target(lhs, rhs, *, shape=16, digest="a" * 64):
    return (
        _half_target(shape, digest)
        .replace("gfx1151", "gfx1201")
        .replace(
            'arch = "gfx1201", k',
            f'arch = "gfx1201", numeric_policy = {{accum = "f32", storage = "{lhs}"}}, rhs_storage = "{rhs}", k',
        )
    )


def _project(source):
    return native._shape_free_target_ir(source, family="matmul", directive="tessera_rocm.wmma_gemm")


def test_all_four_fp8_pairs_have_distinct_shape_independent_native_identity():
    projected = []
    for lhs in ("e4m3", "e5m2"):
        for rhs in ("e4m3", "e5m2"):
            first = _project(_target(lhs, rhs))
            second = _project(_target(lhs, rhs, shape=48, digest="b" * 64))
            assert first == second
            assert f'rhs_storage = "{rhs}"' in first
            projected.append(first)
    assert len(set(projected)) == 4


@pytest.mark.parametrize(
    "source",
    [
        _target("e4m3", "f16"),
        _target("f16", "e4m3"),
        _target("e4m3", "e5m2").replace("gfx1201", "gfx1151"),
        _target("e4m3", "e5m2").replace('numeric_policy = {accum = "f32", storage = "e4m3"}, ', ""),
    ],
)
def test_rhs_format_refuses_invalid_or_incomplete_target_contract(source):
    with pytest.raises(RuntimeError, match="rhs_storage requires"):
        _project(source)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1" and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 device proof",
)
@pytest.mark.parametrize(
    "lhs,rhs",
    [
        ("fp8_e4m3", "fp8_e4m3"),
        ("fp8_e4m3", "fp8_e5m2"),
        ("fp8_e5m2", "fp8_e4m3"),
        ("fp8_e5m2", "fp8_e5m2"),
        ("fp16", "fp16"),
        ("bf16", "bf16"),
    ],
)
@pytest.mark.parametrize(
    "staging,k_unroll", [("register", 1), ("register", 2), ("register", 4), ("register", 8), ("lds", 1)]
)
@pytest.mark.parametrize("architecture", ["gfx1201", "gfx1151"])
def test_projected_image_executes_three_aligned_and_ragged_shapes(lhs, rhs, staging, k_unroll, architecture):
    import ml_dtypes
    from tessera import runtime as rt

    if rt._rocm_live_arch() != architecture:
        pytest.skip("requires the owning architecture")
    if architecture == "gfx1151" and lhs.startswith("fp8"):
        pytest.skip("RDNA3.5 has no FP8 WMMA")
    assert rt._rocm_live_arch() == rt._rocm_chip() == architecture
    types = {
        "fp8_e4m3": ml_dtypes.float8_e4m3fn,
        "fp8_e5m2": ml_dtypes.float8_e5m2,
        "fp16": np.float16,
        "bf16": ml_dtypes.bfloat16,
    }
    images = []
    for shape in ((16, 32, 16), (17, 31, 19), (48, 33, 16)):
        m, k, n = shape
        scheduled = scheduled_matmul.lower_scheduled_matmul(
            _module(target="rocm", shape=shape, dtype=lhs, b_dtype=rhs), target=f"rocm_{architecture}"
        )
        package = native.package_scheduled_matmul(
            scheduled, pipeline_name="tessera-lower-to-rocm", k_unroll=k_unroll, staging=staging
        )
        image, descriptor, target = package.image, package.descriptor, package.target_ir
        if images:
            assert image.compile_state == "warm_cache"
        assert (
            descriptor.provenance["b_storage"]
            == {"fp8_e4m3": "e4m3", "fp8_e5m2": "e5m2", "fp16": "f16", "bf16": "bf16"}[rhs]
        )
        if lhs.startswith("fp8"):
            assert "rhs_storage" in target
        assert descriptor.entry_symbol.startswith("tessera_rocm_matmul_")
        rng = np.random.default_rng(47032 + m)
        a = rng.uniform(-0.5, 0.5, (m, k)).astype(types[lhs])
        b = rng.uniform(-0.5, 0.5, (k, n)).astype(types[rhs])
        output = np.zeros((m, n), np.float32)
        artifact = rt.RuntimeArtifact(
            metadata={"target": image.target},
            native_image=image,
            launch_descriptor=descriptor,
            tile_ir=scheduled.tile_ir,
            target_ir=target,
        )
        result = rt.launch(artifact, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}})
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        np.testing.assert_allclose(output, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4, atol=1e-3)
        images.append(image.image_digest)
    assert len(set(images)) == 1


@pytest.mark.parametrize("architecture", ["gfx1151", "gfx1201"])
def test_generic_launch_checks_selected_device_before_module_acquisition(monkeypatch, architecture):
    from types import SimpleNamespace
    from tessera import runtime as rt

    scheduled = scheduled_matmul.lower_scheduled_matmul(
        _module(target="rocm", shape=(17, 31, 19), dtype="fp16"), target=f"rocm_{architecture}"
    )
    package = native.package_scheduled_matmul(scheduled, pipeline_name="tessera-lower-to-rocm", k_unroll=1)
    selected = {"arch": architecture}
    acquired = []

    class PastArchitectureGuard(RuntimeError):
        pass

    def module_load(*args):
        acquired.append(True)
        raise PastArchitectureGuard("guard allowed module acquisition")

    monkeypatch.setattr(
        rt, "_load_hip_for_launch", lambda: SimpleNamespace(hipInit=lambda _: 0, hipModuleLoadData=module_load)
    )
    monkeypatch.setattr(rt, "_load_rocm_native_image_runtime", lambda: None)
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: selected["arch"])
    buffers = {
        "a": np.zeros((17, 31), np.float16),
        "b": np.zeros((31, 19), np.float16),
        "o": np.full((17, 19), 947, np.float32),
    }
    scalars = {"M": 17, "N": 19, "K": 31}
    with pytest.raises(PastArchitectureGuard):
        rt._submit_rocm_gfx1151_native(package.image, package.descriptor, buffers, scalars, None)
    assert len(acquired) == 1
    for wrong in ("gfx1201" if architecture == "gfx1151" else "gfx1151", None):
        selected["arch"] = wrong
        with pytest.raises(ValueError, match="differs from the live GPU"):
            rt._submit_rocm_gfx1151_native(package.image, package.descriptor, buffers, scalars, None)
        assert len(acquired) == 1
        assert np.all(buffers["o"] == 947)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1" and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 device proof",
)
@pytest.mark.parametrize("target", ["rocm", "rocm_gfx1201"])
def test_public_jit_fp8_pairs_and_shape_replay_use_native_compiler(target):
    import ml_dtypes
    import tessera as ts
    from tessera import runtime as rt

    if rt._rocm_live_arch() != "gfx1201":
        pytest.skip("requires the owning gfx1201")
    public = ts.from_text('def product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="fp32")\n', target=target)
    types = (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2)
    paired_images = {}
    retained = []
    for a_dtype in types:
        for b_dtype in types:
            images = []
            for m, k, n in ((16, 32, 16), (17, 31, 19), (48, 33, 16)):
                rng = np.random.default_rng(9471201 + m)
                a = rng.uniform(-0.5, 0.5, (m, k)).astype(a_dtype)
                b = rng.uniform(-0.5, 0.5, (k, n)).astype(b_dtype)
                original_a, original_b = a.copy(), b.copy()
                a.flags.writeable = b.flags.writeable = False
                output = public(a=a, b=b)
                np.testing.assert_allclose(output, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4, atol=1e-3)
                np.testing.assert_array_equal(a, original_a)
                np.testing.assert_array_equal(b, original_b)
                assert public.execution_kind == "native_gpu"
                assert public._native_descriptor_last_receipt["compiler_path"] == "rocm_gfx1201_native_descriptor"
                assert public.compile_result.executable
                bundle = public.compile_result.bundle
                assert "schedule.matmul" in bundle.schedule.text
                assert "tile.matmul_kernel" in bundle.tile.text
                assert "tessera_rocm.wmma_gemm" in bundle.target_ir.text
                assert public.last_fallback_reason is None
                images.append(bundle.native_image.image_digest)
                retained.append((output, output.copy()))
            assert len(set(images)) == 1
            paired_images[(str(a_dtype), str(b_dtype))] = images[0]
    assert len(set(paired_images.values())) == 4
    for output, expected in retained:
        np.testing.assert_array_equal(output, expected)


def test_canonical_fp8_matmul_defaults_to_native_and_respects_override(monkeypatch):
    from tessera.compiler import driver

    monkeypatch.setattr(native, "native_packaging_available", lambda: True)
    module = _module(target="rocm", dtype="fp8_e4m3", b_dtype="fp8_e5m2")
    assert driver.canonical_compile_options(module, target="rocm_gfx1201")["package_native"] is True
    assert (
        driver.canonical_compile_options(module, target="rocm_gfx1201", options={"package_native": False})[
            "package_native"
        ]
        is False
    )
    assert driver.canonical_compile_options(module, target="rocm_gfx1151")["package_native"] is False


def test_selected_rocm_frontend_compile_failure_propagates(monkeypatch):
    import tessera as ts
    from tessera.compiler import canonical_compile as canonical

    public = ts.from_text(
        'def product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="fp32")\n', target="rocm_gfx1201"
    )

    def refuse(*args, **kwargs):
        raise RuntimeError("selected native compiler failure")

    monkeypatch.setattr(canonical, "canonical_compile", refuse)
    with pytest.raises(RuntimeError, match="selected native compiler failure"):
        public(np.ones((17, 31), np.float16), np.ones((31, 19), np.float16))


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1" and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 device proof",
)
@pytest.mark.parametrize("storage", ["fp16", "bf16"])
def test_public_rocm_matmul_sibling_storage_uses_checked_native_route(storage):
    import ml_dtypes
    import tessera as ts
    from tessera import runtime as rt

    if rt._rocm_live_arch() != "gfx1201":
        pytest.skip("requires the owning gfx1201")
    output_dtype = "int32" if storage == "int8" else "fp32"
    public = ts.from_text(
        f'def product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="{output_dtype}")\n', target="rocm_gfx1201"
    )
    dtype = {"fp16": np.float16, "bf16": ml_dtypes.bfloat16, "int8": np.int8}[storage]
    rng = np.random.default_rng(9471202)
    a = rng.integers(-4, 4, (17, 31)).astype(dtype)
    b = rng.integers(-4, 4, (31, 19)).astype(dtype)
    output = public(a, b)
    expected = a.astype(np.int32 if storage == "int8" else np.float32) @ b.astype(
        np.int32 if storage == "int8" else np.float32
    )
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == np.dtype(output_dtype.replace("fp", "float"))
    assert public.execution_kind == "native_gpu"


def test_public_integer_matmul_keeps_its_existing_wrap_contract():
    import tessera as ts

    public = ts.from_text("def product(a,b):\n    return tessera.ops.matmul(a,b)\n", target="rocm_gfx1201")
    a = np.full((2, 32), 7, np.int8)
    b = np.full((32, 2), 7, np.int8)
    expected = ts.ops.matmul(a, b)
    output = public(a, b)
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == expected.dtype == np.dtype("int8")
    assert public.execution_kind != "native_gpu"
