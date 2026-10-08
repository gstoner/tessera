"""Exact-device proofs for the canonical scheduled SM120 matmul package.

The implementation helpers remain beside the host-free scheduled-contract
tests, but NVIDIA-TEST-6 requires every hardware node to be collected from a
device root.  Exporting marked aliases here keeps one implementation while
making the proof environment explicit.
"""

from __future__ import annotations

import pytest

from tests.unit import test_scheduled_matmul_consumers as _shared


test_sm120_typed_scheduled_matmul_executes_exact_artifact = (
    pytest.mark.hardware_nvidia(
        _shared.test_sm120_typed_scheduled_matmul_executes_exact_artifact
    )
)
test_sm120_macro_cta_reuses_shared_panels_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_macro_cta_reuses_shared_panels_exact_device
    )
)
test_sm120_macro_cta_k_tail_exact_device = pytest.mark.hardware_nvidia(
    _shared._sm120_macro_cta_k_tail_exact_device
)
test_sm120_scheduled_epilogue_reduced_output_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_scheduled_epilogue_reduced_output_exact_device
    )
)
test_sm120_macro_cta_bf16_exact_device = pytest.mark.hardware_nvidia(
    _shared._sm120_macro_cta_bf16_exact_device
)
test_sm120_bounded_dynamic_strided_matmul_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared.test_sm120_bounded_dynamic_strided_matmul_exact_device
    )
)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 19, 23), (48, 67, 17)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("output_dtype", ["fp32","fp16"])
@pytest.mark.parametrize("fused", [False,True])
def test_sm120_static_ragged_mnk_preserves_resident_buffer_canaries(shape, dtype, output_dtype, fused):
    """Prove bounded fragment loads/stores with resident guard allocations."""
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests._support.nvidia import nvidia_cuda_host_ready

    if (not nvidia_cuda_host_ready()
            or not _shared.nvidia_native.tools_available()
            or _shared.scheduled_matmul.find_tessera_opt() is None):
        pytest.skip("requires the owning SM120 compiler, bridge and CUDA device")
    assert rt._nvidia_device_name() == "sm_120"
    m, k, n = shape
    storage = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    bundle = compile_graph_module(
        _shared._module(target="nvidia_sm120", shape=shape, dtype=dtype,
                        output_dtype=output_dtype,bias=fused,residual=fused,
                        activation="relu" if fused else "none"),
        source_origin="W1.1-static-ragged-MNK-resident-canaries",
        target="nvidia_sm120", options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and "tile.matmul_kernel" not in bundle.tile.text
    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image, launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text, target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(0x120 + m + n + k)
    a = rng.normal(0.0, 0.25, (m, k)).astype(storage)
    b = np.asfortranarray(rng.normal(0.0, 0.25, (k, n)).astype(storage))
    guard = 1024
    output_storage = np.float16 if output_dtype == "fp16" else np.float32
    output_guard = output_storage(-12345.5)
    bias = rng.normal(0.0,.1,(n,)).astype(np.float32)
    residual = rng.normal(0.0,.1,(m,n)).astype(np.float32)

    def guarded_flat(array):
        padded = np.full(array.size + 2 * guard, 512, dtype=storage)
        padded[guard:-guard] = array.reshape(-1, order="F" if array is b else "C")
        return padded

    a_host, b_host = guarded_flat(a), guarded_flat(b)
    output_host = np.full(m * n + 2 * guard, output_guard, output_storage)
    with NvidiaDeviceSession() as session:
        a_owner = session.upload(a_host)
        b_owner = session.upload(b_host, layout="col_major")
        output_owner = session.upload(output_host)
        a_view = a_owner.view(guard * np.dtype(storage).itemsize, (m, k), storage)
        b_view = b_owner.view(guard * np.dtype(storage).itemsize, (k, n), storage)
        output_view = output_owner.view(guard * np.dtype(output_storage).itemsize,
                                        (m, n), output_storage)
        extras = {}
        extra_views = []
        extra_owners = []
        if fused:
            for name,value in (("bias",bias),("residual",residual)):
                padded = np.full(value.size+2*guard,512.,np.float32)
                padded[guard:-guard] = value.reshape(-1)
                owner=session.upload(padded)
                view=owner.view(guard*4,value.shape,np.float32)
                extras[name]=view
                extra_views.append(view)
                extra_owners.append((owner,padded))
        try:
            receipt = rt.launch(
                artifact, {"a": a_view, "b": b_view, "o": output_view, **extras,
                           "M": m, "N": n, "K": k}, stream=session.stream,
            )
            assert receipt["ok"], receipt.get("reason")
            assert receipt["execution_kind"] == "native_gpu", receipt
            # Download synchronizes this stream before any borrowed view is closed.
            actual = output_owner.numpy()
            expected = a.astype(np.float32) @ b.astype(np.float32)
            if fused:
                expected = np.maximum(expected+bias,0)+residual
            if output_dtype == "fp16":
                expected = expected.astype(np.float16)
            np.testing.assert_allclose(
                actual[guard:-guard].reshape(m, n), expected, rtol=2e-4, atol=2e-4,
            )
            np.testing.assert_array_equal(actual[:guard], output_host[:guard])
            np.testing.assert_array_equal(actual[-guard:], output_host[-guard:])
            np.testing.assert_array_equal(a_owner.numpy(), a_host)
            np.testing.assert_array_equal(b_owner.numpy(), b_host)
            for owner,padded in extra_owners:
                np.testing.assert_array_equal(owner.numpy(),padded)
        finally:
            for view in (output_view, b_view, a_view, *extra_views):
                view.close()
    assert a_owner._closed and b_owner._closed and output_owner._closed
