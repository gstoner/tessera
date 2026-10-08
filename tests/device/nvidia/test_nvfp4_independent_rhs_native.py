"""Native Tile/LLVM/PTX independent-RHS batch proof; Graph packaging is pending."""
import ctypes
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.runtime import _load_nvidia_ptx_launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("batch,rows,n,k", [(3, 7, 5, 31), (2, 17, 19, 129), (5, 1, 11, 64)])
def test_native_independent_rhs_batch_keeps_ragged_rows_in_owning_batch(batch, rows, n, k):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device and matching compiler")
    root = Path(__file__).resolve().parents[3]
    tile = (root / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/sm120_nvfp4_independent_rhs_batch.mlir").read_text()
    entry = b"tessera_tile_matmul_nvfp4_batched"
    _, ptx, *_ = _compile_tile_ir(tile, entry.decode())
    lib = _load_nvidia_ptx_launch()
    assert lib is not None
    assert lib.tessera_nvidia_ptx_register(entry, ptx.encode()) == 0
    rng = np.random.default_rng(120610 + batch + rows + n + k)
    scale_k = k // 16 + (k % 16 != 0)
    choices = np.asarray([0x30, 0x33, 0x35, 0x38, 0x3A, 0x40], np.uint8)
    for refresh in range(2):
        ac = rng.integers(0, 16, (batch, rows, k), dtype=np.uint8)
        bc = rng.integers(0, 16, (batch, k, n), dtype=np.uint8)
        sa = np.ascontiguousarray(choices[(np.arange(batch)[:, None, None] * 3
            + np.arange(rows)[None, :, None] + np.arange(scale_k)[None, None, :] + refresh) % len(choices)])
        sb = np.ascontiguousarray(choices[(np.arange(batch)[:, None, None] * 2
            + np.arange(scale_k)[None, :, None] + np.arange(n)[None, None, :] + refresh) % len(choices)])
        out = np.full((batch, rows, n), np.nan, np.float32)
        values = [_pack_nvfp4(ac, 2), _pack_nvfp4(bc, 1), sa, sb, out]
        pointers = (ctypes.c_void_p * 5)(*[value.ctypes.data for value in values])
        dims = (ctypes.c_int64 * 5)(batch * rows, n, k, rows, batch)
        assert lib.tessera_nvidia_ptx_invoke(entry, pointers, 5, dims, 5) == 0
        a = _decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=2)[:, :, :k]
        b = _decode_e2m1(bc) * np.repeat(_decode_ue4m3(sb), 16, axis=1)[:, :k, :]
        expected = a.astype(np.float64) @ b.astype(np.float64)
        np.testing.assert_allclose(out, expected, rtol=0, atol=2e-3)
        saved = out.copy()
        for bad in ((batch * rows, n, k, rows + 1, batch),
                    (batch * rows, n, k, 0, batch),
                    (batch * rows, n, k, rows, 0),
                    (65536, n, k, 1, 65536)):
            invalid = (ctypes.c_int64 * 5)(*bad)
            assert lib.tessera_nvidia_ptx_invoke(entry, pointers, 5, invalid, 5) == 5
            np.testing.assert_array_equal(out, saved)
        assert lib.tessera_nvidia_ptx_invoke(entry, pointers, 5, dims, 3) == 5


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("batch,rows,n,k", [(3, 7, 5, 31), (2, 17, 19, 129), (5, 1, 11, 64)])
def test_python_graph_schedule_independent_batch_package_matches_oracle(batch, rows, n, k):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device and matching compiler")
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.nvidia_native import SM120_NVFP4_BATCH_ABI
    from tessera.runtime import launch, _nvidia_native_descriptor_device_latency
    from tests.unit.test_nvfp4_shared_rhs_batch import frontend_batch_module

    module = frontend_batch_module(batch, rows, n, k, "independent_rhs")
    bundle = compile_graph_module(module, source_origin="W1.1", target="nvidia_sm120",
                                 options={"package_native": True}, enable_tool_validation=False)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    descriptor = bundle.launch_descriptor
    assert descriptor is not None and descriptor.abi_id == SM120_NVFP4_BATCH_ABI
    assert [binding.rank for binding in descriptor.buffers] == [3] * 5
    assert len(descriptor.scalars) == 5
    assert "independent_rhs" in bundle.graph.text
    assert "independent_rhs" in bundle.schedule.text
    assert "independent_rhs" in bundle.tile.text
    assert "mma.sync.aligned.m16n8k64" in bundle.native_image.payload.decode()
    rng = np.random.default_rng(120611 + rows + batch + k)
    scale_k = (k + 15) // 16
    ac = rng.integers(0, 16, (batch, rows, k), dtype=np.uint8)
    bc = rng.integers(0, 16, (batch, k, n), dtype=np.uint8)
    choices = np.asarray([0x30, 0x33, 0x38, 0x3A, 0x40], np.uint8)
    sa = np.ascontiguousarray(choices[(np.arange(batch)[:, None, None] * 2
        + np.arange(rows)[None, :, None] + np.arange(scale_k)[None, None, :]) % len(choices)])
    sb = np.ascontiguousarray(choices[(np.arange(batch)[:, None, None] * 3
        + np.arange(scale_k)[None, :, None] + np.arange(n)[None, None, :]) % len(choices)])
    out = np.full((batch, rows, n), np.nan, np.float32)
    arguments = {"a": _pack_nvfp4(ac, 2), "b": _pack_nvfp4(bc, 1), "sa": sa, "sb": sb,
                 descriptor.buffers[4].name: out, "M": batch * rows, "N": n, "K": k,
                 "BatchRows": rows, "BatchCount": batch}
    receipt = launch(artifact, arguments)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    a = _decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=2)[:, :, :k]
    b = _decode_e2m1(bc) * np.repeat(_decode_ue4m3(sb), 16, axis=1)[:, :k, :]
    np.testing.assert_allclose(out, a.astype(np.float64) @ b.astype(np.float64), rtol=0, atol=2e-3)
    # Equal flattened M must not permit a different batch count/row split.
    rebound = dict(arguments, BatchRows=batch, BatchCount=rows)
    if batch != rows:
        assert not launch(artifact, rebound)["ok"]
        with pytest.raises(RuntimeError, match="compiled envelope"):
            _nvidia_native_descriptor_device_latency(bundle.native_image, descriptor, rebound, reps=1, warmup=0)
