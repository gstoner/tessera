"""Exact-device shared-RHS batches through native Graph/Schedule/Tile NVFP4."""
import copy
import numpy as np
import pytest
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvfp4_shared_rhs_batch import batch_module, frontend_batch_module
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("python_frontend", [False, True])
@pytest.mark.parametrize("batch,rows,n,k", [(1, 1, 1, 1), (3, 7, 5, 31), (2, 17, 19, 129), (5, 9, 11, 64)])
def test_shared_rhs_nvfp4_batch_matches_independent_oracle(batch, rows, n, k, python_frontend, monkeypatch):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device and matching compiler")
    from tessera.compiler import scheduled_matmul
    calls = []
    native_run = scheduled_matmul.run_tessera_opt
    def record_native_run(*args, **kwargs):
        calls.extend(value for value in args[2:] if isinstance(value, str))
        return native_run(*args, **kwargs)
    monkeypatch.setattr(scheduled_matmul, "run_tessera_opt", record_native_run)
    module = (frontend_batch_module if python_frontend else batch_module)(batch, rows, n, k)
    original = copy.deepcopy(module)
    bundle = compile_graph_module(module, source_origin="W1.1", target="nvidia_sm120",
                                 options={"package_native": True}, enable_tool_validation=False)
    assert module == original
    assert calls.count("--tessera-graph-to-schedule") == 1
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    descriptor = bundle.launch_descriptor
    assert descriptor is not None
    assert [binding.rank for binding in descriptor.buffers] == [3, 2, 3, 2, 3]
    assert descriptor.provenance["batch_rows"] == [batch, rows]
    assert f"tensor<{batch}x{rows}x{k}x!tessera.nvfp4>" in bundle.graph.text
    assert "shared_rhs_rows" in bundle.schedule.text
    assert "tile.matmul_kernel" in bundle.tile.text
    assert "tessera.storage_pack" in bundle.tile.text
    assert "mma.sync.aligned.m16n8k64" in bundle.native_image.payload.decode()
    rng = np.random.default_rng(120607 + batch + rows + n + k)
    sk = (k + 15) // 16
    choices = np.asarray([0x30, 0x31, 0x33, 0x35, 0x38, 0x3A, 0x40], np.uint8)
    for repeat in range(2):
        ac = rng.integers(0, 16, (batch, rows, k), dtype=np.uint8)
        bc = rng.integers(0, 16, (k, n), dtype=np.uint8)
        sa = choices[(np.arange(batch)[:, None, None] * 3 + np.arange(rows)[None, :, None]
                      + np.arange(sk)[None, None, :] + repeat) % choices.size]
        sb = choices[(np.arange(sk)[:, None] * 2 + np.arange(n)[None, :]) % choices.size]
        out = np.full((batch, rows, n), np.nan, np.float32)
        args = {"a": _pack_nvfp4(ac, 2), "b": _pack_nvfp4(bc, 0),
                "sa": np.ascontiguousarray(sa), "sb": np.ascontiguousarray(sb), descriptor.buffers[4].name: out,
                "M": batch * rows, "N": n, "K": k}
        receipt = launch(artifact, args)
        assert receipt["ok"], receipt
        assert receipt["execution_kind"] == "native_gpu"
        decoded_a = _decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=2)[:, :, :k]
        decoded_b = _decode_e2m1(bc) * np.repeat(_decode_ue4m3(sb), 16, axis=0)[:k, :]
        expected = decoded_a.astype(np.float64) @ decoded_b.astype(np.float64)
        np.testing.assert_allclose(out, expected, rtol=0, atol=2e-3)
        # Shape guards retain the logical outer axis; equal total byte counts
        # cannot authorize a permuted batch/row shape or stale flattened M.
        bad = dict(args, a=args["a"].reshape(rows, batch, (k + 1) // 2))
        if batch != rows:
            assert not launch(artifact, bad)["ok"]
        assert not launch(artifact, dict(args, M=batch * rows + 1))["ok"]


@pytest.mark.hardware_nvidia
def test_authored_rank_two_scaled_nvfp4_preserves_the_existing_scale_abi():
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device and matching compiler")
    from tests.device.nvidia.test_e2e_spine_native import _nvfp4_module
    module = _nvfp4_module(7, 5, 31)
    op = module.functions[0].body[0]
    op.op_name = "tessera.scaled_matmul"
    op.kwargs = dict(batch_module().functions[0].body[0].kwargs)
    op.kwargs.pop("batching")
    bundle = compile_graph_module(module, source_origin="W1.1", target="nvidia_sm120",
                                 options={"package_native": True}, enable_tool_validation=False)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    assert [binding.rank for binding in bundle.launch_descriptor.buffers] == [2] * 5
    rng = np.random.default_rng(120609)
    ac = rng.integers(0, 16, (7, 31), dtype=np.uint8)
    bc = rng.integers(0, 16, (31, 5), dtype=np.uint8)
    sa = np.asarray([[0x30 + i % 7, 0x38] for i in range(7)], np.uint8)
    sb = np.asarray([[0x38] * 5, [0x30 + i for i in range(5)]], np.uint8)
    out = np.empty((7, 5), np.float32)
    result = launch(artifact, {"a": _pack_nvfp4(ac, 1), "b": _pack_nvfp4(bc, 0),
                             "scale_a": sa, "scale_b": sb, "c": out, "M": 7, "N": 5, "K": 31})
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    expected = ((_decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=1)[:, :31]).astype(np.float64)
                @ (_decode_e2m1(bc) * np.repeat(_decode_ue4m3(sb), 16, axis=0)[:31]).astype(np.float64))
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-3)
