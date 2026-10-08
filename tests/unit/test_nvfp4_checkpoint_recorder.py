"""Checkpoint recorder contracts independent of downloads or owning hardware."""
import ml_dtypes
import numpy as np
import pytest

from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as recorder


@pytest.mark.parametrize("name", ["q_proj", "gate_proj", "up_proj"])
def test_projection_loader_preserves_named_tensor_and_independent_global_scale(monkeypatch, name):
    tensor = (recorder.TENSOR if name == "q_proj" else f"model.layers.0.mlp.{name}.weight")
    prefix = tensor.removesuffix(".weight")
    names = [tensor, prefix + ".weight_scale", prefix + ".weight_scale_2"]
    index = {"weight_map": dict.fromkeys(names, "selected.safetensors")}
    monkeypatch.setattr(recorder, "_get_json", lambda url: (index, b"index"))
    monkeypatch.setattr(recorder, "_safetensors_header", lambda *args: ("url", {}, 0))
    seen = []

    def read(repo, revision, selected, weight_map, headers):
        seen.append((repo, selected))
        if repo == recorder.BF16_REPO:
            array = np.ones((3, 32), dtype=ml_dtypes.bfloat16)
            dtype = "BF16"
        elif selected.endswith(".weight_scale_2"):
            array = np.asarray([0.5 if name == "gate_proj" else 2.0], dtype=np.float32)
            dtype = "F32"
        elif selected.endswith(".weight_scale"):
            array = np.ones((3, 2), dtype=ml_dtypes.float8_e4m3fn)
            dtype = "F8_E4M3"
        else:
            array = np.zeros((3, 16), dtype=np.uint8)
            dtype = "U8"
        return array, array.tobytes(), {"dtype": dtype, "shape": list(array.shape)}, "selected.safetensors"

    monkeypatch.setattr(recorder, "_read_tensor", read)
    result = recorder._load_projection(tensor)
    assert result["projection"].name == name
    assert result["projection"].global_scale == (0.5 if name == "gate_proj" else 2.0)
    assert result["source"]["tensor"] == tensor
    assert seen == [(recorder.NVFP4_REPO, n) for n in names] + [(recorder.BF16_REPO, tensor)]


def test_projection_loader_rejects_unsupported_source_before_network(monkeypatch):
    def network(*args):
        pytest.fail("invalid source must be rejected before fetching")
    monkeypatch.setattr(recorder, "_get_json", network)
    with pytest.raises(ValueError, match="q_proj or gate/up"):
        recorder._load_projection("model.layers.3.mlp.gate_proj.weight")


def test_bounded_quality_reduction_matches_whole_matrix_reference():
    rng = np.random.default_rng(920)
    ref = rng.normal(size=(1025, 1024)).astype(np.float32)
    got = ref + np.float32(0.03) * rng.normal(size=ref.shape).astype(np.float32)
    result = recorder._relative_rms(ref, got)
    ref64, diff = ref.astype(np.float64), ref.astype(np.float64) - got.astype(np.float64)
    ratio = np.square(diff).sum() / np.square(ref64).sum()
    assert result["relative_rms_error"] == pytest.approx(np.sqrt(ratio), rel=1e-12)
    assert result["sqnr_db"] == pytest.approx(-10 * np.log10(ratio), rel=1e-12)
    assert result["max_abs_error"] == np.max(np.abs(diff))


@pytest.mark.parametrize("kwargs", [{"m": 0}, {"repeats": 0}, {"iterations": 0},
                                  {"projection_group": "unsupported"}])
def test_invalid_measurement_arguments_are_rejected_before_gpu(monkeypatch, kwargs):
    monkeypatch.setattr(recorder.rt, "_rocm_live_arch", lambda: pytest.fail("unexpected GPU probe"))
    with pytest.raises(ValueError):
        recorder.measure(**kwargs)


def test_resident_recorder_rejects_wrong_output_before_creating_timing_events(monkeypatch):
    from types import SimpleNamespace
    from benchmarks.rocm import benchmark_rocm_nvfp4_ingest_schedule as scheduled

    calls = []

    class Hip:
        def hipInit(self, ordinal):
            return 0

        def hipModuleLoadData(self, handle, payload):
            handle._obj.value = 1
            return 0

        def hipModuleGetFunction(self, handle, module, entry):
            handle._obj.value = 2
            return 0

        def hipMalloc(self, handle, size):
            handle._obj.value = 3
            return 0

        def hipMemcpy(self, *args):
            # Leave the pre-zeroed output unchanged: deliberately wrong.
            return 0

        def hipModuleLaunchKernel(self, *args):
            return 0

        def hipDeviceSynchronize(self):
            return 0

        def hipEventCreate(self, *args):
            pytest.fail("wrong output must abort before the first timing event")

        def hipFree(self, *args):
            calls.append("free")
            return 0

        def hipModuleUnload(self, *args):
            calls.append("unload")
            return 0

    monkeypatch.setattr(scheduled.rt, "_load_hip_for_launch", lambda: Hip())
    package = SimpleNamespace(
        image=SimpleNamespace(payload=b"mock"),
        descriptor=SimpleNamespace(entry_symbol="entry", dynamic_local_memory_bytes=0,
                                   geometry=SimpleNamespace(grid=(1, 1, 1), workgroup=(32, 1, 1))),
    )
    buffers = {name: np.zeros((1,), dtype=np.float32)
               for name in ("a", "b_packed", "a_scale", "b_scale", "output")}
    with pytest.raises(AssertionError):
        scheduled._device_resident_run(package, buffers, (1, 1, 32),
                                       expected=np.ones((1,), dtype=np.float32))
    assert calls == ["free"] * 5 + ["unload"]


@pytest.mark.parametrize("corruption", [None, "codes", "scales", "stats"])
def test_native_checkpoint_conversion_checks_outputs_before_timing(monkeypatch, corruption):
    from types import SimpleNamespace
    from tessera.compiler import rocm_nvfp4_ingest as ingest
    projections = [
        ingest.NVFP4Projection(name, np.full((2, 16), 0x32, np.uint8),
            np.ones((2, 2), dtype=ml_dtypes.float8_e4m3fn), global_scale)
        for name, global_scale in (("gate", 0.5), ("up", 2.0))
    ]
    reference = ingest.ingest_nvfp4_projections(projections)
    codes = np.concatenate([p.packed_codes for p in projections])
    scales = np.concatenate([p.e4m3_scales for p in projections])
    globals_ = np.array([0.5, 2.0], np.float64)
    oracle = ingest.reference_nvfp4_requantize(
        codes, scales, globals_, row_offsets=reference.row_offsets,
        numeric_policy=ingest.nvfp4_requantization_policy())
    package = SimpleNamespace(graph_ir="graph", schedule_ir="schedule", native=SimpleNamespace(
        tile_ir="tile", target_ir="target", image=SimpleNamespace(payload=b"image"),
        descriptor=SimpleNamespace(entry_symbol="native_ingest", abi_id="checked")))
    monkeypatch.setattr(recorder.native_ingest, "build_nvfp4_ingest_graph",
                        lambda *args, **kwargs: "graph")
    monkeypatch.setattr(recorder.native_ingest, "package_nvfp4_ingest_graph",
                        lambda graph: package)
    calls = []

    def execute(package, got_codes, got_scales, got_globals, *, event_samples=None):
        np.testing.assert_array_equal(got_codes, codes)
        np.testing.assert_array_equal(got_scales, scales)
        np.testing.assert_array_equal(got_globals, globals_)
        calls.append(event_samples is not None)
        if event_samples is not None:
            assert corruption is None, "wrong outputs must be rejected before timing"
            event_samples.extend([0.1, 0.2, 0.3])
        outputs = [x.copy() for x in oracle]
        if corruption is not None:
            index = {"codes": 0, "scales": 1, "stats": 2}[corruption]
            outputs[index].flat[0] += 1
        return tuple(outputs)

    monkeypatch.setattr(recorder.native_ingest, "execute_nvfp4_ingest", execute)
    if corruption:
        with pytest.raises(AssertionError):
            recorder._native_checkpoint_ingest(projections, reference)
        assert calls == [False]
    else:
        weights, witness = recorder._native_checkpoint_ingest(projections, reference)
        np.testing.assert_array_equal(weights.packed_codes, reference.packed_codes)
        assert witness["resident_event_ms_median"] == 0.2
        assert witness["route"].startswith("GraphIR->ScheduleIR->TileIR")
        assert calls == [False, True]
