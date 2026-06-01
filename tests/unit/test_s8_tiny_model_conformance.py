"""S8 real tiny-model conformance across the S2-S15 standalone surface."""

from __future__ import annotations

import numpy as np

import tessera as ts
from examples.conformance.s8_tiny_models import TinyModelSpec, manifest, surfaces_covered
from tessera.compiler.primitive_coverage import coverage_for
from tessera.compiler.jit import jit
from tessera.testing.compiler_examples import FOUNDATION_TARGETS


EXPECTED_MODEL_IDS = {
    "tiny_diffusion_dit",
    "tiny_xlstm_recurrent",
    "tiny_mamba_ssm",
    "tiny_hyena_fnet_spectral",
    "tiny_linformer_cosformer",
    "tiny_griffin_megalodon",
    "tiny_jepa",
    "tiny_titans_atlas_memory",
    "tiny_qwen3_moe_decoder",
    "tiny_deepseek_v3_mla_moe_decode",
}


def _assert_tree_shapes(tree, expected: dict[str, tuple[int, ...]]) -> None:
    for name, shape in expected.items():
        assert name in tree, f"missing parameter {name}"
        assert np.asarray(tree[name]).shape == shape


def _assert_loss_grad(spec: TinyModelSpec) -> None:
    params = spec.init()
    batch = spec.sample_batch()
    value, grad_w = ts.value_and_grad(lambda w: spec.loss({**params, "w": w}, batch))(params["w"])
    assert np.asarray(value).shape == ()
    assert np.isfinite(value)
    assert grad_w.shape == spec.expected_grad_shapes["w"]
    assert np.all(np.isfinite(grad_w))


def test_s8_manifest_names_all_real_tiny_model_families():
    specs = manifest()
    ids = {spec.model_id for spec in specs}
    assert ids == EXPECTED_MODEL_IDS
    assert len({spec.family for spec in specs}) == len(specs)
    for spec in specs:
        assert callable(spec.init)
        assert callable(spec.forward)
        assert callable(spec.loss)
        assert callable(spec.sample_batch)
        assert callable(spec.compile_fn)
        assert spec.compile_inputs


def test_s8_suite_covers_s2_through_s15_aggregate_surface():
    assert surfaces_covered() == {f"S{i}" for i in range(2, 16)}
    assert (
        coverage_for("tiny_training_step_conformance").metadata["model_manifest"]
        == "examples.conformance.s8_tiny_models.manifest"
    )


def test_s8_tiny_models_forward_backward_and_shape_variants():
    for spec in manifest():
        params = spec.init()
        _assert_tree_shapes(params, spec.expected_grad_shapes)

        batch = spec.sample_batch()
        y = spec.forward(params, batch)
        assert y.shape == spec.expected_output_shape
        assert np.all(np.isfinite(y))
        assert np.isfinite(spec.loss(params, batch))
        _assert_loss_grad(spec)

        variant_batch = spec.sample_batch(2)
        variant_y = spec.forward(params, variant_batch)
        assert variant_y.shape[-1:] == spec.expected_output_shape[-1:]


def test_s8_cross_cutting_state_rng_data_quant_sharding_checkpoint_and_aot(tmp_path):
    specs = manifest()
    first = specs[0]
    params = first.init()
    batch = first.sample_batch()

    key = ts.rng.RNGKey.from_seed(8808, name="s8")
    shuffled = ts.data.Dataset.from_tensor_slices(np.arange(6)).shuffle(key).batch(3).to_list()
    restored_key = ts.rng.RNGKey.from_state(key.to_state())
    replay = ts.data.Dataset.from_tensor_slices(np.arange(6)).shuffle(restored_key).batch(3).to_list()
    np.testing.assert_array_equal(shuffled[0], replay[0])

    leaves, treedef = ts.state.tree_flatten({"params": params, "metrics": {"loss": np.asarray(first.loss(params, batch))}})
    restored_tree = ts.state.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(restored_tree["params"]["w"], params["w"])

    mesh = ts.NamedMesh(("data",), {"data": 1})
    sharding = ts.named_sharding(mesh, ts.partition_spec("data"))
    assert sharding.mesh is mesh
    np.testing.assert_allclose(ts.psum(np.array([1.0, 2.0]), axis_name="data"), 3.0)

    q, scale, zp = ts.quantize_int8(params["w"].astype(np.float32))
    dq = ts.dequantize_int8(q, scale, zp)
    assert dq.shape == params["w"].shape
    assert ts.fake_quantize(params["w"].astype(np.float32), num_bits=4).shape == params["w"].shape

    @ts.custom_primitive("s8_scale_for_conformance")
    def s8_scale_for_conformance(x):
        return np.asarray(x) * 2.0

    @s8_scale_for_conformance.def_lowering("tile")
    def _s8_scale_lowering(*_args, **_kwargs):
        return "tile.custom.s8_scale"

    np.testing.assert_allclose(s8_scale_for_conformance(np.array([2.0])), [4.0])
    assert s8_scale_for_conformance.lower("tile") == "tile.custom.s8_scale"

    clipped, total_norm = ts.optim.clip_grad_norm({"w": np.ones_like(params["w"])}, max_norm=1.0)
    stepped = ts.optim.sgd({"w": params["w"]}, clipped, lr=0.01)
    assert total_norm > 0.0
    assert stepped["w"].shape == params["w"].shape

    ckpt = tmp_path / "s8_state.npz"
    ts.checkpoint.save_state({"params": stepped, "metrics": {"loss": np.asarray(first.loss(params, batch))}}, ckpt)
    loaded = ts.checkpoint.load_state(ckpt, collections=("params",))
    np.testing.assert_allclose(loaded["params"]["w"], stepped["w"])

    artifact = ts.aot.export(first.compile_fn, *first.compile_inputs, target="cpu", path=tmp_path / "aot")
    assert artifact.runtime_artifact.graph_ir
    assert ts.aot.stablehlo_export(artifact)
    cache = ts.aot.compilation_cache(tmp_path / "cache")
    cache.put(artifact.metadata["cache_key"], artifact)
    assert cache.get(artifact.metadata["cache_key"]) is not None

    vocab = {"tiny": 1, "model": 2}
    assert ts.data.tokenizer_bpe(vocab).encode("tiny model") == [1, 2]


def test_s8_current_gen_reasoning_corner_cases_are_explicit():
    by_id = {spec.model_id: spec for spec in manifest()}
    qwen = by_id["tiny_qwen3_moe_decoder"]
    deepseek = by_id["tiny_deepseek_v3_mla_moe_decode"]

    q_params = qwen.init()
    q_batch = qwen.sample_batch()
    tied = ts.ops.moe(
        np.eye(3, dtype=np.float64),
        q_params["experts"],
        scores=q_batch["router_scores"],
    )
    routed = ts.ops.moe(
        np.eye(3, dtype=np.float64),
        q_params["experts"],
        route=q_batch["route"],
    )
    np.testing.assert_allclose(tied, routed)
    assert np.bincount(q_batch["route"], minlength=q_params["experts"].shape[0]).tolist() == [3, 0, 0, 0]
    assert np.all(np.isfinite(qwen.forward(q_params, q_batch)))

    dispatched = ts.ops.moe_dispatch(q_batch["router_scores"], q_batch["route"])
    combined = ts.ops.moe_combine(np.stack([dispatched, dispatched]), q_batch["route"], reduce="mean")
    np.testing.assert_allclose(combined, dispatched)

    d_params = deepseek.init()
    d_batch = deepseek.sample_batch()
    y = deepseek.forward(d_params, d_batch)
    assert y.shape == deepseek.expected_output_shape
    assert np.all(np.isfinite(y))
    assert np.isfinite(deepseek.loss(d_params, d_batch))

    for name in [
        "gqa_attention",
        "mla_decode_fused",
        "deepseek_sparse_attention",
        "moe",
        "moe_dispatch",
        "moe_combine",
        "silu_mul",
        "rmsnorm_safe",
        "cross_entropy_loss",
    ]:
        entry = coverage_for(name)
        assert entry.status in {"partial", "complete"}
        assert entry.contract_status["tests"] == "complete"
    assert callable(ts.nn.swiglu)


def test_s8_compiler_artifacts_for_foundation_targets():
    representative = {
        "tiny_diffusion_dit",
        "tiny_xlstm_recurrent",
        "tiny_linformer_cosformer",
        "tiny_titans_atlas_memory",
        "tiny_qwen3_moe_decoder",
    }
    for spec in manifest():
        if spec.model_id not in representative:
            continue
        for target in FOUNDATION_TARGETS:
            compiled = jit(spec.compile_fn, target=target)
            artifact = compiled.runtime_artifact()
            assert artifact.graph_ir
            assert artifact.schedule_ir
            assert artifact.tile_ir
            assert artifact.target_ir
            assert artifact.metadata["artifact_hashes"]["graph"]
            if target in {"cuda", "rocm"}:
                assert artifact.metadata["runtime_status"] == "artifact_only"
            elif target == "x86":
                assert artifact.metadata["runtime_status"] == "ready"
