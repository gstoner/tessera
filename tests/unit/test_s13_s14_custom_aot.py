"""S13 custom primitive and S14 AOT/cache coverage."""

from __future__ import annotations

import numpy as np

import tessera as ts


def top_level_aot_add_one(x):
    return x + 1


def test_custom_primitive_vjp_composes_under_grad_vmap():
    @ts.custom_primitive("s13_square_plus_one")
    def square_plus_one(x):
        return x * x + 1.0

    @square_plus_one.def_vjp
    def square_plus_one_vjp(dout, x, **_):
        return (dout * 2.0 * x,)

    def loss(x):
        y = square_plus_one(x)
        return ts.ops.reduce(y, op="sum")

    xs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    per_example = ts.autodiff.vmap(ts.autodiff.grad(loss))(xs)
    np.testing.assert_allclose(per_example, 2.0 * xs)

    entry = ts.ops.registry.get("s13_square_plus_one")
    assert entry is not None
    assert entry.metadata["custom_primitive"] is True


def test_custom_rules_lowering_batching_and_opaque_call_metadata():
    @ts.custom_primitive("s13_shift", effect="pure")
    def shift(x, amount=1.0):
        return x + amount

    @shift.def_shape_rule
    def shift_shape(x_shape, **_):
        return x_shape

    @shift.def_dtype_rule
    def shift_dtype(x_dtype, **_):
        return x_dtype

    @shift.def_batching
    def shift_batch(args, axis=0):
        return args, axis

    @shift.def_lowering("tile")
    def shift_tile_lowering(*_args, **_kwargs):
        return "tile.custom.shift"

    assert shift.lower("tile") == "tile.custom.shift"
    entry = ts.ops.registry.get("s13_shift")
    assert entry.metadata["has_shape_rule"]
    assert entry.metadata["has_dtype_rule"]
    assert entry.metadata["has_batching_rule"]
    assert entry.metadata["lowering_targets"] == ("tile",)
    np.testing.assert_array_equal(shift(np.array([1.0]), amount=2.0), [3.0])

    @ts.custom_call("s13_opaque_state", effect="state", abi="opaque")
    def opaque_state(x):
        return np.asarray(x) + 10

    opaque_entry = ts.ops.registry.get("s13_opaque_state")
    assert opaque_entry.metadata["opaque"] is True
    assert opaque_entry.metadata["effect"] == "state"
    assert opaque_entry.metadata["abi"] == "opaque"
    np.testing.assert_array_equal(opaque_state(np.array([1])), [11])


def test_aot_export_load_run_and_text_exports(tmp_path):
    artifact = ts.aot.export(top_level_aot_add_one, np.array([1.0]), path=tmp_path / "aot", target="cpu")
    assert artifact.metadata["target"] == "cpu"
    np.testing.assert_array_equal(artifact.run(np.array([2])), [3])

    # ``aot.load`` defaults to ``allow_pickle=False`` so untrusted
    # artifacts are safe to inspect.  This round-trip test is the
    # same-process trusted-load case, so we opt in explicitly.
    loaded = ts.aot.load(tmp_path / "aot", allow_pickle=True)
    np.testing.assert_array_equal(loaded.run(np.array([4])), [5])
    assert loaded.artifact_hash == artifact.artifact_hash

    # Default load (untrusted-safe) returns the IR/metadata bundle
    # without the picklable callable.
    safe_loaded = ts.aot.load(tmp_path / "aot")
    assert safe_loaded.artifact_hash == artifact.artifact_hash
    assert safe_loaded.fn is None, (
        "aot.load(allow_pickle=False) must NOT deserialize callable.pkl"
    )

    stablehlo = ts.aot.stablehlo_export(artifact)
    assert "stablehlo reference export" in stablehlo

    gguf_path = ts.aot.gguf_export(artifact, tmp_path / "model.gguf.json")
    assert gguf_path.exists()
    safe_path = ts.aot.safetensors_export({"w": np.array([1.0, 2.0])}, tmp_path / "weights.safetensors")
    assert safe_path.exists()
    assert safe_path.with_suffix(".safetensors.json").exists()


def test_compilation_cache_put_get_invalidate(tmp_path):
    artifact = ts.aot.export(top_level_aot_add_one, np.array([1.0]), target="cpu")
    cache = ts.aot.compilation_cache(tmp_path / "cache")
    key = artifact.metadata["cache_key"]
    cache.put(key, artifact)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.artifact_hash == artifact.artifact_hash
    cache.invalidate(key)
    assert cache.get(key) is None
