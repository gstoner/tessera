"""Bounded saved-LSE JVP native lowering and checked scalar ABI agreement."""
from dataclasses import replace
import inspect
import json
import re
from types import SimpleNamespace

import pytest

from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage, _decode_image
from tessera.compiler.native_gpu_tensor import GridProduct, IndexSpec
from tessera.compiler.native_storage_contract import generate_tensor_binding
from tessera.compiler.nvidia_native import _checkpoint_identity
from tessera.compiler.attention_shape_contract import DYNAMIC_DIM
from tessera.compiler.scheduled_matmul import run_tessera_opt
from test_bounded_attention_jvp_graph import compiler, graph

pytestmark = pytest.mark.compiler_route


def lowered(query=True, key=True, wrt=(0, 1, 2)):
    tool = compiler()
    paired = run_tessera_opt(tool, graph(query, key, wrt),
                            "--tessera-autodiff-forward=export-attention-jvp")
    schedule = run_tessera_opt(tool, paired, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    return tool, paired, schedule, tile


def manifest(tile):
    fields = re.findall(r'tessera.native_tensor_contract = "((?:\\.|[^"\\])*)"', tile)
    assert len(fields) == 1
    return json.loads(_decode_image(fields[0]))


@pytest.mark.parametrize("query,key", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("wrt", [(0,), (1,), (2,), (2, 0, 1)])
def test_bounded_lowering_preserves_actual_extent_abi_and_generation(query, key, wrt):
    tool, _, schedule, tile = lowered(query, key, wrt)
    data = manifest(tile)
    assert data["schema"] == 2
    assert data["grid"] == [{"product": [1, 2, "query_size"]}, 1, 1]
    assert data["block"] == [128, 1, 1]
    assert data["arguments"][-2:] == [
        {"kind": "index", "name": "query_size", "minimum": 1 if query else 4,
         "maximum": 9 if query else 4},
        {"kind": "index", "name": "key_size", "minimum": 1 if key else 6,
         "maximum": 11 if key else 6}]
    symbolic = (1, 2, 1, DYNAMIC_DIM if query else 4, DYNAMIC_DIM if key else 6, 8, 8)
    capacities = (1, 2, 1, 9 if query else 4, 11 if key else 6, 8, 8)
    identity = _checkpoint_identity(symbolic, 8**-.5, True, shape_bounds=capacities)
    assert f'tessera.attention_checkpoint_identity = "{identity}"' in tile
    assert 'shape_policy = "bounded_sequences_v1"' in schedule
    assert "tensor.generate" not in tile and "tessera_attn.checkpoint_jvp" not in tile
    assert "arith.select" in tile  # non-underflowing end-aligned causal gap
    # Reparse registered native storage operations, then prove checked shared
    # arena allocation before downstream target lowering.
    verified = run_tessera_opt(tool, tile, "--tessera-tile-buffer-reuse")
    arena = run_tessera_opt(tool, verified, "--tessera-tile-buffer-arena")
    assert "tile.dynamic_shared_size" in arena
    assert "__tessera_shared_bytes_attention_jvp_saved_lse_jvp" in arena


@pytest.mark.parametrize("sq,sk", [(1, 1), (3, 7), (9, 11)])
def test_native_manifest_binds_actual_shapes_and_exact_row_grid(sq, sk):
    _, _, _, tile = lowered(wrt=(0,))
    data = manifest(tile)
    abi = tuple("pointer" if row["kind"] == "tensor" else "index"
                for row in data["arguments"])
    package = NativeGPUStoragePackage("nvidia", "sm_120", "saved_lse_jvp",
        "size", abi, tile, b"image", b"host", "c"*64, "d"*64, "")
    package = replace(package, binding_digest=package._digest())
    names = [row["name"] for row in data["arguments"]]
    signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                   for name in names])
    call = generate_tensor_binding(package, signature)
    assert call.grid == (GridProduct((1, 2, "query_size")), 1, 1)
    values = []
    indices = {"scratch": 128, "query_size": sq, "key_size": sk}
    for position, spec in enumerate(call.specs):
        if isinstance(spec, IndexSpec):
            values.append(indices[spec.name])
        else:
            shape = tuple(indices[d] if isinstance(d, str) else d for d in spec.shape)
            values.append(SimpleNamespace(__cuda_array_interface__={
                "version": 3, "shape": shape, "typestr": "<f4",
                "data": (4096 + position * 65536, False), "strides": None}))
    raw, _, grid, _, _ = call.prepare(*values)
    assert raw[-3:] == (128, sq, sk)
    assert grid == (2 * sq, 1, 1)


def test_bounded_schedule_replay_rejects_changed_capacity():
    tool, _, schedule, _ = lowered()
    changed = schedule.replace("1, 2, 1, 9, 11, 8, 8", "1, 2, 1, 10, 11, 8, 8")
    assert changed != schedule
    with pytest.raises(RuntimeError, match="changed after hashing"):
        run_tessera_opt(tool, changed, "--tessera-schedule-to-tile")


def test_native_export_rejects_unrelated_symbolic_region():
    tool, paired, _, _ = lowered(wrt=(0,))
    changed = re.sub(r'(arith.constant )0\.000000e\+00( : f32)', r'\g<1>1.000000e+00\2', paired, count=1)
    assert changed != paired
    with pytest.raises(RuntimeError, match="inactive region|isolated paired"):
        run_tessera_opt(tool, changed, "--tessera-graph-to-schedule")


def test_dynamic_direct_product_uses_same_checked_native_manifest():
    from tessera.compiler.native_attention_jvp import source
    symbolic = (1, 2, 1, DYNAMIC_DIM, DYNAMIC_DIM, 4, 3)
    tile = source(symbolic, .5, True, compiler=compiler(),
                  shape_bounds=(1, 2, 1, 9, 11, 4, 3))
    assert manifest(tile)["schema"] == 2
    assert 'shape_policy = "bounded_sequences_v1"' in tile


@pytest.mark.parametrize("bounds", [(1, 2, 1, 9, 11, 8, 8), (1, 2, 1, 10, 11, 8, 8)])
def test_generated_adapter_retains_symbolic_policy_and_checks_capacity(monkeypatch, bounds):
    from tessera.compiler.native_attention_jvp import materialize_generated
    monkeypatch.setattr("tessera.compiler.native_attention_jvp.build_native_gpu_storage",
                        lambda text, **kwargs: text)
    symbolic = (1, 2, 1, DYNAMIC_DIM, DYNAMIC_DIM, 8, 8)
    if bounds[3] == 9:
        tile = materialize_generated(graph(), symbolic, 8**-.5, True,
            compiler=compiler(), llvm_bin="/unused", shape_bounds=bounds)
        assert manifest(tile)["schema"] == 2
    else:
        with pytest.raises(ValueError, match="resident forward generation"):
            materialize_generated(graph(), symbolic, 8**-.5, True,
                compiler=compiler(), llvm_bin="/unused", shape_bounds=bounds)


def test_generated_forward_identity_binds_capacity(monkeypatch):
    from tessera.compiler import nvidia_native as native
    symbolic = (1, 2, 1, DYNAMIC_DIM, DYNAMIC_DIM, 8, 8)
    bounds = (1, 2, 1, 9, 11, 8, 8)
    scheduled = SimpleNamespace(dims=symbolic, shape_bounds=bounds,
        scale=8**-.5, causal=True, bias=False, bias_shape=())
    monkeypatch.setattr("tessera.compiler.scheduled_checkpoint.lower_generated_checkpoint",
                        lambda source: scheduled)
    package = object()
    monkeypatch.setattr(native, "package_scheduled_checkpoint", lambda *args, **kwargs: package)
    result = native.package_generated_attention_forward_checkpoint("source", pipeline_name="unused")
    assert result.forward is package
    assert result.contract_digest == _checkpoint_identity(symbolic, 8**-.5, True, shape_bounds=bounds)
