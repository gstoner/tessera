"""P2/P3 Python regressions from the full-source code review.

  * EffectLattice._cache — keyed by id(fn); a GC'd function's id could be
    reused → stale effect. Now a WeakKeyDictionary keyed by the fn object.  [FIX]
  * flywheel.load_corpus — AutotuneRecord(**d) crashed on an extra/renamed key;
    now filters to known fields.  [FIX]
  * MemoryShardSpec KEY_HASH — hashed raw bytes, so -0.0/+0.0 (and different
    shapes/dtypes with equal bytes) could land on different shards. Now
    canonicalizes -0.0 and folds dtype+shape.  [FIX]
  * aot.CompilationCache — put/get/invalidate joined a caller key under root
    with no traversal check (invalidate deletes!). Now rejects escaping keys. [FIX]
  * dflash_io.load_safetensors — no header-length / data-offset bounds checks
    (crafted file → mis-mapped bytes). Now validated.  [FIX]
"""

from __future__ import annotations

import json
import struct
import weakref

import numpy as np
import pytest

from tessera import dflash_io
from tessera.aot import CompilationCache
from tessera.compiler import flywheel
from tessera.compiler.effects import EffectLattice
from tessera.sharding import MemoryMode, MemoryShardSpec, NamedMesh


# ── EffectLattice cache [FIX] ───────────────────────────────────────────────
def test_effect_cache_is_weak_and_handles_non_weakreferenceable():
    el = EffectLattice()
    assert isinstance(el._cache, weakref.WeakKeyDictionary)
    # builtins are not weak-referenceable — infer() must not crash.
    el.infer(len)

    def pure_fn(x):
        return x + 1

    first = el.infer(pure_fn)
    assert el.infer(pure_fn) == first  # cached, consistent


# ── flywheel.load_corpus tolerates unknown keys [FIX] ───────────────────────
def test_load_corpus_tolerates_unknown_fields(tmp_path):
    rec = flywheel.AutotuneRecord(
        schema_version=1,
        op_chain="matmul",
        problem_shape={"m": 4, "n": 4, "k": 4},
        dtype="f32",
        target="cpu",
        device_id="test",
        schedule={},
        legal=True,
        violation_reason="",
        latency=None,
        achieved_tflops=None,
        roofline_predicted_ms=0.0,
        model_predicted_ms=None,
        search_method="manual",
    )
    path = str(tmp_path / "corpus.json")
    flywheel.save_corpus([rec], path)

    rows = json.loads(open(path).read())
    rows[0]["unknown_future_field"] = 123  # simulate a newer schema
    open(path, "w").write(json.dumps(rows))

    loaded = flywheel.load_corpus(path)  # must not raise
    assert len(loaded) == 1


# ── KEY_HASH canonicalization [FIX] ─────────────────────────────────────────
def test_key_hash_collapses_signed_zero():
    mesh = NamedMesh(axis_names=("memory",), shape=(8,))
    spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.KEY_HASH)
    assert spec.shard_owner(np.array([-0.0]), mesh) == spec.shard_owner(np.array([0.0]), mesh)


# ── CompilationCache path-traversal guard [FIX] ─────────────────────────────
@pytest.mark.parametrize("bad", ["../evil", "../../etc/passwd", "/abs/escape"])
def test_compilation_cache_rejects_traversal_keys(tmp_path, bad):
    cache = CompilationCache(tmp_path)
    with pytest.raises(ValueError):
        cache.get(bad)
    with pytest.raises(ValueError):
        cache.invalidate(bad)


def test_compilation_cache_normal_key_ok(tmp_path):
    cache = CompilationCache(tmp_path)
    assert cache.get("sha256deadbeef") is None  # resolves cleanly, just absent


# ── safetensors bounds validation [FIX] ─────────────────────────────────────
def test_load_safetensors_rejects_oversized_header(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", 10**9) + b"{}")  # header len >> file
    with pytest.raises(ValueError, match="header length"):
        dflash_io.load_safetensors(p)


def test_load_safetensors_rejects_out_of_bounds_offsets(tmp_path):
    header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 4096]}}).encode()
    p = tmp_path / "bad2.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="out of bounds"):
        dflash_io.load_safetensors(p)
