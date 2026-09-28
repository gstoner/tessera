"""Decision #11 for the non-registry autotune rows (paged-KV, conv2d, ReplaySSM).

Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (closes
``AUTOTUNE-KERNEL-IDENTITY-PAGED-KV``). The paged-KV serving rows (gfx1151 and
sm_120), the sm_120 conv2d rows and the ReplaySSM async-ring rows are timed by
their own recorders and read by their own warm-start code, not by the registry
arbiter, so they carried the toolchain pins and no code identity. They now get
the registry contract: each row stamps the identity of every route it timed,
and a warm start refuses a row whose live identities differ, fail closed.

Host-independent. The on-device halves (identities that need the FA-2 image,
the shipped GEMM library or ``tessera-nvidia-opt``) are the re-records' serve
and miss checks on Princess-Luna and The-Super-Bear.
"""
from __future__ import annotations

import json

import pytest


def _routes(values):
    from tessera.compiler.emit import autotune as at

    return {name: at.RouteIdentity(name, lambda name=name: values[name])
            for name in values}


def _record(stamps, candidates=("a", "b")):
    from tessera.compiler.emit import autotune as at

    return at.MeasureRecord(
        winner=candidates[0], latency_ms=1.0,
        candidates={name: 1.0 + i for i, name in enumerate(candidates)},
        evidence={"delegate_identities": stamps} if stamps is not None else {})


def test_route_record_matches_is_the_registry_contract():
    from tessera.compiler.emit import autotune as at

    live = {"a": {"code": "a1"}, "b": {"code": "b1"}}
    assert at.route_record_matches(_record(dict(live)), _routes(live))
    # a changed route, an unstamped row, a row stamped for fewer routes
    assert not at.route_record_matches(
        _record({"a": {"code": "a1"}, "b": {"code": "b0"}}), _routes(live))
    assert not at.route_record_matches(_record(None), _routes(live))
    assert not at.route_record_matches(_record({"a": {"code": "a1"}}), _routes(live))
    # a route that cannot be identified now, an empty identity
    assert not at.route_record_matches(
        _record(dict(live)), _routes({"a": {"code": "a1"}, "b": None}))
    assert not at.route_record_matches(
        _record({"a": {"code": "a1"}, "b": {}}), _routes({"a": {"code": "a1"}, "b": {}}))
    # the field changed: a route added or dropped since the row was timed
    assert not at.route_record_matches(
        _record(dict(live)), _routes({**live, "c": {"code": "c1"}}))
    assert not at.route_record_matches(
        _record({**live, "c": {"code": "c1"}}, ("a", "b", "c")), _routes(live))


def test_route_identities_refuses_an_unidentifiable_route():
    from tessera.compiler.emit import autotune as at

    assert at.route_identities(_routes({"a": {"code": "a1"}})) == {"a": {"code": "a1"}}

    def broken():
        raise RuntimeError("no FA-2 image on this host")

    with pytest.raises(ValueError, match="b: RuntimeError: no FA-2 image"):
        at.route_identities({"a": at.RouteIdentity("a", lambda: {"code": "a1"}),
                             "b": at.RouteIdentity("b", broken)})


def test_nvidia_paged_routes_are_the_resident_stage_source():
    """Both sm_120 paged routes run entries of the emitted resident-stage
    library: host-computable, distinct per route, and moved by the emitter."""
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda

    stamped = at.route_identities(nvidia_cuda.paged_attention_route_identities())
    assert set(stamped) == {"fused_paged_attention", "staged_paged_attention"}
    fused, staged = stamped["fused_paged_attention"], stamped["staged_paged_attention"]
    assert fused["identity"] == "emitted_source" and fused["lang"] == "cuda"
    assert fused["source_sha256"] == staged["source_sha256"]
    assert fused["route_entries"] != staged["route_entries"]


def test_nvidia_paged_identity_follows_the_emitter(monkeypatch):
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda

    before = at.route_identities(nvidia_cuda.paged_attention_route_identities())
    original = nvidia_cuda._synthesize_resident_ops_cuda
    monkeypatch.setattr(nvidia_cuda, "_synthesize_resident_ops_cuda",
                        lambda: original() + "\n// perturbed\n")
    after = at.route_identities(nvidia_cuda.paged_attention_route_identities())
    assert before["fused_paged_attention"]["source_sha256"] != \
        after["fused_paged_attention"]["source_sha256"]


def test_nvidia_conv2d_routes(monkeypatch):
    """direct / shared: resident-stage source, host-computable. im2col_tf32 also
    runs the shipped GEMM's tf32 device entry, so without that library it has
    no identity (a miss), never a partial one."""
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda

    routes = nvidia_cuda.conv2d_route_identities()
    assert set(routes) == {"direct", "shared", "im2col_tf32"}
    direct = routes["direct"].artifact_identity()
    shared = routes["shared"].artifact_identity()
    assert direct and shared and direct["route_entries"] != shared["route_entries"]
    monkeypatch.setattr(nvidia_cuda, "_gemm_runtime_path", lambda: None)
    assert routes["im2col_tf32"].artifact_identity() is None


def test_rocm_direct_route_is_the_emitted_hip_source(monkeypatch):
    """gfx1151 ``direct``: the emitted HIP paged-attention source and the
    hipcc line with the offload arch -- host-computable. ``gather_fa`` needs
    the FA-2 image of a live device; without one it is a miss."""
    from tessera.compiler.emit import rocm_hip

    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1151")
    routes = rocm_hip.rocm_paged_attention_route_identities(
        q_heads=4, kv_heads=4, head_dim=32, causal=True)
    direct = routes["direct"].artifact_identity()
    assert direct["lang"] == "hip" and "--offload-arch=gfx1151" in direct["build"]
    original = rocm_hip._synthesize_paged_attention_direct_hip
    monkeypatch.setattr(rocm_hip, "_synthesize_paged_attention_direct_hip",
                        lambda: original() + "\n// perturbed\n")
    assert routes["direct"].artifact_identity()["source_sha256"] != direct["source_sha256"]
    monkeypatch.setattr(rocm_hip, "_identity_isa", lambda: None)
    assert routes["gather_fa"].artifact_identity() is None


def test_rocm_paged_warm_start_refuses_a_changed_route(monkeypatch):
    from tessera.cache import paged_kv
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import rocm_hip
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key

    key = ("rocm:gfx1151", "rocm", "paged_kv_decode",
           bucket_key((1, 4, 4, 512, 32, 16), SpecPolicy.BUCKET), "f32",
           at.TIMING_END_TO_END)
    stamps = {"direct": {"code": "d"}, "gather_fa": {"code": "g"}}
    live = dict(stamps)
    seen: dict = {}

    def identities(**kw):
        seen.update(kw)
        return _routes(live)

    monkeypatch.setattr(rocm_hip, "rocm_paged_attention_route_identities", identities)

    def fake_load(cache=None, **kw):
        cache.put(key, at.MeasureRecord(
            winner="gather_fa", latency_ms=1.0,
            candidates={"gather_fa": 1.0, "direct": 2.0},
            evidence={"delegate_identities": dict(stamps)}, unmeasured={},
            separation={"separated": True, "margin": .5, "noise": .01,
                        "runner_up": "direct", "factor": 2.0}), fresh=True)

    monkeypatch.setattr(at, "load_corpus", fake_load)
    assert paged_kv._rocm_paged_attention_corpus_winner(4, 4, 1, 512, 32, 16) == "gather_fa"
    assert seen == {"q_heads": 4, "kv_heads": 4, "head_dim": 32, "causal": True}
    assert paged_kv._rocm_paged_attention_corpus_winner(
        4, 4, 1, 512, 32, 16, causal=False) == "gather_fa"
    assert seen["causal"] is False           # the variant is the caller's, not assumed
    live["gather_fa"] = {"code": "g-rebuilt"}
    assert paged_kv._rocm_paged_attention_corpus_winner(4, 4, 1, 512, 32, 16) is None


def test_serving_recorder_stamps_every_route(tmp_path, monkeypatch):
    """`benchmark_serving.update_d2_corpus` stamps the paged routes' and the
    ReplaySSM ring's identities; a ring it cannot identify is refused."""
    import importlib.util
    from pathlib import Path

    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "bench_serving_identity", root / "benchmarks/nvidia/benchmark_serving.py")
    bench = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(bench)
    path = tmp_path / "corpus.json"
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(path))
    calls = []

    def ring(bsz, d, n, capacity, slots):
        calls.append((bsz, d, n, capacity, slots))
        return at.RouteIdentity("async_ring", lambda: {"ring": f"{d}x{n}"})

    monkeypatch.setattr(nvidia_cuda, "ssm_replay_ring_identity", ring)
    rows = [
        {"op": "ssm_replay_decode", "shape": "1x128x64", "dtype": "f32",
         "mode": "async_ring", "tokens": 16, "async_slots": 4,
         "latency_ms": .5, "device_latency_ms": .1},
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "fused_paged_attention", "latency_ms": .2, "device_latency_ms": .1},
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "staged_paged_attention", "latency_ms": .3, "device_latency_ms": .2},
    ]
    bench.update_d2_corpus(rows)
    records = json.loads(path.read_text())["records"]
    assert (1, 128, 64, 17, 4) in calls
    for record in records:
        stamped = record["evidence"]["delegate_identities"]
        assert set(stamped) == set(record["candidates"])
        if record["op"] == "ssm_replay_decode":
            assert stamped == {"async_ring": {"ring": "128x64"}}
    monkeypatch.setattr(nvidia_cuda, "ssm_replay_ring_identity",
                        lambda *a: at.RouteIdentity("async_ring", lambda: None))
    with pytest.raises(ValueError, match="async_ring"):
        bench.update_d2_corpus(rows)


def test_isolated_route_resources_are_attributed_to_their_route():
    """AUTOTUNE-SM120-ROUTE-RESOURCES: one ncu report per route, all of its
    kernels attributed to that route -- the tf32 and fp8 builds of one lane
    share a kernel name and must not collapse into one entry."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "build_manifest_iso", root / "benchmarks/nvidia/build_test5_resource_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    base = {"schema": "tessera.nvidia.route-resources.v1", "sources": [],
            "routes": {"nvidia_mma_fused": ["sha256:f16"]},
            "details": {"nvidia_mma_fused": [{"kernel": "k", "resource_fingerprint": "sha256:f16"}]}}
    row = {"kernel": "tessera_nvidia_mma_fused_kernel"}
    out = mod.add_isolated_routes(base, {
        "nvidia_mma_fused_tf32": {"source": "a.ncu-repz", "source_sha256": "aa",
                                  "rows": [{**row, "resource_fingerprint": "sha256:tf32"}]},
        "nvidia_mma_fused_fp8_e4m3": {"rows": [{**row, "resource_fingerprint": "sha256:e4"}]},
    })
    assert out["routes"]["nvidia_mma_fused_tf32"] == ["sha256:tf32"]
    assert out["routes"]["nvidia_mma_fused_fp8_e4m3"] == ["sha256:e4"]
    assert out["routes"]["nvidia_mma_fused"] == ["sha256:f16"]
    assert out["sources"] == [{"name": "a.ncu-repz", "sha256": "aa",
                               "route": "nvidia_mma_fused_tf32"}]
    assert base["routes"] == {"nvidia_mma_fused": ["sha256:f16"]}   # not mutated
    with pytest.raises(ValueError, match="holds no kernel"):
        mod.add_isolated_routes(base, {"x": {"rows": []}})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        mod.add_isolated_routes(base, {"nvidia_mma_fused": {"rows": [
            {**row, "resource_fingerprint": "sha256:z"}]}})
