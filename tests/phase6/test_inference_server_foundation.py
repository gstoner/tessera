from __future__ import annotations

import json
import pathlib

import pytest

from tessera import server


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _write_pkg(tmp_path: pathlib.Path) -> pathlib.Path:
    pkg = tmp_path / "model.tspkg"
    pkg.mkdir()
    manifest = {
        "name": "gpt-oss-7b",
        "version": "1.3.2",
        "entry": "main.decode_graph",
        "mesh": {"tp": 2, "pp": 1, "dp": 1},
        "dtypes": ["bf16", "fp16", "fp32"],
        "kv_cache": {"pages": 256, "page_size": "2MiB", "swap": "host_pinned"},
        "autotune": {"required": True, "min_arch": "sm90"},
        "compat": {"api": "openai-v1"},
    }
    (pkg / "manifest.yaml").write_text(json.dumps(manifest))
    return pkg


def test_load_package_validates_manifest_and_layout(tmp_path):
    pkg_path = _write_pkg(tmp_path)

    pkg = server.load_package(pkg_path)

    assert pkg.manifest.name == "gpt-oss-7b"
    assert pkg.entry == "main.decode_graph"
    assert pkg.manifest.kv_cache.page_size_bytes == 2 * 1024 * 1024


def test_manifest_rejects_missing_required_fields():
    with pytest.raises(server.ServerConfigError, match="requires"):
        server.ModelManifest.from_dict({
            "name": "",
            "version": "1",
            "entry": "",
            "mesh": {"tp": 1},
            "dtypes": ["bf16"],
            "kv_cache": {"pages": 1, "page_size": "1MiB"},
        })


def test_runtime_capabilities_guard_dtype_support():
    caps = server.capabilities(backend="ptx", arch="sm90", fp8=True, int8=True)

    assert caps.supports_dtype("fp8_e4m3")
    assert caps.supports_dtype("int8")
    assert caps.supports_dtype("bf16")


def test_kv_cache_manager_tracks_capacity_and_hit_rate():
    cfg = server.KVCacheConfig(pages=4, page_size="1MiB", swap="host_pinned")
    kv = server.KVCacheManager(cfg)

    kv.allocate(3)
    assert kv.allocated_pages == 3
    assert kv.capacity_bytes == 4 * 1024 * 1024
    assert kv.hit_rate(hits=9, misses=1) == pytest.approx(0.9)
    kv.evict(2)
    assert kv.allocated_pages == 1

    with pytest.raises(MemoryError):
        kv.allocate(10)


def test_scheduler_session_generates_stream_deltas():
    sched = server.scheduler(policy="continuous_batch", target_p99_ms=150, max_batch_tokens=2048)

    with sched.session(model="gpt-oss-7b") as sess:
        deltas = list(sess.generate([{"role": "user", "content": "Hello"}], max_tokens=3))

    assert len(deltas) == 3
    assert deltas[0]["model"] == "gpt-oss-7b"


def test_app_registers_models_routes_health_and_metrics():
    app = server.App(config="tis.yaml")

    @app.model("/models/gpt")
    def load_model(pkg):
        return pkg

    @app.route("/v1/chat/completions", stream=True)
    def chat(req):
        return req

    health = app.healthz()
    metrics = app.metrics()

    assert "/models/gpt" in health["models"]
    assert "/v1/chat/completions" in health["routes"]
    assert metrics["tis_models_loaded"] == 1.0
    assert app.routes["/v1/chat/completions"].stream is True


def test_invalid_scheduler_policy_rejected():
    with pytest.raises(server.ServerConfigError, match="scheduler policy"):
        server.scheduler(policy="round_robin")


def test_inference_server_guide_is_registered_and_covers_required_topics():
    guide = (ROOT / "docs/guides/Tessera_Inference_Server_Guide.md").read_text()
    readme = (ROOT / "docs/README.md").read_text()

    for needle in [
        "Model Packaging",
        "Runtime Backends And Placement",
        "Scheduling And Batching",
        "KV Cache Management",
        "Quantization And Formats",
        "OpenAI-compatible",
        "Failure Modes And Recovery",
    ]:
        assert needle in guide
    assert "Tessera_Inference_Server_Guide.md" in readme
