from __future__ import annotations

import numpy as np
from pathlib import Path

import tessera as ts
from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.matmul_pipeline import ReferenceKVCache, build_cpu_plan


def test_public_aliases_lower_to_canonical_graph_dialect_names():
    @ts.jit
    def aliased_gemm(A, B):
        return ts.ops.gemm(A, B)

    @ts.jit
    def aliased_conv(X, W):
        return ts.ops.conv2d(X, W, stride=1, padding=0)

    gemm_ir = aliased_gemm.ir_text()
    conv_ir = aliased_conv.ir_text()
    assert "tessera.matmul" in gemm_ir
    assert "tessera.gemm" not in gemm_ir
    assert "tessera.conv2d_nhwc" in conv_ir
    assert "tessera.conv2d(" not in conv_ir


def test_cpu_artifacts_use_graph_schedule_tile_target_spine_names():
    @ts.jit
    def attention(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)

    artifacts = {artifact.level: artifact.text for artifact in attention.lowering_artifacts()}

    assert "tessera.flash_attn" in artifacts["graph"]
    assert "schedule.pipeline.region" in artifacts["schedule"]
    assert "schedule.prefetch" in artifacts["schedule"]
    assert "tile.async_copy" in artifacts["tile"]
    assert "tessera.attn.online_softmax" in artifacts["tile"]
    assert "tile.wait_async" in artifacts["tile"]
    assert "tessera.cpu.flash_attn" in artifacts["target"]


def test_textual_kv_cache_path_emits_schedule_and_tile_contracts():
    source = """
    module decode {
      func step(Cache: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>) -> tensor<?xfp32> {
        C = op.kv_cache_append(Cache, K, V);
        P = op.kv_cache_prune(C) @{max_entries = 1};
        return P;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    plan = build_cpu_plan(module)

    assert plan is not None
    cache = ReferenceKVCache()
    key = np.ones((1, 2), dtype=np.float32)
    value = np.ones((1, 2), dtype=np.float32) * 2
    out = plan.execute((cache, key, value), {}, ["Cache", "K", "V"])

    assert isinstance(out, ReferenceKVCache)
    assert len(out.keys) == 1
    assert "schedule.prefetch" in plan.schedule_ir
    assert "tile.kv_cache" in plan.tile_ir


def test_ods_contracts_cover_graph_schedule_and_tile_spine():
    root = Path(__file__).resolve().parents[2]
    graph = (root / "src/compiler/ir/TesseraOps.td").read_text(encoding="utf-8")
    schedule = (
        root / "src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td"
    ).read_text(encoding="utf-8")
    tile = (root / "src/compiler/programming_model/ir/tile/TileMemoryOps.td").read_text(
        encoding="utf-8"
    )

    for op in ["layer_norm", "dropout", "all_reduce", "fft", "spectral_conv", "adam"]:
        assert f'"{op}"' in graph
    for op in ["tile", "warp", "optimizer_shard"]:
        assert f'"{op}"' in schedule
    for op in ["mma", "kv_cache", "tmem.alloc", "mma.tcgen05"]:
        assert f'"{op}"' in tile
