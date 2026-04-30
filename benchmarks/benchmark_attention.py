"""
benchmark_attention.py — Flash attention benchmark sweep (Phase 6)

Sweeps (batch, heads, seq_len, head_dim) and reports tokens/sec and MFU.

Usage::

    from benchmarks.benchmark_attention import FlashAttnBenchmark

    bench = FlashAttnBenchmark(causal=True)
    results = bench.run()
    print(bench.report(results))
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:
    from benchmarks.compiler_support import compiler_flash_attention_ir
except ImportError:
    from compiler_support import compiler_flash_attention_ir


@dataclass
class AttnConfig:
    """Single flash-attention problem configuration."""
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool = True
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        for name, v in [("batch", self.batch), ("heads", self.heads),
                         ("seq_len", self.seq_len), ("head_dim", self.head_dim)]:
            if v <= 0:
                raise ValueError(f"{name}={v} must be > 0")

    def flops(self) -> int:
        """FLOPs for QK^T + softmax + AV (approximate, causal halves QK^T)."""
        S = self.seq_len
        D = self.head_dim
        B = self.batch
        H = self.heads
        # QK^T: 2*B*H*S*S*D (or half if causal)
        qk_flops = 2 * B * H * S * S * D
        if self.causal:
            qk_flops //= 2
        # AV:  2*B*H*S*S*D
        av_flops = 2 * B * H * S * S * D
        return qk_flops + av_flops

    def tokens(self) -> int:
        return self.batch * self.seq_len

    def bytes_accessed(self) -> int:
        bpe = 2  # bf16 / fp16
        # Q, K, V, O each (B, H, S, D)
        per_tensor = self.batch * self.heads * self.seq_len * self.head_dim * bpe
        return 4 * per_tensor


@dataclass
class AttnResult:
    config: AttnConfig
    latency_ms: float
    tokens_per_sec: float
    tflops: float
    mfu: float
    compiler_path: str = "roofline_model"
    compiler_lowering: str = ""
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"AttnResult(B={self.config.batch}, H={self.config.heads}, "
            f"S={self.config.seq_len}, D={self.config.head_dim}, "
            f"{self.tokens_per_sec/1e3:.1f}K tok/s, {self.tflops:.1f} TFLOPs)"
        )


class FlashAttnBenchmark:
    """
    Flash-attention benchmark sweeping B/H/S/D configurations.

    Parameters
    ----------
    causal : bool
        Use causal (autoregressive) masking.
    dtype : str
        Data type.
    peak_tflops : float
        Hardware peak (TFLOPs/s).
    """

    DEFAULT_CONFIGS: List[Tuple[int, int, int, int]] = [
        # (batch, heads, seq_len, head_dim)
        (1,  32, 2048, 128),
        (4,  32, 2048, 128),
        (1,  32, 4096, 128),
        (4,  16, 4096,  64),
        (8,   8, 1024, 128),
        (1,  64, 8192,  64),
    ]

    def __init__(
        self,
        causal: bool = True,
        dtype: str = "bf16",
        peak_tflops: float = 312.0,
        peak_membw_gbps: float = 2_000.0,
    ) -> None:
        self.causal = causal
        self.dtype = dtype
        self.peak_tflops = peak_tflops
        self.peak_membw_gbps = peak_membw_gbps

    def _roofline_latency_ms(self, cfg: AttnConfig) -> float:
        compute_ms = cfg.flops() / (self.peak_tflops * 1e12) * 1e3
        memory_ms = cfg.bytes_accessed() / (self.peak_membw_gbps * 1e9) * 1e3
        return max(compute_ms, memory_ms) * 1.05

    def _benchmark_one(self, cfg: AttnConfig, *, emit_compiler_ir: bool = False) -> AttnResult:
        latency_ms = self._roofline_latency_ms(cfg)
        tflops = cfg.flops() / (latency_ms * 1e-3) / 1e12
        tokens_per_sec = cfg.tokens() / (latency_ms * 1e-3)
        mfu = tflops / self.peak_tflops
        compiler_path = "roofline_model"
        compiler_lowering = ""
        if emit_compiler_ir:
            info = compiler_flash_attention_ir()
            if info.get("available"):
                compiler_path = "tessera_graph_ir"
                if info.get("uses_compiled_path"):
                    compiler_path = "tessera_jit_cpu"
                compiler_lowering = str(info.get("lowering", ""))
            else:
                compiler_path = "compiler_unavailable"
        return AttnResult(config=cfg, latency_ms=latency_ms,
                          tokens_per_sec=tokens_per_sec,
                          tflops=tflops, mfu=mfu,
                          compiler_path=compiler_path,
                          compiler_lowering=compiler_lowering)

    def run(
        self,
        configs: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        emit_compiler_ir: bool = False,
    ) -> List[AttnResult]:
        if configs is None:
            configs = self.DEFAULT_CONFIGS
        results = []
        for B, H, S, D in configs:
            cfg = AttnConfig(B, H, S, D, causal=self.causal, dtype=self.dtype)
            results.append(self._benchmark_one(cfg, emit_compiler_ir=emit_compiler_ir))
        return results

    def run_single(self, batch: int, heads: int, seq_len: int,
                   head_dim: int) -> AttnResult:
        cfg = AttnConfig(batch, heads, seq_len, head_dim,
                         causal=self.causal, dtype=self.dtype)
        return self._benchmark_one(cfg)

    @staticmethod
    def report(results: List[AttnResult]) -> str:
        lines = [
            f"{'B':>4} {'H':>4} {'S':>6} {'D':>4} | "
            f"{'Lat(ms)':>9} {'Tok/s(K)':>10} {'TFLOPs':>8} {'MFU%':>7}"
        ]
        lines.append("-" * 56)
        for r in results:
            c = r.config
            lines.append(
                f"{c.batch:>4} {c.heads:>4} {c.seq_len:>6} {c.head_dim:>4} | "
                f"{r.latency_ms:>9.3f} {r.tokens_per_sec/1e3:>10.1f} "
                f"{r.tflops:>8.2f} {r.mfu*100:>6.1f}%"
            )
        return "\n".join(lines)

    @staticmethod
    def to_json(results: List[AttnResult], path: str) -> None:
        data = [{
            "batch": r.config.batch, "heads": r.config.heads,
            "seq_len": r.config.seq_len, "head_dim": r.config.head_dim,
            "causal": r.config.causal, "dtype": r.config.dtype,
            "latency_ms": r.latency_ms,
            "tokens_per_sec": r.tokens_per_sec,
            "tflops": r.tflops, "mfu": r.mfu,
            "compiler_path": r.compiler_path,
            "compiler_lowering": r.compiler_lowering,
            "timestamp": r.timestamp,
        } for r in results]
        with open(path, "w") as f:
            json.dump({"benchmark": "flash_attn", "results": data}, f, indent=2)
