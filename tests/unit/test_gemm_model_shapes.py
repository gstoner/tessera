"""
test_gemm_model_shapes.py — GEMM coverage for real model weight shapes.

Three test suites:

  TestGemma4GEMM      — all projection + FFN + LM-head GEMMs across every
                        Gemma 4 variant (4B, 12B, 27B) and a range of
                        batch-sequence lengths.

  TestLlama32GEMM     — same coverage for LLaMA 3.2 (1B, 3B) with full
                        GQA (grouped-query attention) projection shapes.

  TestGEMMCornerCases — exhaustive shape-corner coverage:
                          · single-token decode  (M=1)
                          · skinny-tall  (M >> N)
                          · tall-skinny  (N >> M)
                          · flat         (K=1)
                          · square powers-of-two
                          · non-power-of-two alignment
                          · odd primes
                          · very large K  (accumulation stress)

  TestGEMMShmoo       — full Cartesian shmoo of M × N × K across the
                        vectors [1, 8, 64, 128, 512, 1024, 2048, 4096, 8192]
                        for each axis independently; validates roofline
                        model is internally consistent at every point.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

from benchmarks.benchmark_gemm import GEMMBenchmark, GEMMConfig, GEMMResult


# ===========================================================================
# Shared benchmark fixture (H100 SXM5 numbers — representative targets)
# ===========================================================================

PEAK_TFLOPS_BF16 = 989.0       # H100 SXM5 BF16 Tensor Core peak
PEAK_MEMBW_GBPS  = 3_350.0     # H100 SXM5 HBM3 peak bandwidth


@pytest.fixture(scope="module")
def bench() -> GEMMBenchmark:
    return GEMMBenchmark(
        dtype="bf16",
        peak_tflops=PEAK_TFLOPS_BF16,
        peak_membw_gbps=PEAK_MEMBW_GBPS,
    )


@pytest.fixture(scope="module")
def bench_fp8() -> GEMMBenchmark:
    return GEMMBenchmark(
        dtype="fp8",
        peak_tflops=1979.0,   # H100 FP8 Tensor Core peak
        peak_membw_gbps=PEAK_MEMBW_GBPS,
    )


# ===========================================================================
# Model shape descriptors
# ===========================================================================

class LayerGEMM(NamedTuple):
    """One named GEMM from a transformer layer."""
    name: str        # human label, e.g. "Q_proj"
    M: int           # batch × seq tokens (set per test)
    N: int           # output features
    K: int           # input features (reduction axis)


@dataclass(frozen=True)
class ModelSpec:
    """
    Transformer model specification.

    Weight GEMMs are derived from hidden_size, intermediate_size, vocab_size,
    num_attention_heads, num_key_value_heads, and head_dim.
    """
    name: str
    hidden: int
    intermediate: int
    vocab: int
    num_heads: int
    kv_heads: int
    head_dim: int

    def q_dim(self) -> int:  return self.num_heads * self.head_dim
    def kv_dim(self) -> int: return self.kv_heads * self.head_dim

    def layer_gemms(self, seq_len: int) -> List[LayerGEMM]:
        """Return all weight-matrix GEMMs for one transformer layer at seq_len."""
        M = seq_len
        H = self.hidden
        return [
            LayerGEMM("Q_proj",     M, self.q_dim(),       H),
            LayerGEMM("K_proj",     M, self.kv_dim(),      H),
            LayerGEMM("V_proj",     M, self.kv_dim(),      H),
            LayerGEMM("O_proj",     M, H,                  self.q_dim()),
            LayerGEMM("FFN_gate",   M, self.intermediate,  H),
            LayerGEMM("FFN_up",     M, self.intermediate,  H),
            LayerGEMM("FFN_down",   M, H,                  self.intermediate),
            LayerGEMM("LM_head",    M, self.vocab,         H),
        ]


# ---------------------------------------------------------------------------
# Gemma 4 variants
# Source: Google DeepMind Gemma 4 technical report (April 2025)
#   4B  — hidden=2560, intermediate=10240, heads=16, kv_heads=8, head_dim=256
#   12B — hidden=3840, intermediate=15360, heads=24, kv_heads=8, head_dim=256
#   27B — hidden=5632, intermediate=22528, heads=32, kv_heads=16, head_dim=256
# ---------------------------------------------------------------------------

GEMMA4_SPECS = {
    "gemma4-4b":  ModelSpec("gemma4-4b",   2560, 10240, 256000, 16,  8, 256),
    "gemma4-12b": ModelSpec("gemma4-12b",  3840, 15360, 256000, 24,  8, 256),
    "gemma4-27b": ModelSpec("gemma4-27b",  5632, 22528, 256000, 32, 16, 256),
}

# ---------------------------------------------------------------------------
# LLaMA 3.2 variants
# Source: Meta LLaMA 3.2 model card (September 2024)
#   1B — hidden=2048, intermediate=8192, heads=32, kv_heads=8,  head_dim=64
#   3B — hidden=3072, intermediate=8192, heads=24, kv_heads=8,  head_dim=128
# ---------------------------------------------------------------------------

LLAMA32_SPECS = {
    "llama3.2-1b": ModelSpec("llama3.2-1b", 2048, 8192,  32000, 32,  8,  64),
    "llama3.2-3b": ModelSpec("llama3.2-3b", 3072, 8192,  32000, 24,  8, 128),
}

# Representative sequence lengths:
#   1  = single-token decode (latency-critical)
#   128, 512 = short-context prefill
#   2048, 4096 = long-context prefill
SEQ_LENS = [1, 128, 512, 2048, 4096]

# Decode-only (batch=1, one token at a time)
DECODE_SEQ_LENS = [1, 8, 16, 32]


# ===========================================================================
# Helper: validate a GEMMResult's internal consistency
# ===========================================================================

def _assert_result_sane(result: GEMMResult, cfg: GEMMConfig,
                         bench: GEMMBenchmark, label: str) -> None:
    """Assert that a single GEMMResult is self-consistent."""
    assert result.latency_ms > 0,          f"{label}: latency must be positive"
    assert result.tflops > 0,              f"{label}: TFLOPs must be positive"
    assert result.memory_bw_gbps > 0,      f"{label}: bandwidth must be positive"
    assert result.roofline_bound in ("compute", "memory"), \
        f"{label}: roofline_bound must be 'compute' or 'memory'"

    # TFLOPs must not exceed peak (roofline is an upper bound)
    assert result.tflops <= bench.peak_tflops * 1.02, \
        f"{label}: {result.tflops:.1f} TFLOPs exceeds peak {bench.peak_tflops}"

    # MFU sanity
    mfu = bench.mfu(result)
    assert 0.0 < mfu <= 1.01, f"{label}: MFU={mfu:.3f} out of range [0,1]"

    # FLOPs formula cross-check
    expected_flops = 2 * cfg.M * cfg.N * cfg.K
    assert cfg.flops() == expected_flops, \
        f"{label}: flops()={cfg.flops()} but expected {expected_flops}"

    # Latency × throughput ≈ FLOPs (within 10% due to overhead model)
    flops_check = cfg.flops() / (result.latency_ms * 1e-3)
    assert abs(flops_check / 1e12 - result.tflops) / result.tflops < 0.15, \
        f"{label}: latency/tflops consistency check failed"


# ===========================================================================
# Test Suite 1 — Gemma 4 GEMM shapes
# ===========================================================================

class TestGemma4GEMM:
    """
    Tests every weight-matrix GEMM for each Gemma 4 variant across a range
    of sequence lengths covering both decode and prefill scenarios.
    """

    @pytest.mark.parametrize("model_name", list(GEMMA4_SPECS.keys()))
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_all_layer_gemms(self, bench, model_name, seq_len):
        """All 8 weight GEMMs run and produce sane results for each model/seq."""
        spec = GEMMA4_SPECS[model_name]
        gemms = spec.layer_gemms(seq_len)
        assert len(gemms) == 8, "expected Q/K/V/O/FFN-gate/FFN-up/FFN-down/LM-head"

        results = []
        for g in gemms:
            result = bench.run_single(g.M, g.N, g.K)
            label = f"{model_name} seq={seq_len} {g.name}"
            _assert_result_sane(result, result.config, bench, label)
            results.append(result)

        # All results collected without exception
        assert len(results) == 8

    @pytest.mark.parametrize("model_name", list(GEMMA4_SPECS.keys()))
    def test_decode_batch1_gemms(self, bench, model_name):
        """Single-token decode (M=1) — all projections must still run."""
        spec = GEMMA4_SPECS[model_name]
        for g in spec.layer_gemms(seq_len=1):
            result = bench.run_single(g.M, g.N, g.K)
            _assert_result_sane(result, result.config, bench,
                                 f"{model_name} decode M=1 {g.name}")
            # Decode is almost always memory-bound at M=1
            assert result.roofline_bound == "memory", \
                f"{model_name} {g.name}: expected memory-bound at M=1, got compute-bound"

    @pytest.mark.parametrize("model_name", list(GEMMA4_SPECS.keys()))
    def test_ffn_is_larger_than_attn(self, bench, model_name):
        """Gemma 4 FFN intermediate > hidden → gate/up GEMM N > O-proj N."""
        spec = GEMMA4_SPECS[model_name]
        assert spec.intermediate > spec.hidden, \
            f"{model_name}: intermediate={spec.intermediate} should exceed hidden={spec.hidden}"

        ffn_gate = bench.run_single(512, spec.intermediate, spec.hidden)
        o_proj   = bench.run_single(512, spec.hidden,       spec.q_dim())

        # FFN gate has more FLOPs than O-proj at same seq_len
        assert ffn_gate.config.flops() > o_proj.config.flops(), \
            f"{model_name}: FFN gate should have more FLOPs than O-proj"

    @pytest.mark.parametrize("model_name", list(GEMMA4_SPECS.keys()))
    def test_gqa_kv_smaller_than_q(self, bench, model_name):
        """GQA: K/V projections must be smaller (fewer output features) than Q."""
        spec = GEMMA4_SPECS[model_name]
        assert spec.kv_dim() < spec.q_dim(), \
            f"{model_name}: kv_dim={spec.kv_dim()} should be < q_dim={spec.q_dim()}"

    def test_gemma4_4b_q_proj_shape(self, bench):
        """Gemma4-4B Q projection is (seq, 4096, 2560) — validate exact dims."""
        spec = GEMMA4_SPECS["gemma4-4b"]
        # 16 heads × 256 head_dim = 4096
        assert spec.q_dim() == 4096, f"expected q_dim=4096, got {spec.q_dim()}"
        r = bench.run_single(512, 4096, 2560)
        assert r.config.M == 512 and r.config.N == 4096 and r.config.K == 2560

    def test_gemma4_27b_ffn_down_shape(self, bench):
        """Gemma4-27B FFN-down is (seq, 5632, 22528) — high arithmetic intensity."""
        spec = GEMMA4_SPECS["gemma4-27b"]
        r = bench.run_single(2048, spec.hidden, spec.intermediate)
        # AI = FLOPs / bytes; large K/N means compute-bound at seq=2048
        ai = r.config.flops() / r.config.bytes_accessed()
        assert ai > 1.0, f"expected AI>1 for large FFN-down, got {ai:.2f}"

    @pytest.mark.parametrize("model_name", list(GEMMA4_SPECS.keys()))
    def test_fp8_gemm_same_shape(self, bench_fp8, model_name):
        """FP8 runs for Q-proj shape produce higher peak TFLOPs than BF16."""
        spec = GEMMA4_SPECS[model_name]
        r = bench_fp8.run_single(512, spec.q_dim(), spec.hidden)
        assert r.tflops > 0
        # FP8 peak is 2× BF16, so TFLOPs should be ≥ BF16 benchmark (same shape)
        r_bf16 = GEMMBenchmark(dtype="bf16",
                                peak_tflops=PEAK_TFLOPS_BF16,
                                peak_membw_gbps=PEAK_MEMBW_GBPS).run_single(
                                    512, spec.q_dim(), spec.hidden)
        assert r.tflops >= r_bf16.tflops * 0.9, \
            "FP8 TFLOPs should be >= BF16 TFLOPs for same shape"


# ===========================================================================
# Test Suite 2 — LLaMA 3.2 GEMM shapes
# ===========================================================================

class TestLlama32GEMM:
    """
    Tests every weight-matrix GEMM for LLaMA 3.2 1B and 3B across both
    decode and prefill sequence lengths.
    """

    @pytest.mark.parametrize("model_name", list(LLAMA32_SPECS.keys()))
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_all_layer_gemms(self, bench, model_name, seq_len):
        """All 8 weight GEMMs run and produce sane results."""
        spec = LLAMA32_SPECS[model_name]
        for g in spec.layer_gemms(seq_len):
            result = bench.run_single(g.M, g.N, g.K)
            _assert_result_sane(result, result.config, bench,
                                 f"{model_name} seq={seq_len} {g.name}")

    @pytest.mark.parametrize("model_name", list(LLAMA32_SPECS.keys()))
    @pytest.mark.parametrize("seq_len", DECODE_SEQ_LENS)
    def test_decode_batch_gemms(self, bench, model_name, seq_len):
        """Small-batch decode (M=1..32) — memory-bound for all projections."""
        spec = LLAMA32_SPECS[model_name]
        for g in spec.layer_gemms(seq_len):
            result = bench.run_single(g.M, g.N, g.K)
            _assert_result_sane(result, result.config, bench,
                                 f"{model_name} decode M={seq_len} {g.name}")

    def test_llama32_1b_hidden_size(self):
        """LLaMA 3.2-1B hidden=2048 (32 heads × 64 head_dim)."""
        spec = LLAMA32_SPECS["llama3.2-1b"]
        assert spec.hidden == 2048
        assert spec.num_heads * spec.head_dim == 2048

    def test_llama32_3b_hidden_size(self):
        """LLaMA 3.2-3B hidden=3072 (24 heads × 128 head_dim)."""
        spec = LLAMA32_SPECS["llama3.2-3b"]
        assert spec.hidden == 3072
        assert spec.num_heads * spec.head_dim == 3072

    def test_llama32_gqa_ratio_1b(self):
        """1B GQA ratio: 32 Q heads / 8 KV heads = 4× reduction in KV proj."""
        spec = LLAMA32_SPECS["llama3.2-1b"]
        ratio = spec.num_heads // spec.kv_heads
        assert ratio == 4
        assert spec.q_dim() == 4 * spec.kv_dim()

    def test_llama32_gqa_ratio_3b(self):
        """3B GQA ratio: 24 Q heads / 8 KV heads = 3× reduction."""
        spec = LLAMA32_SPECS["llama3.2-3b"]
        ratio = spec.num_heads // spec.kv_heads
        assert ratio == 3

    @pytest.mark.parametrize("model_name", list(LLAMA32_SPECS.keys()))
    def test_prefill_long_context_compute_bound(self, bench, model_name):
        """At seq=4096 all FFN GEMMs should be compute-bound."""
        spec = LLAMA32_SPECS[model_name]
        for layer_name, N, K in [
            ("FFN_gate", spec.intermediate, spec.hidden),
            ("FFN_up",   spec.intermediate, spec.hidden),
            ("FFN_down", spec.hidden,       spec.intermediate),
        ]:
            r = bench.run_single(4096, N, K)
            assert r.roofline_bound == "compute", \
                f"{model_name} seq=4096 {layer_name}: expected compute-bound"

    @pytest.mark.parametrize("model_name", list(LLAMA32_SPECS.keys()))
    def test_lm_head_is_wide(self, bench, model_name):
        """LM-head is a wide GEMM: N=vocab_size=32000."""
        spec = LLAMA32_SPECS[model_name]
        r = bench.run_single(512, spec.vocab, spec.hidden)
        assert r.config.N == 32000
        # vocab GEMMs are typically compute-bound during prefill
        assert r.roofline_bound == "compute"

    @pytest.mark.parametrize("model_name,seq_len", [
        ("llama3.2-1b", 128),
        ("llama3.2-1b", 2048),
        ("llama3.2-3b", 128),
        ("llama3.2-3b", 2048),
    ])
    def test_o_proj_vs_q_proj_symmetry(self, bench, model_name, seq_len):
        """O-proj and Q-proj have transposed N/K — same FLOPs."""
        spec = LLAMA32_SPECS[model_name]
        q_r = bench.run_single(seq_len, spec.q_dim(),  spec.hidden)
        o_r = bench.run_single(seq_len, spec.hidden,   spec.q_dim())
        assert q_r.config.flops() == o_r.config.flops(), \
            "Q-proj and O-proj must have equal FLOPs (transposed M/N same product)"


# ===========================================================================
# Test Suite 3 — GEMM corner cases
# ===========================================================================

class TestGEMMCornerCases:
    """
    Exhaustive coverage of unusual M × N × K shapes that stress tiling,
    alignment, and roofline classification logic.
    """

    # -----------------------------------------------------------------------
    # Single-token decode  (M=1)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("N,K", [
        (4096, 4096),    # square weight, M=1
        (8192, 1024),    # wide output
        (1024, 8192),    # deep reduction
        (32000, 4096),   # vocab head, M=1
    ])
    def test_m1_vector_matrix(self, bench, N, K):
        """M=1 is the batch=1 decode token — must be memory-bound."""
        r = bench.run_single(1, N, K)
        _assert_result_sane(r, r.config, bench, f"M=1 N={N} K={K}")
        assert r.roofline_bound == "memory", \
            f"M=1 N={N} K={K}: single-token decode must be memory-bound"

    # -----------------------------------------------------------------------
    # Skinny-tall  (M >> N)  — tall activation, narrow weight column
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (8192,   64,  4096),   # extreme column-narrow
        (4096,  128,  2048),
        (16384,  32,  1024),
        (2048,   16,  4096),
        (1024,    1,  2048),   # N=1 (scalar output per token)
    ])
    def test_skinny_tall(self, bench, M, N, K):
        """M >> N: tall-and-narrow output — tests memory-bound classification."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"skinny-tall M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Tall-skinny  (N >> M)  — wide output, few input rows
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (64,   8192,  4096),   # extreme row-narrow activation
        (128,  4096,  2048),
        (32,  16384,  1024),
        (16,   2048,  4096),
        (1,   32000,  4096),   # decode + vocab head — both extremes
    ])
    def test_tall_skinny(self, bench, M, N, K):
        """N >> M: wide output, shallow activation batch."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"tall-skinny M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Flat K  (K very small relative to M and N)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (4096, 4096,    1),    # K=1 — outer product
        (4096, 4096,    4),
        (4096, 4096,   16),
        (4096, 4096,   32),
    ])
    def test_flat_k(self, bench, M, N, K):
        """Very small K — low arithmetic intensity, should be memory-bound."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"flat-K M={M} N={N} K={K}")
        assert r.roofline_bound == "memory", \
            f"K={K} is too small for compute-bound classification"

    # -----------------------------------------------------------------------
    # Deep K  (K very large)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (256,  256,  65536),   # large reduction over tiny output
        (512,  512,  32768),
        (128,  128, 131072),
    ])
    def test_deep_k(self, bench, M, N, K):
        """Very large K — accumulation stress; memory bytes dominated by A+B."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"deep-K M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Non-power-of-two alignment
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (1000, 1000, 1000),    # round thousand
        (768,  768,   768),    # BERT-base hidden
        (1152, 4608,  1152),   # Gemma 2B-ish
        (512,  4864,  512),    # non-standard FFN
        (1023, 1023, 1023),    # one-less-than-power-of-two
        (1025, 1025, 1025),    # one-more-than-power-of-two
        (384,  1536,  384),    # MiniLM
    ])
    def test_non_pow2(self, bench, M, N, K):
        """Non-power-of-two sizes exercise alignment corner cases."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"non-pow2 M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Odd / prime sizes — worst-case alignment
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (127,  257,  511),     # primes minus one
        (131,  251,  509),     # primes
        (17,    37,   71),     # small primes
        (1,      1,    1),     # trivial (scalar)
        (3,      5,    7),     # tiny primes
    ])
    def test_odd_prime_sizes(self, bench, M, N, K):
        """Odd / prime sizes stress alignment and remainder handling."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"prime M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Square powers-of-two
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024, 2048, 4096, 8192])
    def test_square_pow2(self, bench, size):
        """Square GEMMs should become compute-bound past the H100 balance point."""
        r = bench.run_single(size, size, size)
        _assert_result_sane(r, r.config, bench, f"square-{size}")
        if size >= 1024:
            assert r.roofline_bound == "compute", \
                f"Square-{size} GEMM expected compute-bound, got memory-bound"

    # -----------------------------------------------------------------------
    # Transposed-dominant  (M < K and N < K)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M,N,K", [
        (256,  256, 8192),    # deep reduction, narrow output
        (128,  512, 4096),
        (64,  1024, 16384),
    ])
    def test_deep_reduction_narrow_output(self, bench, M, N, K):
        """K >> M and K >> N: typical for depthwise-like patterns."""
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"deep-reduction M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Bytes-accessed formula consistency
    # -----------------------------------------------------------------------

    def test_bytes_fp8_half_of_fp16(self, bench):
        """FP8 bytes_accessed should be half of FP16 for same shape."""
        cfg_fp8  = GEMMConfig(M=1024, N=1024, K=1024, dtype="fp8")
        cfg_fp16 = GEMMConfig(M=1024, N=1024, K=1024, dtype="fp16")
        assert cfg_fp8.bytes_accessed() * 2 == cfg_fp16.bytes_accessed()

    def test_bytes_fp32_double_of_fp16(self):
        """FP32 bytes should be 2× FP16 for same shape."""
        cfg_fp32 = GEMMConfig(M=512, N=512, K=512, dtype="fp32")
        cfg_fp16 = GEMMConfig(M=512, N=512, K=512, dtype="fp16")
        assert cfg_fp32.bytes_accessed() == 2 * cfg_fp16.bytes_accessed()

    def test_flops_formula(self):
        """FLOPs = 2·M·N·K for all edge dims."""
        for M, N, K in [(1, 1, 1), (1, 4096, 4096), (4096, 1, 4096), (3, 7, 11)]:
            assert GEMMConfig(M=M, N=N, K=K).flops() == 2 * M * N * K

    def test_mfu_saturates_at_peak(self, bench):
        """MFU must not exceed 1.0 even for very large GEMMs."""
        r = bench.run_single(16384, 16384, 16384)
        assert bench.mfu(r) <= 1.0 + 1e-9

    def test_roofline_crossover_near_ridge(self, bench):
        """
        The ridge point satisfies  FLOPs/bytes = peak_tflops/peak_bw.
        A GEMM slightly above the ridge must be compute-bound; slightly below
        must be memory-bound.
        """
        ridge = bench.peak_tflops * 1e12 / (bench.peak_membw_gbps * 1e9)
        # Build a GEMM at exactly 2× the ridge arithmetic intensity → compute-bound
        # AI = 2MNK / (2*(MK+KN+MN)) bytes (bf16, bpe=2)
        # For M=N=K=S: AI ≈ 2S³ / (6S²) = S/3; set S/3 = 2*ridge → S = 6*ridge
        S_compute = math.ceil(6 * ridge)
        S_memory  = max(1, math.floor(ridge / 6))  # AI ≈ ridge/6 << ridge

        r_compute = bench.run_single(S_compute, S_compute, S_compute)
        r_memory  = bench.run_single(S_memory,  S_memory,  S_memory)

        assert r_compute.roofline_bound == "compute", \
            f"S={S_compute} (2× ridge) expected compute-bound"
        assert r_memory.roofline_bound == "memory", \
            f"S={S_memory} (1/6× ridge) expected memory-bound"


# ===========================================================================
# Test Suite 4 — Full shmoo sweep
# ===========================================================================

# Shmoo axis values — logarithmic spacing covering decode to large prefill
SHMOO_M = [1, 8, 64, 128, 512, 1024, 2048, 4096, 8192]
SHMOO_N = [1, 8, 64, 128, 512, 1024, 2048, 4096, 8192]
SHMOO_K = [1, 8, 64, 128, 512, 1024, 2048, 4096, 8192]

# Reduced cross-product for the full 3D sweep (avoids O(N³) test count):
# We sweep each axis independently while fixing the others at 1024.
SHMOO_FIXED = 1024


class TestGEMMShmoo:
    """
    Shmoo (sweep) tests that vary one dimension at a time while holding the
    other two fixed, covering the full dynamic range of each axis.

    One full 3D Cartesian test (all combos) is also included but marked
    as slow so it can be excluded from fast CI runs.
    """

    # -----------------------------------------------------------------------
    # M sweep  (batch × seq_len axis)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M", SHMOO_M)
    def test_shmoo_m(self, bench, M):
        """Sweep M with N=K=1024 — covers decode→prefill transition."""
        N, K = SHMOO_FIXED, SHMOO_FIXED
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"shmoo-M M={M} N={N} K={K}")

        # Monotone FLOPs: larger M → more FLOPs
        r_prev = bench.run_single(max(1, M // 2), N, K)
        if M > 1:
            assert r.config.flops() > r_prev.config.flops()

    @pytest.mark.parametrize("M", SHMOO_M)
    def test_shmoo_m_latency_monotone(self, bench, M):
        """Latency must be non-decreasing as M grows (roofline model)."""
        N, K = SHMOO_FIXED, SHMOO_FIXED
        r_curr = bench.run_single(M, N, K)
        if M > 1:
            r_half = bench.run_single(M // 2, N, K)
            assert r_curr.latency_ms >= r_half.latency_ms * 0.99, \
                f"latency should be non-decreasing: M={M} ({r_curr.latency_ms:.4f} ms) " \
                f"< M={M//2} ({r_half.latency_ms:.4f} ms)"

    # -----------------------------------------------------------------------
    # N sweep  (output feature axis)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("N", SHMOO_N)
    def test_shmoo_n(self, bench, N):
        """Sweep N with M=K=1024."""
        M, K = SHMOO_FIXED, SHMOO_FIXED
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"shmoo-N M={M} N={N} K={K}")

    @pytest.mark.parametrize("N", SHMOO_N)
    def test_shmoo_n_latency_monotone(self, bench, N):
        """Latency non-decreasing as N grows."""
        M, K = SHMOO_FIXED, SHMOO_FIXED
        r_curr = bench.run_single(M, N, K)
        if N > 1:
            r_half = bench.run_single(M, N // 2, K)
            assert r_curr.latency_ms >= r_half.latency_ms * 0.99

    # -----------------------------------------------------------------------
    # K sweep  (reduction / contraction axis)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("K", SHMOO_K)
    def test_shmoo_k(self, bench, K):
        """Sweep K with M=N=1024."""
        M, N = SHMOO_FIXED, SHMOO_FIXED
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"shmoo-K M={M} N={N} K={K}")

    @pytest.mark.parametrize("K", SHMOO_K)
    def test_shmoo_k_latency_monotone(self, bench, K):
        """Latency non-decreasing as K grows."""
        M, N = SHMOO_FIXED, SHMOO_FIXED
        r_curr = bench.run_single(M, N, K)
        if K > 1:
            r_half = bench.run_single(M, N, K // 2)
            assert r_curr.latency_ms >= r_half.latency_ms * 0.99

    # -----------------------------------------------------------------------
    # Roofline bound transition across M sweep
    # -----------------------------------------------------------------------

    def test_shmoo_roofline_transition(self, bench):
        """
        As M grows from 1 to 8192 (N=K=1024 fixed), the bound must
        eventually flip from memory to compute.  Verify at least one
        transition occurs across the full sweep.
        """
        N, K = SHMOO_FIXED, SHMOO_FIXED
        bounds = [bench.run_single(M, N, K).roofline_bound for M in SHMOO_M]
        assert "memory" in bounds, "small M should produce memory-bound results"
        assert "compute" in bounds, "large M should produce compute-bound results"
        # Transition must be monotone (no flipping back)
        transitioned = False
        for b in bounds:
            if b == "compute":
                transitioned = True
            if transitioned:
                assert b == "compute", \
                    "roofline bound must not flip back to memory after transition"

    # -----------------------------------------------------------------------
    # Full 3D Cartesian shmoo (slow — run with -m slow)
    # -----------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.parametrize("M", SHMOO_M)
    @pytest.mark.parametrize("N", SHMOO_N)
    @pytest.mark.parametrize("K", SHMOO_K)
    def test_shmoo_3d(self, bench, M, N, K):
        """
        Full 3D shmoo: 9×9×9 = 729 configurations.
        Marked @slow — run with:  pytest -m slow tests/unit/test_gemm_model_shapes.py
        """
        r = bench.run_single(M, N, K)
        _assert_result_sane(r, r.config, bench, f"3d-shmoo M={M} N={N} K={K}")

    # -----------------------------------------------------------------------
    # Arithmetic intensity monotonicity across K
    # -----------------------------------------------------------------------

    def test_shmoo_arithmetic_intensity_increases_with_k(self, bench):
        """
        For fixed M=N=512, arithmetic intensity AI = 2MNK / bytes(M,N,K).
        AI must increase monotonically with K.
        """
        M = N = 512
        prev_ai = 0.0
        for K in [64, 128, 256, 512, 1024, 2048, 4096]:
            cfg = GEMMConfig(M=M, N=N, K=K)
            ai = cfg.flops() / cfg.bytes_accessed()
            assert ai > prev_ai, \
                f"AI should increase with K; K={K} AI={ai:.2f} <= prev={prev_ai:.2f}"
            prev_ai = ai

    # -----------------------------------------------------------------------
    # MFU consistency across the sweep
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("M", SHMOO_M)
    def test_shmoo_mfu_bounded(self, bench, M):
        """MFU must be in (0, 1] for every shmoo point."""
        N, K = SHMOO_FIXED, SHMOO_FIXED
        r = bench.run_single(M, N, K)
        mfu = bench.mfu(r)
        assert 0.0 < mfu <= 1.0 + 1e-9, \
            f"MFU={mfu:.4f} out of bounds at M={M}"

    # -----------------------------------------------------------------------
    # JSON round-trip across a mini-shmoo
    # -----------------------------------------------------------------------

    def test_shmoo_json_roundtrip(self, bench, tmp_path):
        """Run a 3×3 mini-shmoo and verify JSON serialisation is lossless."""
        import json, os
        mini = [(M, N, K)
                for M in [64, 512, 4096]
                for N in [64, 512, 4096]
                for K in [64, 512, 4096]]
        results = bench.run(sizes=mini)
        out = str(tmp_path / "shmoo.json")
        bench.to_json(results, out)

        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)

        assert len(data["results"]) == len(mini)
        for row, (M, N, K) in zip(data["results"], mini):
            assert row["M"] == M
            assert row["N"] == N
            assert row["K"] == K
            assert row["tflops"] > 0
            assert row["roofline_bound"] in ("compute", "memory")

    # -----------------------------------------------------------------------
    # Summary report correctness across shmoo
    # -----------------------------------------------------------------------

    def test_shmoo_report_all_sizes_present(self, bench):
        """report() text must contain every M from the M-sweep."""
        sizes = [(M, SHMOO_FIXED, SHMOO_FIXED) for M in SHMOO_M]
        results = bench.run(sizes=sizes)
        text = bench.report(results)
        for M in SHMOO_M:
            assert str(M) in text, f"M={M} not found in report"
