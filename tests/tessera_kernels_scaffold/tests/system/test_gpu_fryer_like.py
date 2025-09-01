import os, time, math, json, pytest, torch, torch.nn.functional as F
from pathlib import Path

try:
    import pynvml
    HAS_NVML = True
except Exception:
    HAS_NVML = False

from tessera_kernels import gemm_fp16, reduce_sum, HAS_EXT
from tessera_kernels.tiled import flashattn_fwd_tiled, flashattn_bwd_tiled

requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_EXT), reason="CUDA/Ext required")

def _nvml_snapshot():
    if not HAS_NVML: return {}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"tempC": temp, "gpu_util": util.gpu, "mem_util": 100.0*mem.used/mem.total}
    except Exception:
        return {}

@requires_cuda
def test_gpu_fryer_like_stress(tmp_path):
    torch.manual_seed(123)
    device = "cuda"
    start = time.time()
    duration_s = float(os.environ.get("TESSERA_FRYER_DURATION_S", "20"))
    S = int(os.environ.get("TESSERA_FRYER_S", "1024"))
    D = int(os.environ.get("TESSERA_FRYER_D", "128"))
    H = int(os.environ.get("TESSERA_FRYER_H", "16"))
    B = int(os.environ.get("TESSERA_FRYER_B", "1"))
    dropout_p = float(os.environ.get("TESSERA_FRYER_DROPOUT", "0.1"))

    # data
    Q = torch.randn(B,H,S,D, device=device, dtype=torch.float16, requires_grad=True)
    K = torch.randn(B,H,S,D, device=device, dtype=torch.float16, requires_grad=True)
    V = torch.randn(B,H,S,D, device=device, dtype=torch.float16, requires_grad=True)

    # GEMM matrices
    M=N=Kdim=4096
    A = torch.randn(M,Kdim, device=device, dtype=torch.float16)
    Bm = torch.randn(Kdim,N, device=device, dtype=torch.float16)

    stats = {"iters":0, "nan_inf":0, "step_time_p50":0.0, "step_time_p95":0.0, "snapshots":[]}
    ts = []

    while time.time() - start < duration_s:
        t0 = time.perf_counter()
        # 1) GEMM
        C = gemm_fp16(A,Bm)
        assert torch.isfinite(C).all()

        # 2) Attention fwd/bwd
        out = flashattn_fwd_tiled(Q,K,V, mask=None, dropout_p=dropout_p, is_causal=True, seed=1234)
        loss = out.float().pow(2).mean()
        loss.backward()
        assert all(torch.isfinite(t).all() for t in (Q,K,V))
        # Step-ish: reset grads
        for t in (Q,K,V): t.grad = None

        # 3) Reduction
        X = torch.randn(256, 2048, device=device, dtype=torch.float32)
        r = reduce_sum(X)
        assert torch.isfinite(r).all()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ts.append(dt)
        stats["iters"] += 1

        snap = _nvml_snapshot()
        if snap: stats["snapshots"].append(snap)

        # Invariant checks
        if not torch.isfinite(out).all():
            stats["nan_inf"] += 1
            break

    ts.sort()
    if ts:
        stats["step_time_p50"] = ts[len(ts)//2]
        stats["step_time_p95"] = ts[int(len(ts)*0.95)]

    # simple variance-based throttling heuristic: p95/p50 shouldn't explode
    if stats["iters"] >= 10:
        assert stats["step_time_p95"] < 3.0 * stats["step_time_p50"], "High latency variance: possible throttling/instability"

    Path(tmp_path/"gpu_fryer_like.json").write_text(json.dumps(stats, indent=2))
    # Ensure we ran at least a few iterations
    assert stats["iters"] >= 5
