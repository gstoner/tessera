
import os, time, json, math, contextlib
from typing import Dict, Any, Optional, Tuple, Iterable

from .utils import Objective, gpu_fingerprint, now_iso, stable_hash
from .cache import CacheDB

try:
    import torch
except Exception:
    torch = None

def _cuda_timer_start():
    if torch is None or not torch.cuda.is_available():
        return None
    return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

def _cuda_timer_elapsed_ms(ev):
    if ev is None:
        return None
    start, end = ev
    end.synchronize()
    return start.elapsed_time(end)

class ResultLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "results.csv")
        self.jsonl_path = os.path.join(out_dir, "results.jsonl")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("ts,trial,config,budget_iters,latency_ms,tflops,objective,best_so_far,notes\n")
    def append(self, trial: int, config: Dict[str,Any], budget_iters: Optional[int], metrics: Dict[str,Any], objective_value: float, best_so_far: float, notes: str = ""):
        row = {
            "ts": now_iso(),
            "trial": trial,
            "config": config,
            "budget_iters": budget_iters,
            "latency_ms": metrics.get("latency_ms"),
            "tflops": metrics.get("tflops"),
            "objective": objective_value,
            "best_so_far": best_so_far,
            "notes": notes,
        }
        with open(self.jsonl_path, "a") as jf:
            jf.write(json.dumps(row) + "\n")
        with open(self.csv_path, "a") as cf:
            cf.write(f'{row["ts"]},{trial},"{json.dumps(config, sort_keys=True)}",{budget_iters},{row["latency_ms"]},{row["tflops"]},{objective_value},{best_so_far},"{notes}"\n')

class SyntheticGEMMWorkload:
    """Synthetic GEMM; budget maps to 'iters' for Hyperband."""
    def __init__(self, M: int, N: int, K: int, device: str="cuda", dtype: str="bf16", iters: int=30, warmup: int=5):
        self.M,self.N,self.K = M,N,K
        self.device=device; self.dtype=dtype
        self.iters=iters; self.warmup=warmup
    def signature(self) -> dict:
        return {"kind":"synthetic_gemm","M":self.M,"N":self.N,"K":self.K,"dtype":self.dtype}
    def kernel_id(self) -> str:
        return "synthetic_gemm"
    def _torch_dtype(self):
        if torch is None: return None
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(self.dtype, torch.bfloat16)
    @contextlib.contextmanager
    def with_budget(self, budget_iters: Optional[int]):
        old = self.iters
        if budget_iters is not None:
            self.iters = max(1, int(budget_iters))
        try:
            yield
        finally:
            self.iters = old
    def evaluate(self, config: Dict[str,Any]) -> Dict[str,Any]:
        num_stages = int(config.get("num_stages",2))
        num_warps = int(config.get("num_warps",8))
        block_m = int(config.get("BLOCK_M",128))
        block_n = int(config.get("BLOCK_N",128))
        block_k = int(config.get("BLOCK_K",64))
        if torch is not None:
            device = self.device if torch.cuda.is_available() else "cpu"
            dtype = self._torch_dtype()
            a = torch.randn(self.M, self.K, device=device, dtype=dtype)
            b = torch.randn(self.K, self.N, device=device, dtype=dtype)
            if device == "cuda":
                torch.cuda.synchronize()
            for _ in range(self.warmup):
                c = a @ b
            if device == "cuda":
                torch.cuda.synchronize()
            ev = _cuda_timer_start()
            t0 = time.perf_counter()
            if ev: ev[0].record()
            for _ in range(self.iters):
                for km in range(0, self.K, block_k):
                    ks = slice(km, min(km+block_k, self.K))
                    c = a[:, ks] @ b[ks, :]
            if ev: ev[1].record()
            if device == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed_ms = _cuda_timer_elapsed_ms(ev) if ev else (t1 - t0) * 1000.0
        else:
            import numpy as np
            a = np.random.randn(self.M, self.K).astype("float32")
            b = np.random.randn(self.K, self.N).astype("float32")
            for _ in range(self.warmup): c = a @ b
            t0 = time.perf_counter()
            for _ in range(self.iters): c = a @ b
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        flops = 2.0 * self.M * self.N * self.K * self.iters
        warp_penalty = 1.0 + 0.10 * (abs(num_warps - 8) / 8.0)
        stage_bonus = 1.0 + 0.03 * max(0, num_stages - 1)
        block_aspect = (block_m * block_n) / max(1, block_k)
        import math as _m
        intensity_scale = 1.0 + 0.05 * _m.log10(max(1.0, block_aspect))
        adj_ms = elapsed_ms * warp_penalty / (stage_bonus * intensity_scale)
        tflops = (flops / (adj_ms * 1e-3)) / 1e12
        return {"latency_ms": adj_ms, "tflops": tflops}

class Autotuner:
    def __init__(self, objective: Objective, out_dir: str, cache_path: Optional[str] = None):
        self.objective = objective
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = ResultLogger(out_dir)
        self.cache = CacheDB(cache_path or os.path.join(out_dir, "cache.db"))
        self.device_fp = gpu_fingerprint()
        self.device_class = self.device_fp.get("gpu_name","cpu")

    def _schedule_key(self, workload, config: Dict[str,Any]) -> str:
        key = {
            "kernel_id": workload.kernel_id(),
            "problem": workload.signature(),
            "config": config,
            "device_class": self.device_class,
            "objective": {"mode": self.objective.mode, "key": self.objective.key},
        }
        return stable_hash(key)

    def run(self, candidates_iter: Iterable, workload, max_trials: Optional[int] = None, early_stop=None) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        best_cfg, best_metrics = None, None
        best_val = float("-inf") if self.objective.mode == "max" else float("inf")
        trial = 0

        # Support Hyperband-style generators that yield (cfg,budget) and rung separators (None)
        pending_scores = []  # (cfg,budget,val) for current rung

        for item in candidates_iter:
            if item is None:
                # rung boundary: keep top 1/eta externally? We assume the generator wants tuner to filter.
                # We sort pending_scores and keep top third.
                if pending_scores:
                    k = max(1, len(pending_scores)//3)
                    pending_scores = sorted(pending_scores, key=lambda x: x[2], reverse=(self.objective.mode=="max"))[:k]
                    # Convert back into (cfg, budget) for next rung by re-yielding to the iterator is not possible.
                    # Instead we stash them so the generator had better re-sample; practical HB flow will be via a driver.
                continue

            if isinstance(item, tuple):
                cfg, budget = item
            else:
                cfg, budget = item, None

            trial += 1
            if max_trials is not None and trial > max_trials:
                break

            # Cache lookup
            skey = self._schedule_key(workload, cfg)
            cached = self.cache.lookup(self.device_class, workload.kernel_id(), workload.signature(), skey)
            if cached is not None:
                metrics = cached
                val = metrics.get(self.objective.key, float("nan"))
            else:
                # Evaluate (with optional budget override)
                if hasattr(workload, "with_budget"):
                    ctx = workload.with_budget(budget)
                else:
                    import contextlib
                    ctx = contextlib.nullcontext()
                with ctx:
                    metrics = workload.evaluate(cfg)
                val = metrics.get(self.objective.key, float("nan"))
                self.cache.put_trial({
                    "ts": now_iso(),
                    "device_class": self.device_class,
                    "kernel_id": workload.kernel_id(),
                    "problem_sig": workload.signature(),
                    "schedule_key": skey,
                    "config": cfg,
                    "budget_iters": budget,
                    "metrics": metrics,
                    "objective_value": float(val) if isinstance(val,(int,float)) else 0.0,
                })

            if isinstance(val, (int,float)):
                improved = (val > best_val) if self.objective.mode == "max" else (val < best_val)
                if improved:
                    best_cfg, best_metrics, best_val = cfg, metrics, val

            self.logger.append(trial, cfg, budget, metrics, float(val) if isinstance(val,(int,float)) else float("nan"), best_val, notes="cache" if cached else "")

            if early_stop is not None and isinstance(best_val,(int,float)):
                if early_stop.update(best_val):
                    break

        return best_cfg or {}, best_metrics or {}
