
#!/usr/bin/env python3
A minimal, standalone script that mirrors the notebook flow.

- Tries to use the real `tessera` package if present.
- Falls back to a shim so the script still runs.
- Demonstrates: define function → compile → run → (attempt) IR dumps.

import math, importlib, sys

# Try to import tessera
HAVE_TESSERA = importlib.util.find_spec("tessera") is not None
if HAVE_TESSERA:
    import tessera as tsr
else:
    # Shim API so script can run without tessera installed
    class _TensorShim:
        def __init__(self, t, shape=None, dtype=None):
            self._t = t
            self._shape = shape or getattr(t, "shape", ())
        @property
        def shape(self): return self._shape
        def transpose(self, a, b): return self  # illustrative only
    class _TSRShim:
        Tensor = _TensorShim
        def tensor(self, x, shape=None, dtype=None): return _TensorShim(x, shape, dtype)
        def matmul(self, a, b): return a
        def softmax(self, x, dim=-1): return x
        def __getattr__(self, k):
            if k == "function":
                def deco(fn):
                    class _F:
                        def __init__(self): self.compiled=False
                        def compile(self): self.compiled=True
                        def __call__(self, *args, **kwargs): return args[0]
                        def profile(self, *args, **kwargs):
                            return {"kernel_time_ms": 0.42, "occupancy_percentage": 75,
                                    "memory_bandwidth_gb_s": 900.0, "tensor_core_utilization": 80.5,
                                    "bottleneck": "compute_bound"}
                        def autotune(self, *a, **kw): print("Autotune (shim): done")
                    return _F()
                return deco
            raise AttributeError(k)
    tsr = _TSRShim()

def main():
    import torch
    # Define a Flash Attention function (illustrative)
    @tsr.function
    def flash_attention(q: tsr.Tensor,
                        k: tsr.Tensor,
                        v: tsr.Tensor) -> tsr.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1] if hasattr(q, 'shape') else 64)
        scores = tsr.matmul(q, k)  # illustrative
        probs = tsr.softmax(scores, dim=-1)
        return tsr.matmul(probs, v)

    # Sample inputs
    B, H, S, D = 2, 4, 128, 64
    q = tsr.tensor(torch.randn(B, H, S, D, dtype=torch.float16), shape=[B, H, S, D])
    k = tsr.tensor(torch.randn(B, H, S, D, dtype=torch.float16), shape=[B, H, S, D])
    v = tsr.tensor(torch.randn(B, H, S, D, dtype=torch.float16), shape=[B, H, S, D])

    # Compile & run
    flash_attention.compile()
    out = flash_attention(q, k, v)
    print("-> Ran flash_attention; type(out):", type(out).__name__)
    if hasattr(out, "shape"):
        print("-> Output shape:", out.shape)

    # Profile
    prof = flash_attention.profile(q, k, v)
    print("-> Profile:", prof)

    # Autotune (optional)
    try:
        flash_attention.autotune([(q, k, v)], evaluation_budget=10)
    except Exception as e:
        print("Autotune skipped:", e)

    # Attempt IR dump hook if available (pseudo-API)
    if HAVE_TESSERA and hasattr(tsr, "dump_ir"):
        for stage in ["graph", "schedule", "tile", "target"]:
            try:
                dump = tsr.dump_ir(flash_attention, stage=stage)
                print(f"==== {stage.upper()} IR ====")
                print(str(dump)[:1000])
            except Exception as e:
                print(f"[dump_ir error @ {stage}]:", e)
    else:
        print("IR dumps require tessera runtime/API; using notebook for illustrative snippets.")

if __name__ == "__main__":
    main()
