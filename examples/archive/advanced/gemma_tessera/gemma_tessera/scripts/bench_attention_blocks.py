
#!/usr/bin/env python3
import torch, time, argparse, itertools
from tessera_gemma.kernels.native_attention_tessera import native_flash_attention, _repeat_kv
from tessera_gemma.kernels.native_attention_paged_tessera import native_flash_attention_paged
from tessera_gemma.kernels.kv_cache_tessera import PagedKVCache

def timer_ms(fn, iters=10):
    device = next((p for p in fn.__code__.co_freevars), None)  # not used
    def run():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) * 1000.0
    # warmup
    for _ in range(3):
        _ = fn()
    times = [run() for _ in range(iters)]
    return sum(times) / len(times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, nargs="+", default=[64, 80, 128])
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--blocks", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--paged_page", type=int, default=128)
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print("mode,T,D,block,avg_ms")
    for T in args.T:
        for D in args.D:
            B, H = args.B, args.H
            Hk = max(1, H//4)
            q = torch.randn(B, T, H, D, device=device, dtype=dtype)
            k = torch.randn(B, T, Hk, D, device=device, dtype=dtype)
            v = torch.randn(B, T, Hk, D, device=device, dtype=dtype)
            k2, v2 = _repeat_kv(k, v, H, Hk)

            # Non-paged: sweep blocks
            for bs in args.blocks:
                avg = timer_ms(lambda: native_flash_attention(q, k2, v2, causal=True, dropout_p=0.0, block_size=bs))
                print(f"native,{T},{D},{bs},{avg:.3f}")

            # Paged: build cache pages
            cache = PagedKVCache(B, Hk, D, page_size=args.paged_page, device=device, dtype=dtype)
            cache.append(k, v)
            for bs in args.blocks:
                avg = timer_ms(lambda: native_flash_attention_paged(q, list(cache.pages()), causal=True, dropout_p=0.0, block_size=bs))
                print(f"paged,{T},{D},{bs},{avg:.3f}")

if __name__ == "__main__":
    main()
