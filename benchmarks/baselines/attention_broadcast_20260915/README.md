# Attention broadcast masks on every axis — SM120 device rows

Owner: FRONTEND-IR-MEDIUM-1 (sync `ATTN-QK-BROADCAST-2026-09-15`).
Parent: `origin/main` at `78c73e6cb6527e106c27b2cf3e6ceb4a7b9d986a`, plus the
source fingerprints in `sources.json`. Packets are correctness evidence; none
promotes a performance candidate or transfers evidence between architectures.

Host: The-Super-Bear (WSL2), RTX 5070 (sm_120), CUDA 13.4 (nvcc 13.4.59),
driver 610.88, `build-nvidia-cuda/` rebuilt from the fingerprinted sources
(`tessera-nvidia-opt` and `libtessera_nvidia_ptx_launch.so` — the runtime loads
both from that tree, not from `build/`; a stale copy produced `rc=5` and then a
silently wrong bias index before the rebuild, so `compiler_sha256` and
`runtime_sha256` in each packet name the binaries that ran).

Each packet is one raised attention bucket, B=2, Hq=4, Hkv=2, D=4, Dv=2, with
end-aligned causal masking and a left window of 2, executed without host
expansion of the bias and compared to the numpy oracle
(`grouped_masked_bias` over the broadcast view):

| Packet | Bias (physical) | Form |
|---|---|---|
| `attention_batch_head_{3_5,5_3}.json` | `[1,1,Q,K]` | shared across batch and head (re-recorded under the v3 ABI) |
| `attention_key_padding_{3_5,5_3}.json` | `[1,1,1,K]` | one additive key-padding row for every batch, head and query |
| `attention_per_query_{3_5,5_3}.json` | `[B,Hq,Q,1]` | a per-(batch, head, query) constant over every key, carried as a per-row power-of-two magnitude so the f32 rounding signature identifies the row the kernel read (a constant would cancel out of the softmax); the row must also differ from the unbiased output |

`3_5` is Q=3/K=5 (ragged, K>Q) and `5_3` is Q=5/K=3. The key-axis forms carry
a `-inf` column at `K//2` that is masked wherever the axis broadcasts; the
same run refuses an empty first row (`fully masked`) after causal/window
composition. `max_abs_error` is the largest absolute deviation from the oracle
(all six ≤ 6e-8). The ABI is
`tessera.nvidia.attention.q_k_v_bias_o_dims_bias_shape.f32_f32acc.v3`: the
host copies `BiasB×BiasH×BiasQ×BiasK` floats and the kernel bakes the physical
extents in; the seven logical kernel extents are unchanged.

Reproduce on the owning host:

```bash
source .venv/bin/activate && source scripts/_nvidia_env.sh
export TESSERA_NVIDIA_PTX_LAUNCH_LIB=$PWD/build-nvidia-cuda/src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/libtessera_nvidia_ptx_launch.so
TESSERA_TEST_RAISED_ATTENTION=1 TESSERA_BROADCAST_EVIDENCE=<dir> \
  PYTHONPATH=python python -m pytest tests/unit/test_attention_broadcast.py -q
```

Boolean/padding masks as an operand, Apple/ROCm/x86 broadcast consumers,
f16/bf16 broadcast storage and performance admission remain open. WSL timings
are not selector-grade performance evidence, and nothing here is one.
