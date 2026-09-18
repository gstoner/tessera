# Native Hessian-vector products on the owning devices — 2026-09-17

Recorded by `benchmarks/record_native_hvp.py`: the compiler-owned HVP package
(`native_hvp.materialize_native_hvp`, `--tessera-autodiff-hvp-prepare` →
`--tessera-autodiff-forward=export-hvp=true` → `--tessera-native-tape-to-gpu`)
executed device-resident for two source functions (`cubic`, `repeated_square`)
× three shapes, gradient and Hessian-vector product checked against their
closed forms.

| Packet | Host | Result |
|---|---|---|
| `nvidia_sm120.json` | The-Super-Bear, RTX 5070 (sm_120), CUDA 13.4 / driver 610.88, WSL2 | 6 rows exact (worst abs error 0) |
| `rocm_gfx1201.json` | Tajasarus, RX 9070 XT (gfx1201), ROCm 10.0, WSL2 | 6 rows exact (worst abs error 0) |

Correctness evidence only (`promotion.performance_eligible = false`): per-call
host transfers, no timer, and WSL2 wall clock does not promote. The two
packets are separate proofs and never transfer. This closes the NVIDIA queue's
"CUDA higher-order package binding and exact SM120 execution remain open"
(2026-09-14); see `docs/audit/backend/nvidia/todo.md` §"sm_120 engineering loops".
