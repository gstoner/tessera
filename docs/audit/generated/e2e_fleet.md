# E2E fleet release evidence

**Generated from the E2E-SPINE-3 packet registry and validated sealed packets. Do not hand-edit.**

`release_ready` is family-granular. It does not promote a whole target or transfer exact-device evidence.

| Target | Architecture | Backend | Family | State | Tested commit | Packet |
|---|---|---|---|---|---|---|
| `apple_gpu` | `apple7` | `apple` | `matmul` | `packet_pending` | - | - |
| `apple_gpu` | `apple7` | `apple` | `softmax` | `packet_pending` | - | - |
| `apple_gpu` | `apple7` | `apple` | `linalg` | `packet_pending` | - | - |
| `apple_gpu` | `apple7` | `apple` | `ppo` | `packet_pending` | - | - |
| `apple_gpu` | `apple7` | `apple` | `ebm` | `packet_pending` | - | - |
| `apple_gpu` | `apple7` | `apple` | `clifford` | `packet_pending` | - | - |
| `apple_cpu` | `apple_m1_max` | `apple` | `matmul` | `packet_pending` | - | - |
| `apple_cpu` | `apple_m1_max` | `apple` | `linalg` | `packet_pending` | - | - |
| `x86` | `x86_64_base` | `x86` | `softmax` | `release_ready` | `9f3757ef2dda` | `docs/audit/evidence/e2e_spine/x86/x86_64_base` |
| `x86` | `x86_64_base` | `x86` | `reduction` | `release_ready` | `9f3757ef2dda` | `docs/audit/evidence/e2e_spine/x86/x86_64_base` |
| `x86` | `x86_64_avx512` | `x86` | `matmul` | `packet_pending` | - | - |
| `x86` | `x86_64_avx512` | `x86` | `softmax` | `packet_pending` | - | - |
| `x86` | `x86_64_avx512` | `x86` | `reduction` | `packet_pending` | - | - |
| `x86` | `x86_64_avx512` | `x86` | `attention` | `packet_pending` | - | - |
| `x86` | `x86_64_avx512` | `x86` | `linalg` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `matmul` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `softmax` | `release_ready` | `9f3757ef2dda` | `docs/audit/evidence/e2e_spine/nvidia_sm120/sm_120a` |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `reduction` | `release_ready` | `9f3757ef2dda` | `docs/audit/evidence/e2e_spine/nvidia_sm120/sm_120a` |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `epilogue` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `attention` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `paged_kv` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `replay_ssm` | `packet_pending` | - | - |
| `nvidia_sm120` | `sm_120a` | `nvidia` | `moe` | `packet_pending` | - | - |
| `nvidia_sm90` | `sm_90a` | `nvidia` | `-` | `hardware_deferred` | - | - |
| `nvidia_sm100` | `sm_100a` | `nvidia` | `-` | `hardware_deferred` | - | - |
| `rocm_gfx1151` | `gfx1151` | `rocm` | `softmax` | `packet_pending` | - | - |
| `rocm_gfx1151` | `gfx1151` | `rocm` | `reduction` | `packet_pending` | - | - |
| `rocm_gfx1151` | `gfx1151` | `rocm` | `paged_kv` | `packet_pending` | - | - |
| `rocm_gfx1151` | `gfx1151` | `rocm` | `moe` | `packet_pending` | - | - |

The CSV companion retains the full reason and commit.
