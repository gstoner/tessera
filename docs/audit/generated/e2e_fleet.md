# E2E fleet release evidence

**Generated from the E2E-SPINE-3 packet registry and validated sealed packets. Do not hand-edit.**

`release_ready` is family-granular. It does not promote a whole target or transfer exact-device evidence.

| Target | Backend | Family | State | Tested commit | Packet |
|---|---|---|---|---|---|
| `apple_gpu` | `apple` | `matmul` | `packet_pending` | - | - |
| `apple_gpu` | `apple` | `softmax` | `packet_pending` | - | - |
| `apple_gpu` | `apple` | `linalg` | `packet_pending` | - | - |
| `apple_gpu` | `apple` | `ppo` | `packet_pending` | - | - |
| `apple_gpu` | `apple` | `ebm` | `packet_pending` | - | - |
| `apple_gpu` | `apple` | `clifford` | `packet_pending` | - | - |
| `apple_cpu` | `apple` | `matmul` | `packet_pending` | - | - |
| `apple_cpu` | `apple` | `linalg` | `packet_pending` | - | - |
| `x86` | `x86` | `matmul` | `packet_pending` | - | - |
| `x86` | `x86` | `softmax` | `packet_pending` | - | - |
| `x86` | `x86` | `reduction` | `packet_pending` | - | - |
| `x86` | `x86` | `attention` | `packet_pending` | - | - |
| `x86` | `x86` | `linalg` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `matmul` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `softmax` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `reduction` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `epilogue` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `attention` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `paged_kv` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `replay_ssm` | `packet_pending` | - | - |
| `nvidia_sm120` | `nvidia` | `moe` | `packet_pending` | - | - |
| `nvidia_sm90` | `nvidia` | `-` | `hardware_deferred` | - | - |
| `nvidia_sm100` | `nvidia` | `-` | `hardware_deferred` | - | - |
| `rocm_gfx1151` | `rocm` | `softmax` | `packet_pending` | - | - |
| `rocm_gfx1151` | `rocm` | `reduction` | `packet_pending` | - | - |
| `rocm_gfx1151` | `rocm` | `paged_kv` | `packet_pending` | - | - |
| `rocm_gfx1151` | `rocm` | `moe` | `packet_pending` | - | - |

The CSV companion retains the full reason and commit.
