# Canonical compilation spine inventory

**Generated from target capabilities, pipeline ownership, and the runtime execution matrix. Do not hand-edit.**

Level A is native/reference runtime execution, Level B is a typed compiler seam, and Level C is the canonical Graph→native-image→launch path. `partial` never implies fleet closure.

| Target | Driver pipeline | Declared pipeline | Resolution | A | B | C | Owner |
|---|---|---|---|---|---|---|---|
| `apple_cpu` | `tessera-lower-to-apple_cpu` | `tessera-lower-to-apple_cpu` | `declared_exact` | `native` | `partial` | `partial` | `apple` |
| `apple_gpu` | `tessera-lower-to-apple_gpu-runtime` | `tessera-lower-to-apple_gpu-runtime` | `declared_exact` | `native` | `partial` | `partial` | `apple` |
| `cpu` | `tessera-lower-to-x86` | `tessera-lower-to-x86` | `family_selector` | `native` | `partial` | `absent` | `shared_x86` |
| `nvidia_sm100` | `tessera-lower-to-gpu` | `tessera-lower-to-nvidia-sm100` | `declared_exact` | `absent` | `partial` | `absent` | `nvidia` |
| `nvidia_sm120` | `tessera-lower-to-gpu` | `tessera-lower-to-nvidia-sm120` | `declared_exact` | `native` | `partial` | `absent` | `nvidia` |
| `nvidia_sm80` | `tessera-lower-to-gpu` | `-` | `unsupported_no_exact_pipeline` | `absent` | `absent` | `absent` | `nvidia` |
| `nvidia_sm90` | `tessera-lower-to-gpu` | `tessera-lower-to-nvidia-sm90` | `declared_exact` | `absent` | `partial` | `absent` | `nvidia` |
| `rocm` | `tessera-lower-to-rocm` | `tessera-lower-to-rocm` | `family_selector` | `native` | `partial` | `absent` | `rocm` |
| `rocm_gfx1100` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx1151` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `native` | `partial` | `absent` | `rocm` |
| `rocm_gfx1200` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx1201` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx1250` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx90a` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx940` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx942` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `rocm_gfx950` | `tessera-target-artifact` | `tessera-lower-to-rocm` | `declared_shared_builder` | `absent` | `partial` | `absent` | `rocm` |
| `x86` | `tessera-lower-to-x86` | `tessera-lower-to-x86` | `declared_exact` | `native` | `partial` | `absent` | `x86` |

The canonical CSV companion retains target scope, runtime backend, and the complete resolution reason.
