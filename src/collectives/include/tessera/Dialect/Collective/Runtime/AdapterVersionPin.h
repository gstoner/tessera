#pragma once
//===- AdapterVersionPin.h - NCCL 2.22 / RCCL 2.22 version pin ----------===//
//
// Sprint G-9 + H-8 (2026-05-11) — Compile-time version checks for the
// NCCL (NVIDIA) and RCCL (AMD) collective libraries Tessera links
// against under CUDA 13.3 and ROCm 7.2.4 respectively.
//
// Both pins match the Python source of truth:
//   * TESSERA_TARGET_NCCL_MIN = "2.22" in python/tessera/compiler/gpu_target.py
//   * TESSERA_TARGET_RCCL_MIN = "2.22" in python/tessera/compiler/rocm_target.py
//
// When the build includes NCCL/RCCL headers (TESSERA_HAS_NCCL /
// TESSERA_HAS_RCCL defined), this header emits a `static_assert` that
// the installed library is at least the pinned version.  When the
// libraries are absent (CPU-only / mock build), the header is a no-op.
//
// Used by NCCLAdapter and RCCLAdapter in this directory.
//
//===----------------------------------------------------------------------===//

// Sprint G-9 / H-8 version constants — must match the Python source.
#define TESSERA_NCCL_MIN_MAJOR 2
#define TESSERA_NCCL_MIN_MINOR 22
#define TESSERA_RCCL_MIN_MAJOR 2
#define TESSERA_RCCL_MIN_MINOR 22

// Toolchain pins matching docs/backends/nvidia/kernel-inventory.md +
// docs/backends/rocm/kernel-inventory.md.
#define TESSERA_TARGET_CUDA_TOOLKIT "13.3"
#define TESSERA_TARGET_PTX_ISA      "9.3"
#define TESSERA_TARGET_ROCM         "7.2.4"
#define TESSERA_TARGET_HIP          "7.2.4"


#if defined(TESSERA_HAS_NCCL)
#include <nccl.h>
// NCCL exposes NCCL_MAJOR / NCCL_MINOR / NCCL_PATCH.
#if (NCCL_MAJOR < TESSERA_NCCL_MIN_MAJOR) || \
    ((NCCL_MAJOR == TESSERA_NCCL_MIN_MAJOR) && (NCCL_MINOR < TESSERA_NCCL_MIN_MINOR))
#error "Tessera requires NCCL >= 2.22 (matches TESSERA_TARGET_NCCL_MIN in gpu_target.py). " \
       "Set -DTESSERA_SKIP_NCCL_VERSION_CHECK=1 to override."
#endif
#endif // TESSERA_HAS_NCCL


#if defined(TESSERA_HAS_RCCL)
#include <rccl/rccl.h>
// RCCL also exposes NCCL_MAJOR/NCCL_MINOR (it's NCCL-compatible).
#if !defined(NCCL_MAJOR) || \
    (NCCL_MAJOR < TESSERA_RCCL_MIN_MAJOR) || \
    ((NCCL_MAJOR == TESSERA_RCCL_MIN_MAJOR) && (NCCL_MINOR < TESSERA_RCCL_MIN_MINOR))
#error "Tessera requires RCCL >= 2.22 (matches TESSERA_TARGET_RCCL_MIN in rocm_target.py). " \
       "Set -DTESSERA_SKIP_RCCL_VERSION_CHECK=1 to override."
#endif
#endif // TESSERA_HAS_RCCL


namespace tessera {
namespace collective {

/// Compile-time tag: which collective backends are linked into this build.
struct AdapterBuildInfo {
#if defined(TESSERA_HAS_NCCL)
    static constexpr bool kHasNCCL = true;
    static constexpr int  kNCCLMajor = NCCL_MAJOR;
    static constexpr int  kNCCLMinor = NCCL_MINOR;
#else
    static constexpr bool kHasNCCL = false;
    static constexpr int  kNCCLMajor = 0;
    static constexpr int  kNCCLMinor = 0;
#endif

#if defined(TESSERA_HAS_RCCL)
    static constexpr bool kHasRCCL = true;
    static constexpr int  kRCCLMajor = NCCL_MAJOR;
    static constexpr int  kRCCLMinor = NCCL_MINOR;
#else
    static constexpr bool kHasRCCL = false;
    static constexpr int  kRCCLMajor = 0;
    static constexpr int  kRCCLMinor = 0;
#endif

    static constexpr int kRequiredNCCLMajor = TESSERA_NCCL_MIN_MAJOR;
    static constexpr int kRequiredNCCLMinor = TESSERA_NCCL_MIN_MINOR;
    static constexpr int kRequiredRCCLMajor = TESSERA_RCCL_MIN_MAJOR;
    static constexpr int kRequiredRCCLMinor = TESSERA_RCCL_MIN_MINOR;
};

} // namespace collective
} // namespace tessera
