#include "tessera/x86/amx_runtime.h"
#include <cstring>
#include <immintrin.h> // AMX intrinsics live here on recent compilers

#if defined(__linux__)
#include <sys/prctl.h>
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cpuid.h>
#endif

namespace tessera { namespace x86 {

static inline void cpuid(unsigned int leaf, unsigned int subleaf,
                         unsigned int& a, unsigned int& b,
                         unsigned int& c, unsigned int& d) {
#if defined(__linux__) || defined(__GNUC__)
    __cpuid_count(leaf, subleaf, a, b, c, d);
#else
    a=b=c=d=0;
#endif
}

// Returns true if CPU advertises AMX_TILE + AMX_BF16
bool tessera_x86_amx_supported() {
    unsigned int a,b,c,d;
    // CPUID.(EAX=7, ECX=0): EDX bits
    cpuid(7, 0, a,b,c,d);
    bool amx_tile = (d & (1u << 24)) != 0;
    bool amx_bf16 = (d & (1u << 22)) != 0;
    return amx_tile && amx_bf16;
}

bool tessera_x86_amx_enable_linux() {
#if defined(__linux__)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1022
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
    // Ask kernel for permission to use AMX tile data
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    // 0 on success, -1 on failure (EPERM/EINVAL if unsupported or denied)
    return rc == 0;
#else
    return true; // Non-Linux: assume OK or manage via OS-specific APIs
#endif
}

// Pack a 64-byte tile config structure per Intel SDM layout
// We keep a simple builder that zeros everything and fills rows/colsb.
static void pack_tilecfg_bytes(const AMXTileConfig& in, unsigned char out[64]) {
    std::memset(out, 0, 64);
    out[0] = in.palette_id; // palette
    // bytes 16..47: colsb (16 x uint16)
    for (int i=0;i<16;i++) {
        unsigned short v = in.colsb[i];
        out[16 + 2*i + 0] = (unsigned char)(v & 0xFF);
        out[16 + 2*i + 1] = (unsigned char)((v >> 8) & 0xFF);
    }
    // bytes 48..63: rows (16 x uint8)
    for (int i=0;i<16;i++) {
        out[48 + i] = in.rows[i];
    }
}

// Defaults for BF16 GEMM tiles:
// T0 (acc) 16x64 (fp32), T1 (A) 16x64, T2 (B) 64x16
void amx_build_default_bf16_config(AMXTileConfig& cfg) {
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1; // palette 1
    // Set all tiles to zero rows; we will use T0..T2
    for (int i=0;i<16;i++){ cfg.rows[i]=0; cfg.colsb[i]=0; }
    // Accumulator T0: 16 rows, 64 cols of FP32 => colsb = 64*4 = 256 bytes
    cfg.rows[0]  = 16; cfg.colsb[0]  = 256;
    // A tile T1: 16 rows, 64 cols of BF16 => colsb = 64*2 = 128 bytes
    cfg.rows[1]  = 16; cfg.colsb[1]  = 128;
    // B tile T2: 64 rows, 16 cols of BF16 => colsb = 16*2 = 32 bytes
    cfg.rows[2]  = 64; cfg.colsb[2]  = 32;
}

void amx_load_config(const AMXTileConfig& cfg) {
    unsigned char raw[64];
    pack_tilecfg_bytes(cfg, raw);
    _tile_loadconfig(raw);
}

void amx_release() {
    _tile_release();
}

}} // namespace


bool tessera_x86_amx_int8_supported() {
    unsigned int a,b,c,d;
    cpuid(7, 0, a,b,c,d);
    bool amx_tile = (d & (1u << 24)) != 0;
    bool amx_int8 = (d & (1u << 25)) != 0;
    return amx_tile && amx_int8;
}
