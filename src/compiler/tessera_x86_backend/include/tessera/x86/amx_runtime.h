#pragma once
#include <cstdint>

namespace tessera { namespace x86 {

// Hardware feature query
bool tessera_x86_amx_supported();
bool tessera_x86_amx_int8_supported();
// Request Linux permission for AMX tile data (no-op on non-Linux)
bool tessera_x86_amx_enable_linux();

// A simple tile config builder (16x64 default shapes)
struct AMXTileConfig {
    // Tiles 0..15 rows and bytes-per-row
    unsigned char rows[16];
    unsigned short colsb[16]; // bytes per row
    unsigned char palette_id; // 1 or 2
};

void amx_build_default_bf16_config(AMXTileConfig& cfg);
void amx_load_config(const AMXTileConfig& cfg);
void amx_release();

}} // namespace
