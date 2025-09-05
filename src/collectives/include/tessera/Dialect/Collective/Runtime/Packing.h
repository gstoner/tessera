#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

namespace tessera { namespace collective {

inline uint16_t fp32_to_bf16(float f) {
  uint32_t x = *reinterpret_cast<uint32_t*>(&f);
  uint16_t hi = static_cast<uint16_t>(x >> 16);
  // round-to-nearest-even
  uint32_t lsb = (x >> 15) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  x += rounding_bias;
  return static_cast<uint16_t>(x >> 16);
}

// minimal E4M3 FP8 (clamped; bias=7)
inline uint8_t fp32_to_fp8_e4m3(float f) {
  if (!std::isfinite(f)) return 0; // clamp
  int s = std::signbit(f) ? 1 : 0;
  float af = std::fabs(f);
  if (af < std::ldexp(1.0f, -6)) return (uint8_t)(s<<7); // underflow to zero
  int e; float m = std::frexp(af, &e); // af = m * 2^(e-1), m in [0.5,1)
  e = e - 1 + 7; // bias
  if (e <= 0) e = 0;
  if (e >= 0xF) e = 0xF;
  // 3-bit mantissa
  int mant = (int)std::ldexp(m, 4) & 0x7; // 4 because hidden bit -> 3 mant bits
  return (uint8_t)((s<<7) | ((e & 0xF)<<3) | (mant & 0x7));
}

enum class WireDType { FP32, BF16, FP8 };

struct PackResult {
  std::vector<uint8_t> bytes;
  size_t in_elems = 0;
  size_t out_bytes = 0;
};

inline PackResult pack_cast_fp32(const float* in, size_t n, WireDType to, bool rle=false) {
  PackResult r; r.in_elems=n;
  if (to == WireDType::BF16) {
    r.bytes.resize(n * 2);
    for (size_t i=0;i<n;++i) {
      uint16_t b = fp32_to_bf16(in[i]);
      r.bytes[2*i] = (uint8_t)(b & 0xFF);
      r.bytes[2*i+1] = (uint8_t)(b >> 8);
    }
  } else if (to == WireDType::FP8) {
    r.bytes.resize(n);
    for (size_t i=0;i<n;++i) r.bytes[i] = fp32_to_fp8_e4m3(in[i]);
  } else { // FP32 passthrough
    r.bytes.resize(n * 4);
    std::memcpy(r.bytes.data(), in, n*4);
  }
  r.out_bytes = r.bytes.size();
  if (rle && !r.bytes.empty()) {
    // simple byte-level run-length encoding: [count(1B), value(1B)]*
    std::vector<uint8_t> enc; enc.reserve(r.bytes.size());
    uint8_t cur = r.bytes[0], cnt=1;
    for (size_t i=1;i<r.bytes.size();++i) {
      if (r.bytes[i]==cur && cnt<255) { ++cnt; }
      else { enc.push_back(cnt); enc.push_back(cur); cur=r.bytes[i]; cnt=1; }
    }
    enc.push_back(cnt); enc.push_back(cur);
    r.bytes.swap(enc); r.out_bytes = r.bytes.size();
  }
  return r;
}

}} // ns
