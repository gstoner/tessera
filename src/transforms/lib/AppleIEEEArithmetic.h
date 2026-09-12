// Integer f32 multiplication/division. No floating operation can flush a
// significand. Packing performs exactly one round-to-nearest, ties-to-even.
#ifndef TESSERA_APPLE_IEEE_ARITHMETIC_H
#define TESSERA_APPLE_IEEE_ARITHMETIC_H
static constexpr const char *appleIEEEArithmeticMSL = R"MSL(
inline ulong tessera_rne(ulong m, int shift, bool sticky) {
  if (shift <= 0) return m << (-shift);
  if (shift > 64) return 0;
  if (shift == 64) return m > (1ul << 63) || (m == (1ul << 63) && sticky) ? 1 : 0;
  ulong q = m >> shift;
  ulong r = m & ((1ul << shift) - 1ul);
  ulong halfway = 1ul << (shift - 1);
  return q + (r > halfway || (r == halfway && (sticky || (q & 1ul))));
}
inline uint tessera_pack_f32(uint sign, ulong m, int exponent, bool sticky) {
  if (!m) return sign;
  int top = 0; ulong bits = m;
  while (bits > 1ul) { bits >>= 1; ++top; }
  int e = top + exponent;
  if (e > 127) return sign | 0x7f800000u;
  if (e < -126) return sign | uint(tessera_rne(m, -149 - exponent, sticky));
  ulong rounded = tessera_rne(m, top - 23, sticky);
  if (rounded == (1ul << 24)) { rounded >>= 1; ++e; }
  if (e > 127) return sign | 0x7f800000u;
  return sign | (uint(e + 127) << 23) | (uint(rounded) & 0x7fffffu);
}
inline ulong tessera_shift_jam(ulong m, int shift) {
  if (!shift) return m;
  if (shift >= 64) return m != 0;
  return (m >> shift) | ((m & ((1ul << shift) - 1ul)) != 0);
}
inline float tessera_ieee_add(float x, float y) {
  uint a = as_type<uint>(x), b = as_type<uint>(y);
  uint ax = a & 0x7fffffffu, bx = b & 0x7fffffffu;
  if (ax > 0x7f800000u) return as_type<float>(a | 0x00400000u);
  if (bx > 0x7f800000u) return as_type<float>(b | 0x00400000u);
  if (ax == 0x7f800000u && bx == ax && ((a ^ b) >> 31)) return as_type<float>(0x7fc00000u);
  if (ax == 0x7f800000u) return x;
  if (bx == 0x7f800000u) return y;
  if (!ax && !bx) return as_type<float>(a & b & 0x80000000u);
  if (!ax) return y;
  if (!bx) return x;
  if (ax < bx) { uint t=a; a=b; b=t; t=ax; ax=bx; bx=t; }
  uint ea = ax >> 23, eb = bx >> 23;
  ulong ma = (ax & 0x7fffffu) | (ea ? 0x800000u : 0u);
  ulong mb = (bx & 0x7fffffu) | (eb ? 0x800000u : 0u);
  int exponent = ea ? int(ea) - 150 : -149;
  int other = eb ? int(eb) - 150 : -149;
  while (ma < 0x800000ul) { ma <<= 1; --exponent; }
  while (mb < 0x800000ul) { mb <<= 1; --other; }
  ma <<= 32;
  mb = tessera_shift_jam(mb << 32, exponent - other);
  ulong sum = ((a ^ b) >> 31) ? ma - mb : ma + mb;
  if (!sum) return 0.0f;
  return as_type<float>(tessera_pack_f32(a & 0x80000000u, sum, exponent - 32, false));
}
inline float tessera_ieee_sub(float x, float y) {
  return tessera_ieee_add(x, as_type<float>(as_type<uint>(y) ^ 0x80000000u));
}
inline float tessera_ieee_mul(float x, float y) {
  uint a = as_type<uint>(x), b = as_type<uint>(y);
  uint ax = a & 0x7fffffffu, bx = b & 0x7fffffffu;
  uint sign = (a ^ b) & 0x80000000u;
  if (ax > 0x7f800000u) return as_type<float>(a | 0x00400000u);
  if (bx > 0x7f800000u) return as_type<float>(b | 0x00400000u);
  if (ax == 0x7f800000u || bx == 0x7f800000u)
    return as_type<float>((!ax || !bx) ? 0x7fc00000u : sign | 0x7f800000u);
  uint ea = ax >> 23, eb = bx >> 23;
  ulong ma = (ax & 0x7fffffu) | (ea ? 0x800000u : 0u);
  ulong mb = (bx & 0x7fffffu) | (eb ? 0x800000u : 0u);
  int exponent = (ea ? int(ea) - 150 : -149) + (eb ? int(eb) - 150 : -149);
  return as_type<float>(tessera_pack_f32(sign, ma * mb, exponent, false));
}
inline float tessera_ieee_div(float x, float y) {
  uint a = as_type<uint>(x), b = as_type<uint>(y);
  uint ax = a & 0x7fffffffu, bx = b & 0x7fffffffu;
  uint sign = (a ^ b) & 0x80000000u;
  if (ax > 0x7f800000u) return as_type<float>(a | 0x00400000u);
  if (bx > 0x7f800000u) return as_type<float>(b | 0x00400000u);
  if ((!ax && !bx) || (ax == 0x7f800000u && bx == 0x7f800000u)) return as_type<float>(0x7fc00000u);
  if (ax == 0x7f800000u || !bx) return as_type<float>(sign | 0x7f800000u);
  if (!ax || bx == 0x7f800000u) return as_type<float>(sign);
  uint ea = ax >> 23, eb = bx >> 23;
  ulong ma = (ax & 0x7fffffu) | (ea ? 0x800000u : 0u);
  ulong mb = (bx & 0x7fffffu) | (eb ? 0x800000u : 0u);
  int exponent = (ea ? int(ea) - 150 : -149) - (eb ? int(eb) - 150 : -149);
  while (ma < 0x800000ul) { ma <<= 1; --exponent; }
  while (mb < 0x800000ul) { mb <<= 1; ++exponent; }
  ulong numerator = ma << 32;
  return as_type<float>(tessera_pack_f32(sign, numerator / mb, exponent - 32, numerator % mb != 0));
}
)MSL";
#endif
