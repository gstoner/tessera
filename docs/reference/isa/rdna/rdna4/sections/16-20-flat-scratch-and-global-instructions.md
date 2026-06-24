# 16.20. FLAT, Scratch and Global Instructions

> RDNA4 ISA — pages 671–685

16.20. FLAT, Scratch and Global Instructions
The bitfield map of the FLAT format is:

16.20.1. VFLAT Instructions
Flat instructions look at the per work-item address and determine for each work-item if the target memory
address is in global, private or scratch memory.

FLAT_LOAD_U8                                                                                                      16

Load 8 bits of unsigned data from the flat aperture, zero extend to 32 bits and store the result into a vector
register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })

FLAT_LOAD_I8                                                                                                      17

Load 8 bits of signed data from the flat aperture, sign extend to 32 bits and store the result into a vector
register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))

FLAT_LOAD_U16                                                                                                     18

Load 16 bits of unsigned data from the flat aperture, zero extend to 32 bits and store the result into a vector

register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })

FLAT_LOAD_I16                                                                                                     19

Load 16 bits of signed data from the flat aperture, sign extend to 32 bits and store the result into a vector
register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))

FLAT_LOAD_B32                                                                                                     20

Load 32 bits of data from the flat aperture into a vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 0] = MEM[addr].b32

FLAT_LOAD_B64                                                                                                     21

Load 64 bits of data from the flat aperture into a vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32

FLAT_LOAD_B96                                                                                                     22

Load 96 bits of data from the flat aperture into a vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32

FLAT_LOAD_B128                                                                     23

Load 128 bits of data from the flat aperture into a vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32

FLAT_STORE_B8                                                                      24

Store 8 bits of data from a vector register into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b8 = VDATA[7 : 0]

FLAT_STORE_B16                                                                     25

Store 16 bits of data from a vector register into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b16 = VDATA[15 : 0]

FLAT_STORE_B32                                                                     26

Store 32 bits of data from vector input registers into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b32 = VDATA[31 : 0]

FLAT_STORE_B64                                                                     27

Store 64 bits of data from vector input registers into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]

FLAT_STORE_B96                                                                                                     28

Store 96 bits of data from vector input registers into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]

FLAT_STORE_B128                                                                                                    29

Store 128 bits of data from vector input registers into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]

FLAT_LOAD_D16_U8                                                                                                   30

Load 8 bits of unsigned data from the flat aperture, zero extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.

FLAT_LOAD_D16_I8                                                                                                   31

Load 8 bits of signed data from the flat aperture, sign extend to 16 bits and store the result into the low 16 bits
of a 32-bit vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.

FLAT_LOAD_D16_B16                                                                                                  32

Load 16 bits of unsigned data from the flat aperture and store the result into the low 16 bits of a 32-bit vector
register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.

FLAT_LOAD_D16_HI_U8                                                                                                  33

Load 8 bits of unsigned data from the flat aperture, zero extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.

FLAT_LOAD_D16_HI_I8                                                                                                  34

Load 8 bits of signed data from the flat aperture, sign extend to 16 bits and store the result into the high 16 bits
of a 32-bit vector register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.

FLAT_LOAD_D16_HI_B16                                                                                                 35

Load 16 bits of unsigned data from the flat aperture and store the result into the high 16 bits of a 32-bit vector
register.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.

FLAT_STORE_D16_HI_B8                                                                                                 36

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);

  MEM[addr].b8 = VDATA[23 : 16]

FLAT_STORE_D16_HI_B16                                                                                            37

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the flat aperture.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  MEM[addr].b16 = VDATA[31 : 16]

FLAT_ATOMIC_SWAP_B32                                                                                             51

Swap an unsigned 32-bit integer value in the data register with a location in the flat aperture. Store the original
value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp

FLAT_ATOMIC_CMPSWAP_B32                                                                                          52

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in the flat
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_ADD_U32                                                                                              53

Add two unsigned 32-bit integer values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;

  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_SUB_U32                                                                                               54

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in the flat
aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables atomic
return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_SUB_CLAMP_U32                                                                                         55

Subtract an unsigned 32-bit integer location in the flat aperture from a value in the data register and clamp the
result to zero. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  declare new_value : 32'U;
  old_value = MEM[ADDR].u32;
  if old_value < DATA.u32 then
      new_value = 0U
  else
      new_value = old_value - DATA.u32
  endif;
  MEM[ADDR].u32 = new_value;
  RETURN_DATA.u32 = old_value

FLAT_ATOMIC_MIN_I32                                                                                               56

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

FLAT_ATOMIC_MIN_U32                                                                                               57

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_MAX_I32                                                                                               58

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

FLAT_ATOMIC_MAX_U32                                                                                               59

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_AND_B32                                                                                               60

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp

FLAT_ATOMIC_OR_B32                                                                                                   61

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

FLAT_ATOMIC_XOR_B32                                                                                                  62

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

FLAT_ATOMIC_INC_U32                                                                                                  63

Increment an unsigned 32-bit integer value from a location in the flat aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from flat aperture into a vector register iff
the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_DEC_U32                                                                                                  64

Decrement an unsigned 32-bit integer value from a location in the flat aperture with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from flat aperture into a
vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_SWAP_B64                                                                                             65

Swap an unsigned 64-bit integer value in the data register with a location in the flat aperture. Store the original
value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp

FLAT_ATOMIC_CMPSWAP_B64                                                                                          66

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in the flat
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from flat aperture into a vector register iff the temporal hint enables atomic return.

                RETURN_DATA[2:3] is not modified.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_ADD_U64                                                                                              67

Add two unsigned 64-bit integer values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;

  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_SUB_U64                                                                                               68

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in the flat
aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables atomic
return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_MIN_I64                                                                                               69

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

FLAT_ATOMIC_MIN_U64                                                                                               70

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_MAX_I64                                                                                               71

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a

location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

FLAT_ATOMIC_MAX_U64                                                                                               72

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_AND_B64                                                                                               73

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

FLAT_ATOMIC_OR_B64                                                                                                74

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

FLAT_ATOMIC_XOR_B64                                                                                                  75

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the temporal hint enables
atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

FLAT_ATOMIC_INC_U64                                                                                                  76

Increment an unsigned 64-bit integer value from a location in the flat aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from flat aperture into a vector register iff
the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_DEC_U64                                                                                                  77

Decrement an unsigned 64-bit integer value from a location in the flat aperture with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from flat aperture into a
vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

FLAT_ATOMIC_COND_SUB_U32                                                                                             80

Subtract an unsigned 32-bit integer value in the data register from a location in the flat aperture only if the
memory value is greater than or equal to the data register value. Store the original value from flat aperture into
a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = tmp >= src ? tmp - src : tmp;
  RETURN_DATA.u32 = tmp

FLAT_ATOMIC_MIN_NUM_F32                                                                                              81

Select the IEEE minimumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in the flat aperture. Update the flat aperture with the selected value. Store the original
value from flat aperture into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
      MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
      MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
      MEM[ADDR].f32 = src.f32
  elsif ((src.f32 < tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && sign(src.f32) &&
  !sign(tmp.f32))) then
      // NOTE: -0<+0 is TRUE in this comparison
      MEM[ADDR].f32 = src.f32
  else
      MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

FLAT_ATOMIC_MAX_NUM_F32                                                                                              82

Select the IEEE maximumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in the flat aperture. Update the flat aperture with the selected value. Store the original
value from flat aperture into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
      MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
      MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
      MEM[ADDR].f32 = src.f32
  elsif ((src.f32 > tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && !sign(src.f32) &&
  sign(tmp.f32))) then
      // NOTE: +0>-0 is TRUE in this comparison
      MEM[ADDR].f32 = src.f32
  else

        MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

FLAT_ATOMIC_ADD_F32                                                                                                86

Add two single-precision float values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  addr = CalcFlatAddr(v_addr.b64, flat_scratch.b64);
  tmp = MEM[addr].f32;
  MEM[addr].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

FLAT_ATOMIC_PK_ADD_F16                                                                                             89

Add a packed 2-component half-precision float value from the data register to a location in the flat aperture.
Store the original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].f16 = src[15 : 0].f16 + tmp[15 : 0].f16;
  dst[31 : 16].f16 = src[31 : 16].f16 + tmp[31 : 16].f16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

FLAT_ATOMIC_PK_ADD_BF16                                                                                            90

Add a packed 2-component BF16 float value from the data register to a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].bf16 = src[15 : 0].bf16 + tmp[15 : 0].bf16;
  dst[31 : 16].bf16 = src[31 : 16].bf16 + tmp[31 : 16].bf16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32
