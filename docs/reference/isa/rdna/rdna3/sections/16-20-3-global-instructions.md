# 16.20.3. Global Instructions

> RDNA3 ISA — pages 598–609

16.20.3. Global Instructions
Global instructions are like Flat, but assume all work-item addresses fall in global memory space.

GLOBAL_LOAD_U8                                                                                              16

Untyped buffer load unsigned byte, zero extend in data register.

  VDATA.u = 32'U({ 24'0, MEM[ADDR].u8 })

GLOBAL_LOAD_I8                                                                                              17

Untyped buffer load signed byte, sign extend in data register.

  VDATA.i = 32'I(signext(MEM[ADDR].i8))

GLOBAL_LOAD_U16                                                                                             18

Untyped buffer load unsigned short, zero extend in data register.

  VDATA.u = 32'U({ 16'0, MEM[ADDR].u16 })

GLOBAL_LOAD_I16                                                                                             19

Untyped buffer load signed short, sign extend in data register.

  VDATA.i = 32'I(signext(MEM[ADDR].i16))

GLOBAL_LOAD_B32                                                                                             20

Untyped buffer load dword.

  VDATA.b = MEM[ADDR].b

GLOBAL_LOAD_B64                                       21

Untyped buffer load 2 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b

GLOBAL_LOAD_B96                                       22

Untyped buffer load 3 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b;
  VDATA[95 : 64] = MEM[ADDR + 8U].b

GLOBAL_LOAD_B128                                      23

Untyped buffer load 4 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b;
  VDATA[95 : 64] = MEM[ADDR + 8U].b;
  VDATA[127 : 96] = MEM[ADDR + 12U].b

GLOBAL_STORE_B8                                       24

Untyped buffer store byte.

  MEM[ADDR].b8 = VDATA[7 : 0]

GLOBAL_STORE_B16                                      25

Untyped buffer store short.

  MEM[ADDR].b16 = VDATA[15 : 0]

GLOBAL_STORE_B32                                                              26

Untyped buffer store dword.

  MEM[ADDR].b = VDATA[31 : 0]

GLOBAL_STORE_B64                                                              27

Untyped buffer store 2 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32]

GLOBAL_STORE_B96                                                              28

Untyped buffer store 3 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32];
  MEM[ADDR + 8U].b = VDATA[95 : 64]

GLOBAL_STORE_B128                                                             29

Untyped buffer store 4 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32];
  MEM[ADDR + 8U].b = VDATA[95 : 64];
  MEM[ADDR + 12U].b = VDATA[127 : 96]

GLOBAL_LOAD_D16_U8                                                            30

Untyped buffer load unsigned byte, use low 16 bits of data register.

  VDATA[15 : 0].u16 = 16'U({ 8'0, MEM[ADDR].u8 });
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_I8                                                             31

Untyped buffer load signed byte, use low 16 bits of data register.

  VDATA[15 : 0].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_B16                                                            32

Untyped buffer load short, use low 16 bits of data register.

  VDATA[15 : 0].b16 = MEM[ADDR].b16;
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_HI_U8                                                          33

Untyped buffer load unsigned byte, use high 16 bits of data register.

  VDATA[31 : 16].u16 = 16'U({ 8'0, MEM[ADDR].u8 });
  // VDATA[15:0] is preserved.

GLOBAL_LOAD_D16_HI_I8                                                          34

Untyped buffer load signed byte, use high 16 bits of data register.

  VDATA[31 : 16].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[15:0] is preserved.

GLOBAL_LOAD_D16_HI_B16                                                         35

Untyped buffer load short, use high 16 bits of data register.

  VDATA[31 : 16].b16 = MEM[ADDR].b16;
  // VDATA[15:0] is preserved.

GLOBAL_STORE_D16_HI_B8                                                         36

Untyped buffer store byte, use high 16 bits of data register.

  MEM[ADDR].b8 = VDATA[23 : 16].b8

GLOBAL_STORE_D16_HI_B16                                                                                       37

Untyped buffer store short, use high 16 bits of data register.

  MEM[ADDR].b16 = VDATA[31 : 16].b16

GLOBAL_LOAD_ADDTID_B32                                                                                        40

Untyped buffer load dword. No VGPR address is supplied in this instruction. TID is added to the address as
shown below:

memory_Addr = sgpr_addr(64) + inst_offset(12) + tid*4

GLOBAL_STORE_ADDTID_B32                                                                                       41

Untyped buffer store dword. No VGPR address is supplied in this instruction. TID is added to the address as
shown below:

memory_Addr = sgpr_addr(64) + inst_offset(12) + tid*4

GLOBAL_ATOMIC_SWAP_B32                                                                                        51

Swap values in data register and memory.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = DATA.b;
  RETURN_DATA.b = tmp

GLOBAL_ATOMIC_CMPSWAP_B32                                                                                     52

Compare and swap with memory value.

  tmp = MEM[ADDR].b;
  src = DATA[31 : 0].b;

  cmp = DATA[63 : 32].b;
  MEM[ADDR].b = tmp == cmp ? src : tmp;
  RETURN_DATA.b = tmp

GLOBAL_ATOMIC_ADD_U32                                             53

Add data register to memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u += DATA.u;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_SUB_U32                                             54

Subtract data register from memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u -= DATA.u;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_CSUB_U32                                            55

Subtract data register from memory value, clamp to zero.

  declare new_value : 32'U;
  old_value = MEM[ADDR].u;
  if old_value < DATA.u then
      new_value = 0U
  else
      new_value = old_value - DATA.u
  endif;
  MEM[ADDR].u = new_value;
  RETURN_DATA.u = old_value

GLOBAL_ATOMIC_MIN_I32                                             56

Minimum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src < tmp ? src : tmp;

  RETURN_DATA.i = tmp

GLOBAL_ATOMIC_MIN_U32                                    57

Minimum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src < tmp ? src : tmp;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_MAX_I32                                    58

Maximum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src > tmp ? src : tmp;
  RETURN_DATA.i = tmp

GLOBAL_ATOMIC_MAX_U32                                    59

Maximum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src > tmp ? src : tmp;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_AND_B32                                    60

Bitwise AND of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp & DATA.b);
  RETURN_DATA.b = tmp

GLOBAL_ATOMIC_OR_B32                                                                           61

Bitwise OR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp | DATA.b);
  RETURN_DATA.b = tmp

GLOBAL_ATOMIC_XOR_B32                                                                          62

Bitwise XOR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp ^ DATA.b);
  RETURN_DATA.b = tmp

GLOBAL_ATOMIC_INC_U32                                                                          63

Increment memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_DEC_U32                                                                          64

Decrement memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u = tmp

GLOBAL_ATOMIC_SWAP_B64                                                                         65

Swap 64-bit values in data register and memory.

  tmp = MEM[ADDR].b64;

  MEM[ADDR].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_CMPSWAP_B64                                 66

Compare and swap with 64-bit memory value.

  tmp = MEM[ADDR].b64;
  src = DATA[63 : 0].b64;
  cmp = DATA[127 : 64].b64;
  MEM[ADDR].b64 = tmp == cmp ? src : tmp;
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_ADD_U64                                     67

Add data register to 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_SUB_U64                                     68

Subtract data register from 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_MIN_I64                                     69

Minimum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

GLOBAL_ATOMIC_MIN_U64                                           70

Minimum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_MAX_I64                                           71

Maximum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src > tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

GLOBAL_ATOMIC_MAX_U64                                           72

Maximum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src > tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_AND_B64                                           73

Bitwise AND of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_OR_B64                                            74

Bitwise OR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_XOR_B64                                                                                 75

Bitwise XOR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_INC_U64                                                                                 76

Increment 64-bit memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_DEC_U64                                                                                 77

Decrement 64-bit memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_CMPSWAP_F32                                                                             80

Compare and swap with floating-point memory value.

  tmp = MEM[ADDR].f;
  src = DATA[31 : 0].f;
  cmp = DATA[63 : 32].f;
  MEM[ADDR].f = tmp == cmp ? src : tmp;
  RETURN_DATA.f = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_MIN_F32                                      81

Minimum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src < tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_MAX_F32                                      82

Maximum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src > tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_ADD_F32                                      86

Add data register to floating-point memory value.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src + tmp;
  RETURN_DATA.f = tmp

Notes

Floating-point addition handles NAN/INF/denorm.
