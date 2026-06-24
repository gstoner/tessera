# 16.20.2. Scratch Instructions

> RDNA3 ISA — pages 593–597

Maximum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src > tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

FLAT_ATOMIC_ADD_F32                                                                                              86

Add data register to floating-point memory value.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src + tmp;
  RETURN_DATA.f = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

16.20.2. Scratch Instructions
Scratch instructions are like Flat, but assume all work-item addresses fall in scratch (private) space.

SCRATCH_LOAD_U8                                                                                                  16

Untyped buffer load unsigned byte, zero extend in data register.

  VDATA.u = 32'U({ 24'0, MEM[ADDR].u8 })

SCRATCH_LOAD_I8                                                                                                  17

Untyped buffer load signed byte, sign extend in data register.

  VDATA.i = 32'I(signext(MEM[ADDR].i8))

SCRATCH_LOAD_U16                                                           18

Untyped buffer load unsigned short, zero extend in data register.

  VDATA.u = 32'U({ 16'0, MEM[ADDR].u16 })

SCRATCH_LOAD_I16                                                           19

Untyped buffer load signed short, sign extend in data register.

  VDATA.i = 32'I(signext(MEM[ADDR].i16))

SCRATCH_LOAD_B32                                                           20

Untyped buffer load dword.

  VDATA.b = MEM[ADDR].b

SCRATCH_LOAD_B64                                                           21

Untyped buffer load 2 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b

SCRATCH_LOAD_B96                                                           22

Untyped buffer load 3 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b;
  VDATA[95 : 64] = MEM[ADDR + 8U].b

SCRATCH_LOAD_B128                                     23

Untyped buffer load 4 dwords.

  VDATA[31 : 0] = MEM[ADDR + 0U].b;
  VDATA[63 : 32] = MEM[ADDR + 4U].b;
  VDATA[95 : 64] = MEM[ADDR + 8U].b;
  VDATA[127 : 96] = MEM[ADDR + 12U].b

SCRATCH_STORE_B8                                      24

Untyped buffer store byte.

  MEM[ADDR].b8 = VDATA[7 : 0]

SCRATCH_STORE_B16                                     25

Untyped buffer store short.

  MEM[ADDR].b16 = VDATA[15 : 0]

SCRATCH_STORE_B32                                     26

Untyped buffer store dword.

  MEM[ADDR].b = VDATA[31 : 0]

SCRATCH_STORE_B64                                     27

Untyped buffer store 2 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32]

SCRATCH_STORE_B96                                                             28

Untyped buffer store 3 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32];
  MEM[ADDR + 8U].b = VDATA[95 : 64]

SCRATCH_STORE_B128                                                            29

Untyped buffer store 4 dwords.

  MEM[ADDR + 0U].b = VDATA[31 : 0];
  MEM[ADDR + 4U].b = VDATA[63 : 32];
  MEM[ADDR + 8U].b = VDATA[95 : 64];
  MEM[ADDR + 12U].b = VDATA[127 : 96]

SCRATCH_LOAD_D16_U8                                                           30

Untyped buffer load unsigned byte, use low 16 bits of data register.

  VDATA[15 : 0].u16 = 16'U({ 8'0, MEM[ADDR].u8 });
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_I8                                                           31

Untyped buffer load signed byte, use low 16 bits of data register.

  VDATA[15 : 0].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_B16                                                          32

Untyped buffer load short, use low 16 bits of data register.

  VDATA[15 : 0].b16 = MEM[ADDR].b16;
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_HI_U8                                                         33

Untyped buffer load unsigned byte, use high 16 bits of data register.

  VDATA[31 : 16].u16 = 16'U({ 8'0, MEM[ADDR].u8 });
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_I8                                                         34

Untyped buffer load signed byte, use high 16 bits of data register.

  VDATA[31 : 16].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_B16                                                        35

Untyped buffer load short, use high 16 bits of data register.

  VDATA[31 : 16].b16 = MEM[ADDR].b16;
  // VDATA[15:0] is preserved.

SCRATCH_STORE_D16_HI_B8                                                        36

Untyped buffer store byte, use high 16 bits of data register.

  MEM[ADDR].b8 = VDATA[23 : 16].b8

SCRATCH_STORE_D16_HI_B16                                                       37

Untyped buffer store short, use high 16 bits of data register.

  MEM[ADDR].b16 = VDATA[31 : 16].b16
