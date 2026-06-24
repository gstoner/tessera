# 16.20.2. Scratch Instructions

> RDNA3.5 ISA — pages 635–639

FLAT_ATOMIC_MAX_F32                                                                                                    82

Select the maximum of two single-precision float inputs, given two values stored in the data register and a
location in the flat aperture. Store the original value from flat aperture into a vector register iff the GLC bit is
set.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  MEM[ADDR].f32 = src > tmp ? src : tmp;
  RETURN_DATA.f32 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

FLAT_ATOMIC_ADD_F32                                                                                                    86

Add two single-precision float values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].f32;
  MEM[ADDR].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

16.20.2. Scratch Instructions
Scratch instructions are like Flat, but assume all work-item addresses fall in scratch (private) space.

SCRATCH_LOAD_U8                                                                                                        16

Load 8 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

  VDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })

SCRATCH_LOAD_I8                                                                                                    17

Load 8 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

  VDATA.i32 = 32'I(signext(MEM[ADDR].i8))

SCRATCH_LOAD_U16                                                                                                   18

Load 16 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

  VDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })

SCRATCH_LOAD_I16                                                                                                   19

Load 16 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

  VDATA.i32 = 32'I(signext(MEM[ADDR].i16))

SCRATCH_LOAD_B32                                                                                                   20

Load 32 bits of data from the scratch aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32

SCRATCH_LOAD_B64                                                                                                   21

Load 64 bits of data from the scratch aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32

SCRATCH_LOAD_B96                                                                                                   22

Load 96 bits of data from the scratch aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32;
  VDATA[95 : 64] = MEM[ADDR + 8U].b32

SCRATCH_LOAD_B128                                                                     23

Load 128 bits of data from the scratch aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32;
  VDATA[95 : 64] = MEM[ADDR + 8U].b32;
  VDATA[127 : 96] = MEM[ADDR + 12U].b32

SCRATCH_STORE_B8                                                                      24

Store 8 bits of data from a vector register into the scratch aperture.

  MEM[ADDR].b8 = VDATA[7 : 0]

SCRATCH_STORE_B16                                                                     25

Store 16 bits of data from a vector register into the scratch aperture.

  MEM[ADDR].b16 = VDATA[15 : 0]

SCRATCH_STORE_B32                                                                     26

Store 32 bits of data from vector input registers into the scratch aperture.

  MEM[ADDR].b32 = VDATA[31 : 0]

SCRATCH_STORE_B64                                                                     27

Store 64 bits of data from vector input registers into the scratch aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32]

SCRATCH_STORE_B96                                                                                                   28

Store 96 bits of data from vector input registers into the scratch aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32];
  MEM[ADDR + 8U].b32 = VDATA[95 : 64]

SCRATCH_STORE_B128                                                                                                  29

Store 128 bits of data from vector input registers into the scratch aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32];
  MEM[ADDR + 8U].b32 = VDATA[95 : 64];
  MEM[ADDR + 12U].b32 = VDATA[127 : 96]

SCRATCH_LOAD_D16_U8                                                                                                 30

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the low
16 bits of a 32-bit vector register.

  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_I8                                                                                                 31

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

  VDATA[15 : 0].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_B16                                                                                             32

Load 16 bits of unsigned data from the scratch aperture and store the result into the low 16 bits of a 32-bit
vector register.

  VDATA[15 : 0].b16 = MEM[ADDR].b16;
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_HI_U8                                                                                           33

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the high
16 bits of a 32-bit vector register.

  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_I8                                                                                           34

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

  VDATA[31 : 16].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_B16                                                                                          35

Load 16 bits of unsigned data from the scratch aperture and store the result into the high 16 bits of a 32-bit
vector register.

  VDATA[31 : 16].b16 = MEM[ADDR].b16;
  // VDATA[15:0] is preserved.

SCRATCH_STORE_D16_HI_B8                                                                                          36

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

  MEM[ADDR].b8 = VDATA[23 : 16]
