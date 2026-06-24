# 16.20.2. VSCRATCH Instructions

> RDNA4 ISA — pages 686–690

16.20.2. VSCRATCH Instructions
Scratch instructions are like Flat, but assume all work-item addresses fall in scratch (private) space.

SCRATCH_LOAD_U8                                                                                                    16

Load 8 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })

SCRATCH_LOAD_I8                                                                                                    17

Load 8 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))

SCRATCH_LOAD_U16                                                                                                   18

Load 16 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })

SCRATCH_LOAD_I16                                                                                                   19

Load 16 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))

SCRATCH_LOAD_B32                                                                                                   20

Load 32 bits of data from the scratch aperture into a vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 0] = MEM[addr].b32

SCRATCH_LOAD_B64                                                                 21

Load 64 bits of data from the scratch aperture into a vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32

SCRATCH_LOAD_B96                                                                 22

Load 96 bits of data from the scratch aperture into a vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32

SCRATCH_LOAD_B128                                                                23

Load 128 bits of data from the scratch aperture into a vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32

SCRATCH_STORE_B8                                                                 24

Store 8 bits of data from a vector register into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b8 = VDATA[7 : 0]

SCRATCH_STORE_B16                                                                      25

Store 16 bits of data from a vector register into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b16 = VDATA[15 : 0]

SCRATCH_STORE_B32                                                                      26

Store 32 bits of data from vector input registers into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b32 = VDATA[31 : 0]

SCRATCH_STORE_B64                                                                      27

Store 64 bits of data from vector input registers into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]

SCRATCH_STORE_B96                                                                      28

Store 96 bits of data from vector input registers into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]

SCRATCH_STORE_B128                                                                     29

Store 128 bits of data from vector input registers into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b32 = VDATA[31 : 0];

  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]

SCRATCH_LOAD_D16_U8                                                                                                 30

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the low
16 bits of a 32-bit vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_I8                                                                                                 31

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_B16                                                                                                32

Load 16 bits of unsigned data from the scratch aperture and store the result into the low 16 bits of a 32-bit
vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.

SCRATCH_LOAD_D16_HI_U8                                                                                              33

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the high
16 bits of a 32-bit vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_I8                                                                                           34

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.

SCRATCH_LOAD_D16_HI_B16                                                                                          35

Load 16 bits of unsigned data from the scratch aperture and store the result into the high 16 bits of a 32-bit
vector register.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.

SCRATCH_STORE_D16_HI_B8                                                                                          36

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b8 = VDATA[23 : 16]

SCRATCH_STORE_D16_HI_B16                                                                                         37

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

  addr = CalcScratchAddr(v_addr_off.b64, s_saddr_off.b64);
  MEM[addr].b16 = VDATA[31 : 16]

SCRATCH_LOAD_BLOCK                                                                                               83

Load a block of data from the scratch aperture.

  for i in 0 : 31 do
