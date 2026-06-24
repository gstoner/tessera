# 16.6. SMEM Instructions

> RDNA4 ISA — pages 289–298

16.6. SMEM Instructions

S_LOAD_B32                                                                                                   0

Load 32 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B64                                                                                                   1

Load 64 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B128                                                                                                  2

Load 128 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B256                                                                                                  3

Load 256 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B512                                                                                                  4

Load 512 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32;
  SDATA[287 : 256] = MEM[addr + 32U].b32;
  SDATA[319 : 288] = MEM[addr + 36U].b32;
  SDATA[351 : 320] = MEM[addr + 40U].b32;
  SDATA[383 : 352] = MEM[addr + 44U].b32;
  SDATA[415 : 384] = MEM[addr + 48U].b32;
  SDATA[447 : 416] = MEM[addr + 52U].b32;
  SDATA[479 : 448] = MEM[addr + 56U].b32;
  SDATA[511 : 480] = MEM[addr + 60U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B96                                                                                                       5

Load 96 bits of data from the scalar memory into a scalar register.

  addr = CalcGlobalAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_I8                                                                                                        8

Load 8 bits of signed data from the scalar memory, sign extend to 32 bits and store the result into a scalar
register.

  SDATA.i32 = 32'I(signext(MEM[ADDR].i8))

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_U8                                                                                                        9

Load 8 bits of unsigned data from the scalar memory, zero extend to 32 bits and store the result into a scalar
register.

  SDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_I16                                                                                                        10

Load 16 bits of signed data from the scalar memory, sign extend to 32 bits and store the result into a scalar
register.

  SDATA.i32 = 32'I(signext(MEM[ADDR].i16))

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_U16                                                                                                        11

Load 16 bits of unsigned data from the scalar memory, zero extend to 32 bits and store the result into a scalar
register.

  SDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B32                                                                                                 16

Load 32 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B64                                                                                            17

Load 64 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B128                                                                                           18

Load 128 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B256                                                                                           19

Load 256 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B512                                                                                           20

Load 512 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32;
  SDATA[287 : 256] = MEM[addr + 32U].b32;
  SDATA[319 : 288] = MEM[addr + 36U].b32;
  SDATA[351 : 320] = MEM[addr + 40U].b32;
  SDATA[383 : 352] = MEM[addr + 44U].b32;
  SDATA[415 : 384] = MEM[addr + 48U].b32;
  SDATA[447 : 416] = MEM[addr + 52U].b32;
  SDATA[479 : 448] = MEM[addr + 56U].b32;
  SDATA[511 : 480] = MEM[addr + 60U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B96                                                                                            21

Load 96 bits of data from a scalar buffer surface into a scalar register.

  addr = CalcBufferAddr(sgpr_base.b64, offset.b64);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_I8                                                                                                     24

Load 8 bits of signed data from a scalar buffer surface, sign extend to 32 bits and store the result into a scalar
register.

  SDATA.i32 = 32'I(signext(MEM[ADDR].i8))

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_U8                                                                                                     25

Load 8 bits of unsigned data from a scalar buffer surface, zero extend to 32 bits and store the result into a scalar
register.

  SDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_I16                                                                                                    26

Load 16 bits of signed data from a scalar buffer surface, sign extend to 32 bits and store the result into a scalar
register.

  SDATA.i32 = 32'I(signext(MEM[ADDR].i16))

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_U16                                                                                                    27

Load 16 bits of unsigned data from a scalar buffer surface, zero extend to 32 bits and store the result into a

scalar register.

  SDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_DCACHE_INV                                                                                                 33

Invalidate the scalar data L0 cache.

S_PREFETCH_INST                                                                                              36

Prefetch instructions into the shader instruction cache, relative to a base address provided.

If SCALAR_PREFETCH_EN is 0 for this wavefront then this instruction is treated as a NOP.

Scalar cache does not send completion or error status to the wave.

  if MODE.SCALAR_PREFETCH_EN.u1 then
        mem_addr = (64'U(S0[63 : 0].i64 + 64'I(IOFFSET.i24)) & 0xffffffffffffff80ULL);
        // Force 128B alignment
        length = S2.u32;
        // SGPR or M0
        length += SDATA.u32;
        // SDATA is an immediate
        length = (length & 31U);
        // Length restricted to 0..31
        length = (length + 1U) * 128U;
        // Prefetch 1-32 cachelines, units of 128B
        PrefetchScalarInst(mem_addr, length)
  endif

S_PREFETCH_INST_PC_REL                                                                                       37

Prefetch instructions into the shader instruction cache, relative to the current PC address.

If SCALAR_PREFETCH_EN is 0 for this wavefront then this instruction is treated as a NOP.

Scalar cache does not send completion or error status to the wave.

  if MODE.SCALAR_PREFETCH_EN.u1 then

      mem_addr = (64'U(PC[63 : 0].i64 + 8LL + 64'I(IOFFSET.i24)) & 0xffffffffffffff80ULL);
      // Force 128B alignment
      length = S1.u32;
      // SGPR or M0
      length += SDATA.u32;
      // SDATA is an immediate
      length = (length & 31U);
      // Length restricted to 0..31
      length = (length + 1U) * 128U;
      // Prefetch 1-32 cachelines, units of 128B
      PrefetchScalarInst(mem_addr, length)
  endif

S_PREFETCH_DATA                                                                                                 38

Prefetch data into the scalar data cache, relative to a base address provided.

If SCALAR_PREFETCH_EN is 0 for this wavefront then this instruction is treated as a NOP.

Scalar cache does not send completion or error status to the wave.

  if MODE.SCALAR_PREFETCH_EN.u1 then
      mem_addr = (64'U(S0[63 : 0].i64 + 64'I(IOFFSET.i24)) & 0xffffffffffffff80ULL);
      // Force 128B alignment
      length = S2.u32;
      // SGPR or M0
      length += SDATA.u32;
      // SDATA is an immediate
      length = (length & 31U);
      // Length restricted to 0..31
      length = (length + 1U) * 128U;
      // Prefetch 1-32 cachelines, units of 128B
      PrefetchScalarData(mem_addr, length)
  endif

S_BUFFER_PREFETCH_DATA                                                                                          39

Prefetch data into the scalar data cache, relative to a base address provided in a resource descriptor constant.

If SCALAR_PREFETCH_EN is 0 for this wavefront then this instruction is treated as a NOP.

Scalar cache does not send completion or error status to the wave.

  if MODE.SCALAR_PREFETCH_EN.u1 then
      mem_addr = (64'U(S0[47 : 0].i64 + 64'I(IOFFSET.i24)) & 0xffffffffffffff80ULL);
      // Force 128B alignment
      length = S2.u32;
      // SGPR or M0
      length += SDATA.u32;

        // SDATA is an immediate
        length = (length & 31U);
        // Length restricted to 0..31
        length = (length + 1U) * 128U;
        // Prefetch 1-32 cachelines, units of 128B
        PrefetchScalarData(mem_addr, length)
  endif

Notes

The scalar address operand is typically the first two dwords of a buffer resource constant. This instruction
masks and shifts the value to construct the equivalent byte address.

S_PREFETCH_DATA_PC_REL                                                                                         40

Prefetch data into the scalar data cache, relative to the current PC address.

If SCALAR_PREFETCH_EN is 0 for this wavefront then this instruction is treated as a NOP.

Scalar cache does not send completion or error status to the wave.

  if MODE.SCALAR_PREFETCH_EN.u1 then
        mem_addr = (64'U(PC[63 : 0].i64 + 8LL + 64'I(IOFFSET.i24)) & 0xffffffffffffff80ULL);
        // Force 128B alignment
        length = S1.u32;
        // SGPR or M0
        length += SDATA.u32;
        // SDATA is an immediate
        length = (length & 31U);
        // Length restricted to 0..31
        length = (length + 1U) * 128U;
        // Prefetch 1-32 cachelines, units of 128B
        PrefetchScalarData(mem_addr, length)
  endif
