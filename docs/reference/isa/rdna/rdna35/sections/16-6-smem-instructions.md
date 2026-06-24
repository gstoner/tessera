# 16.6. SMEM Instructions

> RDNA3.5 ISA — pages 271–275

16.6. SMEM Instructions

S_LOAD_B32                                                                                                   0

Load 32 bits of data from the scalar memory into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B64                                                                                                   1

Load 64 bits of data from the scalar memory into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B128                                                                                                  2

Load 128 bits of data from the scalar memory into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B256                                                                                                  3

Load 256 bits of data from the scalar memory into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32;
  SDATA[159 : 128] = MEM[ADDR + 16U].b32;
  SDATA[191 : 160] = MEM[ADDR + 20U].b32;
  SDATA[223 : 192] = MEM[ADDR + 24U].b32;
  SDATA[255 : 224] = MEM[ADDR + 28U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_LOAD_B512                                                                                                  4

Load 512 bits of data from the scalar memory into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32;
  SDATA[159 : 128] = MEM[ADDR + 16U].b32;
  SDATA[191 : 160] = MEM[ADDR + 20U].b32;
  SDATA[223 : 192] = MEM[ADDR + 24U].b32;
  SDATA[255 : 224] = MEM[ADDR + 28U].b32;
  SDATA[287 : 256] = MEM[ADDR + 32U].b32;
  SDATA[319 : 288] = MEM[ADDR + 36U].b32;
  SDATA[351 : 320] = MEM[ADDR + 40U].b32;
  SDATA[383 : 352] = MEM[ADDR + 44U].b32;
  SDATA[415 : 384] = MEM[ADDR + 48U].b32;
  SDATA[447 : 416] = MEM[ADDR + 52U].b32;
  SDATA[479 : 448] = MEM[ADDR + 56U].b32;
  SDATA[511 : 480] = MEM[ADDR + 60U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B32                                                                                             8

Load 32 bits of data from a scalar buffer surface into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B64                                                                                             9

Load 64 bits of data from a scalar buffer surface into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B128                                                                                           10

Load 128 bits of data from a scalar buffer surface into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B256                                                                                           11

Load 256 bits of data from a scalar buffer surface into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32;
  SDATA[159 : 128] = MEM[ADDR + 16U].b32;
  SDATA[191 : 160] = MEM[ADDR + 20U].b32;
  SDATA[223 : 192] = MEM[ADDR + 24U].b32;
  SDATA[255 : 224] = MEM[ADDR + 28U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_BUFFER_LOAD_B512                                                                                           12

Load 512 bits of data from a scalar buffer surface into a scalar register.

  SDATA[31 : 0] = MEM[ADDR].b32;
  SDATA[63 : 32] = MEM[ADDR + 4U].b32;
  SDATA[95 : 64] = MEM[ADDR + 8U].b32;
  SDATA[127 : 96] = MEM[ADDR + 12U].b32;
  SDATA[159 : 128] = MEM[ADDR + 16U].b32;
  SDATA[191 : 160] = MEM[ADDR + 20U].b32;
  SDATA[223 : 192] = MEM[ADDR + 24U].b32;
  SDATA[255 : 224] = MEM[ADDR + 28U].b32;
  SDATA[287 : 256] = MEM[ADDR + 32U].b32;
  SDATA[319 : 288] = MEM[ADDR + 36U].b32;
  SDATA[351 : 320] = MEM[ADDR + 40U].b32;
  SDATA[383 : 352] = MEM[ADDR + 44U].b32;
  SDATA[415 : 384] = MEM[ADDR + 48U].b32;
  SDATA[447 : 416] = MEM[ADDR + 52U].b32;
  SDATA[479 : 448] = MEM[ADDR + 56U].b32;
  SDATA[511 : 480] = MEM[ADDR + 60U].b32

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

S_GL1_INV                                                                                                    32

Invalidate the GL1 cache only.

S_DCACHE_INV                                    33

Invalidate the scalar data L0 cache.
