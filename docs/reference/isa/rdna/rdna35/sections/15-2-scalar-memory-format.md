# 15.2. Scalar Memory Format

> RDNA3.5 ISA — pages 162–162

15.2. Scalar Memory Format

15.2.1. SMEM

    Description     Scalar Memory data load

                                                    Table 75. SMEM Fields
Field Name               Bits            Format or Description
SBASE                    [5:0]           SGPR-pair that provides base address or SGPR-quad that provides V#. (LSB of SGPR
                                         address is omitted).
SDATA                    [12:6]          SGPR that provides write data or accepts return data.
DLC                      [14]            Device level coherent.
GLC                      [16]            Globally memory Coherent. Force bypass of L1 cache, or for atomics, cause pre-op
                                         value to be returned.
OP                       [25:18]         See Opcode table below.
ENCODING                 [31:26]         'b111101
OFFSET                   [52:32]         An immediate signed byte offset. Ignored for cache invalidations.
SOFFSET                  [63:57]         SGPR that supplies an unsigned byte offset. Disabled if set to NULL.

                        Table 76. SMEM Opcodes
Opcode # Name                        Opcode # Name
0          S_LOAD_B32                9               S_BUFFER_LOAD_B64
1          S_LOAD_B64                10              S_BUFFER_LOAD_B128
2          S_LOAD_B128               11              S_BUFFER_LOAD_B256
3          S_LOAD_B256               12              S_BUFFER_LOAD_B512
4          S_LOAD_B512               32              S_GL1_INV
8          S_BUFFER_LOAD_B32         33              S_DCACHE_INV
