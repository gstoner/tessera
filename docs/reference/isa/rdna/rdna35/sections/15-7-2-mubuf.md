# 15.7.2. MUBUF

> RDNA3.5 ISA — pages 189–190

Opcode # Name                                           Opcode # Name
6          TBUFFER_STORE_FORMAT_XYZ                     14          TBUFFER_STORE_D16_FORMAT_XYZ
7          TBUFFER_STORE_FORMAT_XYZW                    15          TBUFFER_STORE_D16_FORMAT_XYZW

15.7.2. MUBUF

    Description     Memory Untyped-Buffer Instructions

                                                    Table 103. MUBUF Fields
Field Name               Bits            Format or Description
OFFSET                   [11:0]          Address offset, unsigned byte.
SLC                      [12]            System Level Coherent. Used in conjunction with DLC to determine L2 cache
                                         policies.
DLC                      [13]            0 = normal, 1 = Device Coherent
GLC                      [14]            0 = normal, 1 = globally coherent (bypass L0 cache) or for atomics, return pre-op
                                         value to VGPR.
OP                       [25:18]         Opcode. See table below.
ENCODING                 [31:26]         'b111000
VADDR                    [39:32]         Address of VGPR to supply first component of address (offset or index). When both
                                         index and offset are used, index is in the first VGPR and offset in the second.
VDATA                    [47:40]         Address of VGPR to supply first component of write data or receive first component
                                         of read-data.
SRSRC                    [52:48]         SGPR to supply V# (resource constant) in 4 or 8 consecutive SGPRs. It is missing 2
                                         LSB’s of SGPR-address since it is aligned to 4 SGPRs.
TFE                      [53]            Partially resident texture, texture fault enable.
OFFEN                    [54]            1 = enable offset VGPR, 0 = use zero for address offset
IDXEN                    [55]            1 = enable index VGPR, 0 = use zero for address index
SOFFSET                  [63:56]         Address offset, unsigned byte.

                                           Table 104. MUBUF Opcodes
Opcode # Name                                                Opcode # Name
0          BUFFER_LOAD_FORMAT_X                              37         BUFFER_STORE_D16_HI_B16
1          BUFFER_LOAD_FORMAT_XY                             38         BUFFER_LOAD_D16_HI_FORMAT_X
2          BUFFER_LOAD_FORMAT_XYZ                            39         BUFFER_STORE_D16_HI_FORMAT_X
3          BUFFER_LOAD_FORMAT_XYZW                           43         BUFFER_GL0_INV
4          BUFFER_STORE_FORMAT_X                             44         BUFFER_GL1_INV
5          BUFFER_STORE_FORMAT_XY                            51         BUFFER_ATOMIC_SWAP_B32
6          BUFFER_STORE_FORMAT_XYZ                           52         BUFFER_ATOMIC_CMPSWAP_B32
7          BUFFER_STORE_FORMAT_XYZW                          53         BUFFER_ATOMIC_ADD_U32
8          BUFFER_LOAD_D16_FORMAT_X                          54         BUFFER_ATOMIC_SUB_U32
9          BUFFER_LOAD_D16_FORMAT_XY                         55         BUFFER_ATOMIC_CSUB_U32
10         BUFFER_LOAD_D16_FORMAT_XYZ                        56         BUFFER_ATOMIC_MIN_I32
11         BUFFER_LOAD_D16_FORMAT_XYZW                       57         BUFFER_ATOMIC_MIN_U32
12         BUFFER_STORE_D16_FORMAT_X                         58         BUFFER_ATOMIC_MAX_I32

Opcode # Name                             Opcode # Name
13         BUFFER_STORE_D16_FORMAT_XY     59      BUFFER_ATOMIC_MAX_U32
14         BUFFER_STORE_D16_FORMAT_XYZ    60      BUFFER_ATOMIC_AND_B32
15         BUFFER_STORE_D16_FORMAT_XYZW   61      BUFFER_ATOMIC_OR_B32
16         BUFFER_LOAD_U8                 62      BUFFER_ATOMIC_XOR_B32
17         BUFFER_LOAD_I8                 63      BUFFER_ATOMIC_INC_U32
18         BUFFER_LOAD_U16                64      BUFFER_ATOMIC_DEC_U32
19         BUFFER_LOAD_I16                65      BUFFER_ATOMIC_SWAP_B64
20         BUFFER_LOAD_B32                66      BUFFER_ATOMIC_CMPSWAP_B64
21         BUFFER_LOAD_B64                67      BUFFER_ATOMIC_ADD_U64
22         BUFFER_LOAD_B96                68      BUFFER_ATOMIC_SUB_U64
23         BUFFER_LOAD_B128               69      BUFFER_ATOMIC_MIN_I64
24         BUFFER_STORE_B8                70      BUFFER_ATOMIC_MIN_U64
25         BUFFER_STORE_B16               71      BUFFER_ATOMIC_MAX_I64
26         BUFFER_STORE_B32               72      BUFFER_ATOMIC_MAX_U64
27         BUFFER_STORE_B64               73      BUFFER_ATOMIC_AND_B64
28         BUFFER_STORE_B96               74      BUFFER_ATOMIC_OR_B64
29         BUFFER_STORE_B128              75      BUFFER_ATOMIC_XOR_B64
30         BUFFER_LOAD_D16_U8             76      BUFFER_ATOMIC_INC_U64
31         BUFFER_LOAD_D16_I8             77      BUFFER_ATOMIC_DEC_U64
32         BUFFER_LOAD_D16_B16            80      BUFFER_ATOMIC_CMPSWAP_F32
33         BUFFER_LOAD_D16_HI_U8          81      BUFFER_ATOMIC_MIN_F32
34         BUFFER_LOAD_D16_HI_I8          82      BUFFER_ATOMIC_MAX_F32
35         BUFFER_LOAD_D16_HI_B16         86      BUFFER_ATOMIC_ADD_F32
36         BUFFER_STORE_D16_HI_B8
