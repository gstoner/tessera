# 15.7. Vector Memory Buffer Formats

> RDNA4 ISA — pages 206–207

15.7. Vector Memory Buffer Formats
There are two memory buffer instruction formats:

VBUFFER
     typed buffer access (data type is defined by the instruction) or untyped (defined by the resource-constant)

15.7.1. VBUFFER

    Description      Memory Typed-Buffer Instructions

                                              Table 111. VBUFFER Fields
Field Name        Bits        Format or Description
SOFFSET           [6:0]       Address offset from SGPR or NULL, unsigned byte.
OP                [21:14]     Opcode. See table below. (combined bits 53 with 18-16 to form opcode)
TFE               [22]        Partially resident texture, texture fault enable.
ENCODING          [31:26]     'b110001
VDATA             [39:32]     Address of VGPR to supply first component of write data or receive first component of
                              read-data.
RSRC              [49:41]     SGPR to supply V# (resource constant) in 4 or 8 consecutive SGPRs. Must be multiple of 4 in
                              the range 0-120.
SCOPE             [51:50]     Memory Scope
TH                [54:52]     Memory Temporal Hint
FORMAT            [61:55]     Data Format of data in memory buffer. See Buffer Image format Table
OFFEN             [62]        1 = enable offset VGPR, 0 = use zero for address offset
IDXEN             [63]        1 = enable index VGPR, 0 = use zero for address index
VADDR             [71:64]     Address of VGPR to supply first component of address (offset or index). When both index
                              and offset are used, index is in the first VGPR and offset in the second.
IOFFSET           [95:72]     Address offset, signed byte; must be non-negative.

                                         Table 112. VBUFFER Opcodes
Opcode # Name                                            Opcode # Name
0           BUFFER_LOAD_FORMAT_X                         56          BUFFER_ATOMIC_MIN_I32
1           BUFFER_LOAD_FORMAT_XY                        57          BUFFER_ATOMIC_MIN_U32
2           BUFFER_LOAD_FORMAT_XYZ                       58          BUFFER_ATOMIC_MAX_I32
3           BUFFER_LOAD_FORMAT_XYZW                      59          BUFFER_ATOMIC_MAX_U32
4           BUFFER_STORE_FORMAT_X                        60          BUFFER_ATOMIC_AND_B32
5           BUFFER_STORE_FORMAT_XY                       61          BUFFER_ATOMIC_OR_B32
6           BUFFER_STORE_FORMAT_XYZ                      62          BUFFER_ATOMIC_XOR_B32
7           BUFFER_STORE_FORMAT_XYZW                     63          BUFFER_ATOMIC_INC_U32
8           BUFFER_LOAD_D16_FORMAT_X                     64          BUFFER_ATOMIC_DEC_U32
9           BUFFER_LOAD_D16_FORMAT_XY                    65          BUFFER_ATOMIC_SWAP_B64
10          BUFFER_LOAD_D16_FORMAT_XYZ                   66          BUFFER_ATOMIC_CMPSWAP_B64

Opcode # Name                             Opcode # Name
11         BUFFER_LOAD_D16_FORMAT_XYZW    67      BUFFER_ATOMIC_ADD_U64
12         BUFFER_STORE_D16_FORMAT_X      68      BUFFER_ATOMIC_SUB_U64
13         BUFFER_STORE_D16_FORMAT_XY     69      BUFFER_ATOMIC_MIN_I64
14         BUFFER_STORE_D16_FORMAT_XYZ    70      BUFFER_ATOMIC_MIN_U64
15         BUFFER_STORE_D16_FORMAT_XYZW   71      BUFFER_ATOMIC_MAX_I64
16         BUFFER_LOAD_U8                 72      BUFFER_ATOMIC_MAX_U64
17         BUFFER_LOAD_I8                 73      BUFFER_ATOMIC_AND_B64
18         BUFFER_LOAD_U16                74      BUFFER_ATOMIC_OR_B64
19         BUFFER_LOAD_I16                75      BUFFER_ATOMIC_XOR_B64
20         BUFFER_LOAD_B32                76      BUFFER_ATOMIC_INC_U64
21         BUFFER_LOAD_B64                77      BUFFER_ATOMIC_DEC_U64
22         BUFFER_LOAD_B96                80      BUFFER_ATOMIC_COND_SUB_U32
23         BUFFER_LOAD_B128               81      BUFFER_ATOMIC_MIN_NUM_F32
24         BUFFER_STORE_B8                82      BUFFER_ATOMIC_MAX_NUM_F32
25         BUFFER_STORE_B16               86      BUFFER_ATOMIC_ADD_F32
26         BUFFER_STORE_B32               89      BUFFER_ATOMIC_PK_ADD_F16
27         BUFFER_STORE_B64               90      BUFFER_ATOMIC_PK_ADD_BF16
28         BUFFER_STORE_B96               128     TBUFFER_LOAD_FORMAT_X
29         BUFFER_STORE_B128              129     TBUFFER_LOAD_FORMAT_XY
30         BUFFER_LOAD_D16_U8             130     TBUFFER_LOAD_FORMAT_XYZ
31         BUFFER_LOAD_D16_I8             131     TBUFFER_LOAD_FORMAT_XYZW
32         BUFFER_LOAD_D16_B16            132     TBUFFER_STORE_FORMAT_X
33         BUFFER_LOAD_D16_HI_U8          133     TBUFFER_STORE_FORMAT_XY
34         BUFFER_LOAD_D16_HI_I8          134     TBUFFER_STORE_FORMAT_XYZ
35         BUFFER_LOAD_D16_HI_B16         135     TBUFFER_STORE_FORMAT_XYZW
36         BUFFER_STORE_D16_HI_B8         136     TBUFFER_LOAD_D16_FORMAT_X
37         BUFFER_STORE_D16_HI_B16        137     TBUFFER_LOAD_D16_FORMAT_XY
38         BUFFER_LOAD_D16_HI_FORMAT_X    138     TBUFFER_LOAD_D16_FORMAT_XYZ
39         BUFFER_STORE_D16_HI_FORMAT_X   139     TBUFFER_LOAD_D16_FORMAT_XYZW
51         BUFFER_ATOMIC_SWAP_B32         140     TBUFFER_STORE_D16_FORMAT_X
52         BUFFER_ATOMIC_CMPSWAP_B32      141     TBUFFER_STORE_D16_FORMAT_XY
53         BUFFER_ATOMIC_ADD_U32          142     TBUFFER_STORE_D16_FORMAT_XYZ
54         BUFFER_ATOMIC_SUB_U32          143     TBUFFER_STORE_D16_FORMAT_XYZW
55         BUFFER_ATOMIC_SUB_CLAMP_U32
