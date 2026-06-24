# 15.7.1. MTBUF

> RDNA3.5 ISA — pages 188–188

15.7. Vector Memory Buffer Formats
There are two memory buffer instruction formats:

MTBUF
     typed buffer access (data type is defined by the instruction)

MUBUF
     untyped buffer access (data type is defined by the buffer / resource-constant)

15.7.1. MTBUF

    Description      Memory Typed-Buffer Instructions

                                                    Table 101. MTBUF Fields
Field Name                Bits           Format or Description
OFFSET                    [11:0]         Address offset, unsigned byte.
SLC                       [12]           System Level Coherent. Used in conjunction with DLC to determine L2 cache
                                         policies.
DLC                       [13]           0 = normal, 1 = Device Coherent
GLC                       [14]           0 = normal, 1 = globally coherent (bypass L0 cache) or for atomics, return pre-op
                                         value to VGPR.
OP                        [18:15]        Opcode. See table below.
FORMAT                    [25:19]        Data Format of data in memory buffer. See Buffer Image format Table
ENCODING                  [31:26]        'b111010
VADDR                     [39:32]        Address of VGPR to supply first component of address (offset or index). When both
                                         index and offset are used, index is in the first VGPR and offset in the second.
VDATA                     [47:40]        Address of VGPR to supply first component of write data or receive first component
                                         of read-data.
SRSRC                     [52:48]        SGPR to supply V# (resource constant) in 4 or 8 consecutive SGPRs. It is missing 2
                                         LSB’s of SGPR-address since it is aligned to 4 SGPRs.
TFE                       [53]           Partially resident texture, texture fault enable.
OFFEN                     [54]           1 = enable offset VGPR, 0 = use zero for address offset
IDXEN                     [55]           1 = enable index VGPR, 0 = use zero for address index
SOFFSET                   [63:56]        Address offset, unsigned byte.

                                          Table 102. MTBUF Opcodes
Opcode # Name                                           Opcode # Name
0           TBUFFER_LOAD_FORMAT_X                       8           TBUFFER_LOAD_D16_FORMAT_X
1           TBUFFER_LOAD_FORMAT_XY                      9           TBUFFER_LOAD_D16_FORMAT_XY
2           TBUFFER_LOAD_FORMAT_XYZ                     10          TBUFFER_LOAD_D16_FORMAT_XYZ
3           TBUFFER_LOAD_FORMAT_XYZW                    11          TBUFFER_LOAD_D16_FORMAT_XYZW
4           TBUFFER_STORE_FORMAT_X                      12          TBUFFER_STORE_D16_FORMAT_X
5           TBUFFER_STORE_FORMAT_XY                     13          TBUFFER_STORE_D16_FORMAT_XY
