# 15.8. Vector Memory Image Format

> RDNA4 ISA — pages 208–209

15.8. Vector Memory Image Format

    Description         Memory Image Instructions - image load, store and atomics (instructions that do not use a
                        sampler)

                                                  Table 113. VIMAGE Fields
Field Name        Bits           Format or Description
DIM               [2:0]          Dimensionality of the resource constant. Set to bits [3:1] of the resource type field.
R128              [4]            Resource constant size: 1 = 128bit, 0 = 256bit
D16               [5]            Data components are 16-bits (instead of the usual 32 bits).
A16               [6]            Address components are 16-bits (instead of the usual 32 bits).
OP                [21:14]        Opcode. See table below. (combined bits 53 with 18-16 to form opcode)
DMASK             [25:22]        Data VGPR enable mask: 1 .. 4 consecutive VGPRs
                                 Reads: defines which components are returned:
                                 0=red,1=green,2=blue,3=alpha
                                 Writes: defines which components are written with data from VGPRs (missing components
                                 replicate first component).
                                 Enabled components come from consecutive VGPRs.
                                 E.G. dmask=1001 : Red is in VGPRn and alpha in VGPRn+1.
                                 For D16 writes, DMASK is only used as a word count: each bit represents 16 bits of data to
                                 be written starting at the LSB’s of VDATA, then MSBs, then VDATA+1 etc. Bit position is
                                 ignored.
ENCODING          [31:26]        'b110100
VDATA             [39:32]        Address of VGPR to supply first component of write data or receive first component of
                                 read-data.
RSRC              [49:41]        SGPR to supply T# (resource constant) in 4 or 8 consecutive SGPRs. Must be multiple of 4 in
                                 the range 0-120.
SCOPE             [51:50]        Memory Scope
TH                [54:52]        Memory Temporal Hint
TFE               [55]           Partially resident texture, texture fault enable.
VADDR4            [56:63]        Address of VGPR to supply fifth component of address.
VADDR0            [71:64]        Address of VGPR to supply first component of address.
VADDR1            [79:72]        Address of VGPR to supply second component of address.
VADDR2            [87:80]        Address of VGPR to supply third component of address.
VADDR3            [95:88]        Address of VGPR to supply fourth component of address.

IMAGE_BVH ops use VADDR0-VADDR4 to specify component-groups instead of single components.

                                      Table 114. VIMAGE Opcodes
Opcode # Name                                     Opcode # Name
0          IMAGE_LOAD                             17          IMAGE_ATOMIC_MAX_UINT
1          IMAGE_LOAD_MIP                         18          IMAGE_ATOMIC_AND
2          IMAGE_LOAD_PCK                         19          IMAGE_ATOMIC_OR
3          IMAGE_LOAD_PCK_SGN                     20          IMAGE_ATOMIC_XOR
4          IMAGE_LOAD_MIP_PCK                     21          IMAGE_ATOMIC_INC_UINT

Opcode # Name                          Opcode # Name
5          IMAGE_LOAD_MIP_PCK_SGN      22      IMAGE_ATOMIC_DEC_UINT
6          IMAGE_STORE                 23      IMAGE_GET_RESINFO
7          IMAGE_STORE_MIP             25      IMAGE_BVH_INTERSECT_RAY
8          IMAGE_STORE_PCK             26      IMAGE_BVH64_INTERSECT_RAY
9          IMAGE_STORE_MIP_PCK         128     IMAGE_BVH_DUAL_INTERSECT_RAY
10         IMAGE_ATOMIC_SWAP           129     IMAGE_BVH8_INTERSECT_RAY
11         IMAGE_ATOMIC_CMPSWAP        131     IMAGE_ATOMIC_ADD_FLT
12         IMAGE_ATOMIC_ADD_UINT       132     IMAGE_ATOMIC_MIN_FLT
13         IMAGE_ATOMIC_SUB_UINT       133     IMAGE_ATOMIC_MAX_FLT
14         IMAGE_ATOMIC_MIN_INT        134     IMAGE_ATOMIC_PK_ADD_F16
15         IMAGE_ATOMIC_MIN_UINT       135     IMAGE_ATOMIC_PK_ADD_BF16
16         IMAGE_ATOMIC_MAX_INT
