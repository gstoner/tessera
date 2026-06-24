# 15.8. Vector Memory Image Format

> RDNA3.5 ISA — pages 191–193

15.8. Vector Memory Image Format

15.8.1. MIMG

  Description       Memory Image Instructions

Memory Image instructions (MIMG format) can be between 2 and 3 DWORDs. There are two variations of the
instruction:

  • Normal, where the address VGPRs are specified in the "ADDR" field, and are a contiguous set of VGPRs.
    This is a 2-DWORD instruction.
  • Non-Sequential-Address (NSA), where each address VGPR is specified individually and the address VGPRs
    can be scattered. This version uses 1 extra DWORD to specify the individual address VGPRs.

                                                    Table 105. MIMG Fields
Field Name               Bits            Format or Description
NSA                      [0]             Non-sequential address. Specifies that an additional instruction DWORD exists
                                         holding up to 4 unique VGPR addresses.
DIM                      [4:2]           Dimensionality of the resource constant. Set to bits [3:1] of the resource type field.
UNRM                     [7]             Force address to be un-normalized. User must set to 1 for Image stores & atomics.
DMASK                    [11:8]          Data VGPR enable mask: 1 .. 4 consecutive VGPRs
                                         Reads: defines which components are returned:
                                         0=red,1=green,2=blue,3=alpha
                                         Writes: defines which components are written with data from VGPRs (missing
                                         components get 0).
                                         Enabled components come from consecutive VGPRs.
                                         E.G. dmask=1001 : Red is in VGPRn and alpha in VGPRn+1.
                                         For D16 writes, DMASK is only used as a word count: each bit represents 16 bits of
                                         data to be written starting at the LSB’s of VDATA, then MSBs, then VDATA+1 etc. Bit
                                         position is ignored.
SLC                      [12]            System Level Coherent. Used in conjunction with DLC to determine L2 cache
                                         policies.
DLC                      [13]            0 = normal, 1 = Device Coherent
GLC                      [14]            0 = normal, 1 = globally coherent (bypass L0 cache) or for atomics, return pre-op
                                         value to VGPR.
R128                     [15]            Resource constant size: 1 = 128bit, 0 = 256bit
A16                      [16]            Address components are 16-bits (instead of the usual 32 bits).
                                         When set, all address components are 16 bits (packed into 2 per DWORD), except:
                                         Texel offsets (3 6bit UINT packed into 1 DWORD)
                                         PCF reference (for "_C" instructions)
                                         Address components are 16b uint for image ops without sampler; 16b float with
                                         sampler.
D16                      [17]            Data components are 16-bits (instead of the usual 32 bits).
OP                       [25:18]         Opcode. See table below.
ENCODING                 [31:26]         'b111100

Field Name               Bits            Format or Description
VADDR                    [39:32]         Address of VGPR to supply first component of address.
VDATA                    [47:40]         Address of VGPR to supply first component of write data or receive first component
                                         of read-data.
SRSRC                    [52:48]         SGPR to supply T# (resource constant) in 4 or 8 consecutive SGPRs. It is missing 2
                                         LSB’s of SGPR-address since it is aligned to 4 SGPRs.
TFE                      [53]            Partially resident texture, texture fault enable.
LWE                      [54]            LOD Warning Enable. When set to 1, a texture fetch may return "LOD_CLAMPED =
                                         1".
SSAMP                    [62:58]         SGPR to supply S# (sampler constant) in 4 or 8 consecutive SGPRs. It is missing 2
                                         LSB’s of SGPR-address since it is aligned to 4 SGPRs.
ADDR1                    [71:64]         Second Address register or group. Present only when NSA=1.
ADDR2                    [79:72]         Third Address register or group. Present only when NSA=1.

                                   Table 106. MIMG Opcodes
Opcode # Name                                        Opcode # Name
0          IMAGE_LOAD                                42           IMAGE_SAMPLE_C_O
1          IMAGE_LOAD_MIP                            43           IMAGE_SAMPLE_C_D_O
2          IMAGE_LOAD_PCK                            44           IMAGE_SAMPLE_C_L_O
3          IMAGE_LOAD_PCK_SGN                        45           IMAGE_SAMPLE_C_B_O
4          IMAGE_LOAD_MIP_PCK                        46           IMAGE_SAMPLE_C_LZ_O
5          IMAGE_LOAD_MIP_PCK_SGN                    47           IMAGE_GATHER4
6          IMAGE_STORE                               48           IMAGE_GATHER4_L
7          IMAGE_STORE_MIP                           49           IMAGE_GATHER4_B
8          IMAGE_STORE_PCK                           50           IMAGE_GATHER4_LZ
9          IMAGE_STORE_MIP_PCK                       51           IMAGE_GATHER4_C
10         IMAGE_ATOMIC_SWAP                         52           IMAGE_GATHER4_C_LZ
11         IMAGE_ATOMIC_CMPSWAP                      53           IMAGE_GATHER4_O
12         IMAGE_ATOMIC_ADD                          54           IMAGE_GATHER4_LZ_O
13         IMAGE_ATOMIC_SUB                          55           IMAGE_GATHER4_C_LZ_O
14         IMAGE_ATOMIC_SMIN                         56           IMAGE_GET_LOD
15         IMAGE_ATOMIC_UMIN                         57           IMAGE_SAMPLE_D_G16
16         IMAGE_ATOMIC_SMAX                         58           IMAGE_SAMPLE_C_D_G16
17         IMAGE_ATOMIC_UMAX                         59           IMAGE_SAMPLE_D_O_G16
18         IMAGE_ATOMIC_AND                          60           IMAGE_SAMPLE_C_D_O_G16
19         IMAGE_ATOMIC_OR                           64           IMAGE_SAMPLE_CL
20         IMAGE_ATOMIC_XOR                          65           IMAGE_SAMPLE_D_CL
21         IMAGE_ATOMIC_INC                          66           IMAGE_SAMPLE_B_CL
22         IMAGE_ATOMIC_DEC                          67           IMAGE_SAMPLE_C_CL
23         IMAGE_GET_RESINFO                         68           IMAGE_SAMPLE_C_D_CL
24         IMAGE_MSAA_LOAD                           69           IMAGE_SAMPLE_C_B_CL
25         IMAGE_BVH_INTERSECT_RAY                   70           IMAGE_SAMPLE_CL_O
26         IMAGE_BVH64_INTERSECT_RAY                 71           IMAGE_SAMPLE_D_CL_O
27         IMAGE_SAMPLE                              72           IMAGE_SAMPLE_B_CL_O
28         IMAGE_SAMPLE_D                            73           IMAGE_SAMPLE_C_CL_O
29         IMAGE_SAMPLE_L                            74           IMAGE_SAMPLE_C_D_CL_O
30         IMAGE_SAMPLE_B                            75           IMAGE_SAMPLE_C_B_CL_O
31         IMAGE_SAMPLE_LZ                           84           IMAGE_SAMPLE_C_D_CL_G16
32         IMAGE_SAMPLE_C                            85           IMAGE_SAMPLE_D_CL_O_G16

Opcode # Name                            Opcode # Name
33         IMAGE_SAMPLE_C_D              86      IMAGE_SAMPLE_C_D_CL_O_G16
34         IMAGE_SAMPLE_C_L              95      IMAGE_SAMPLE_D_CL_G16
35         IMAGE_SAMPLE_C_B              96      IMAGE_GATHER4_CL
36         IMAGE_SAMPLE_C_LZ             97      IMAGE_GATHER4_B_CL
37         IMAGE_SAMPLE_O                98      IMAGE_GATHER4_C_CL
38         IMAGE_SAMPLE_D_O              99      IMAGE_GATHER4_C_L
39         IMAGE_SAMPLE_L_O              100     IMAGE_GATHER4_C_B
40         IMAGE_SAMPLE_B_O              101     IMAGE_GATHER4_C_B_CL
41         IMAGE_SAMPLE_LZ_O             144     IMAGE_GATHER4H
