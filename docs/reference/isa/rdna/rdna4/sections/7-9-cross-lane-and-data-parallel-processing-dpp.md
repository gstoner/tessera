# 7.9. Cross-Lane and Data Parallel Processing (DPP)

> RDNA4 ISA — pages 94–95

Field           Size   Description
vdstX           8      Destination VGPR for X operation.
vdstY           7      Destination VGPR for Y operation. vdstY specifies bits [7:1]. The LSB of the destination address is:
                       ~vdstX[0]. vdstX and vdstY: one must be even and the other is an odd VGPR.

OPX Opcodes:

#             Opcode                                 #          Opcode
0             V_DUAL_FMAC_F32                        7          V_DUAL_MUL_DX9_ZERO_F32
1             V_DUAL_FMAAK_F32                       8          V_DUAL_MOV_B32
2             V_DUAL_FMAMK_F32                       9          V_DUAL_CNDMASK_B32
3             V_DUAL_MUL_F32                         10         V_DUAL_MAX_NUM_F32
4             V_DUAL_ADD_F32                         11         V_DUAL_MIN_NUM_F32
5             V_DUAL_SUB_F32                         12         V_DUAL_DOT2ACC_F32_F16
6             V_DUAL_SUBREV_F32                      13         V_DUAL_DOT2ACC_F32_BF16

OPY Opcodes:

#             Opcode                                 #          Opcode
0             V_DUAL_FMAC_F32                        9          V_DUAL_CNDMASK_B32
1             V_DUAL_FMAAK_F32                       10         V_DUAL_MAX_NUM_F32
2             V_DUAL_FMAMK_F32                       11         V_DUAL_MIN_NUM_F32
3             V_DUAL_MUL_F32                         12         V_DUAL_DOT2ACC_F32_F16
4             V_DUAL_ADD_F32                         13         V_DUAL_DOT2ACC_F32_BF16
5             V_DUAL_SUB_F32                         16         V_DUAL_ADD_NC_U32
6             V_DUAL_SUBREV_F32                      17         V_DUAL_LSHLREV_B32
7             V_DUAL_MUL_DX9_ZERO_F32                18         V_DUAL_AND_B32
8             V_DUAL_MOV_B32

V_CNDMASK_B32 is the "VOP2" form that uses VCC as the select. VCC counts as one SGPR read.

VOPD instruction pairs generate only a single exception if either or both raise an exception.

7.9. Cross-Lane and Data Parallel Processing (DPP)
The VALU offers a number of capabilities for transferring data between different lanes of a wave:

Instruction                    Function
V_PERM_B32                     Byte swizzle within 64-bits of source data; unique swizzle per lane (not cross-lane)
V_PERMLANE16_B32               Arbitrary gather-style lane swizzle with group of 16-lanes (0-15, 16-31, etc) with uniform
                               swizzle control
V_PERMLANE16_VAR_B32           Same as above, but unique lane-select per lane.
V_PERMLANEX16_B32              Same as V_PERMLANE16_B32, but access opposite group of 16-lanes: lanes 0-15 read from
                               lanes 16-31, and lanes 16-31 read from lanes 0-15.
V_PERMLANEX16_VAR_B32 Same as above, but unique lane-select per lane.
V_PERMLANE64_B32               Swap upper 32-lanes and lower 32-lanes. NOP for wave32.
DPP8                           ALU instruction that also has an arbitrary 8 lane swizzle within groups of 8 lanes (0..7, 8..15,
                               etc)

Instruction                   Function
DPP16                         ALU instruction that also has a swizzle selected from a menu of swizzles, selecting one
                              input within groups of 16-lanes.
DS_SWIZZLE                    LDS operation: swizzles within a group of 32 lanes from a fixed menu of swizzles (rotate,
                              broadcast, swap)
DS_PERMUTE / BPERMUTE         LDS operation: permute or backwards-permute across all lanes (64 for wave64)

Permutes are opcodes, and DPP is specified as an operand.

Data Parallel Processing (DPP) operations allow VALU instruction to select operands from different lanes
(threads) rather than just using a thread’s own data. DPP operations are indicated by the use of the inline
constant: DPP8, DPP8FI, or DPP16 in the SRC0 operand. Note that since SRC0 is set to the DPP value, the actual
VGPR address for SRC0 comes from the DPP DWORD.

One example of using DPP is for scan operations. A scan operation is one that computes a value per thread that
is based on the values of the previous threads and possibly itself. E.g. a running sum is the sum of the values
from previous threads in the vector. A reduction operation is essentially a scan that returns a single value from
the highest numbered active thread. A scan operation requires that the EXEC mask to be set to all 1’s for proper
operation. Unused threads (lanes) should be set prior to the scan to a value that does not affect the result.

There are two forms of the DPP instruction word:

  DPP8        allows arbitrary swizzling between groups of 8 lanes

  DPP16       allows a set of predefined swizzles between groups of 16 lanes

DPP may be used only with: VOP1, VOP2, VOPC, VOP3 and VOP3P (but not "packed math" ops).

                     Table 38. Which instructions support DPP
Encoding         Opcodes                             Rule
VOP1             All 64-bit opcodes                  NO DPP
                 READFIRSTLANE_B32                   NO DPP
                 SWAP                                NO DPP
                 V_NOP                               NO DPP
                 PERMLANE                            NO DPP
                 All Others                          Allow DPP
VOP2             All 64-bit Opcodes                  NO DPP
                 FMAMK/AK_F32/F16                    NO DPP
                 All Others                          Allow DPP
VOP3P            V_DOT4_I32_IU8                      NO DPP
                 V_DOT4_U32_U8
                 V_DOT8_I32_IU4
                 V_DOT8_U32_U4
                 V_PK_*
                 WMMA
                 V_FMA_MIX*                          Allow DPP
VINTERP          ALL                                 NO DPP
VOPD             ALL                                 NO DPP
