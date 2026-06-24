# 12.6.1. GS NGG Streamout Instructions

> RDNA3 ISA — pages 139–139

Field                 Size   Description
ADDR                  8      STACK_VGPR: Both a source and destination VGPR:
                             supplies the LDS stack address and is written back with updated address.
                             stack_addr[31:18] = stack_base[15:2] : stack base address (relative to allocated LDS space).
                             stack_addr[17:16] = stack_size[1:0] : 0=8DWORDs, 1=16, 2=32, 3=64 DWORDs per thread
                             stack_addr[15:0] = stack_index[15:0]. (bits [1:0] must be zero).
DATA0                 8      LVADDR: Last Visited Address. Is compared with data values (next field) to determine the next
                             node to visit.
DATA1                 8      4 VGPRs (X,Y,Z,W).
M0                    16     Unused.

12.6. Global Data Share
Global data Share is similar to LDS, but is a single memory accessible by all waves on the GPU. Global Data
Share uses the same instruction format as local data share (indexed operations only - no interpolation or direct
loads). Instructions increment the LGKMcnt for all loads, stores and atomics, and decrement LGKMcnt when
the instruction completes. GDS instructions support only one active lane per instruction. The first active lane
(based on EXEC) is used and others are ignored.

M0 is used for:
    • [15:0] holds SIZE, in bytes
    • [31:16] holds BASE address in bytes

12.6.1. GS NGG Streamout Instructions
The DS_ADD_GS_REG_RTN and DS_SUB_GS_REG_RTN instructions are used only by the GS stage, and are
used for streamout. These instructions perform atomic add or sub operations to data in dedicated registers, not
in GDS memory, and return the pre-op value. The source register is 32 bits and is an unsigned int. These 2
instructions increment the wave’s LGKMcnt, and decrement LGKMcnt when the instruction completes.

                                           Table 61. GDS Streamout Register Targets
offset[5:2]        Register                                            offset[5:2] Register
                   32-bit source, 32-bit dest & return value                       32-bit source, 64-bit dest & return value
0                  GDS_STRMOUT_DWORDS_WRITTEN_0                        8           GDS_STRMOUT_PRIMS_NEEDED_0
1                  GDS_STRMOUT_DWORDS_WRITTEN_1                        9           GDS_STRMOUT_PRIMS_WRITTEN_0
2                  GDS_STRMOUT_DWORDS_WRITTEN_2                        10          GDS_STRMOUT_PRIMS_NEEDED_1
3                  GDS_STRMOUT_DWORDS_WRITTEN_3                        11          GDS_STRMOUT_PRIMS_WRITTEN_1
4                  GDS_GS_0                                            12          GDS_STRMOUT_PRIMS_NEEDED_2
5                  GDS_GS_1                                            13          GDS_STRMOUT_PRIMS_WRITTEN_2
6                  GDS_GS_2                                            14          GDS_STRMOUT_PRIMS_NEEDED_3
7                  GDS_GS_3                                            15          GDS_STRMOUT_PRIMS_WRITTEN_3

                                 Table 62. DS_ADD_GS_REG_RTN* and DS_SUB_GS_REG_RTN:
              Field          Size   Description
              OP             8      ds_add_gs_reg_rtn, ds_sub_gs_reg_rtn
              OFFSET0        8      gs_reg_index[3:0]=offset0[5:2] indexes the GS register array
              VDST           8      VGPR to write pre-op value to
