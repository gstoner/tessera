# 12.7. Alignment and Errors

> RDNA3 ISA — pages 140–140

             Field        Size    Description
             DATA0        8       operand, from the first valid data; if no valid data (i.e., EXEC==0), the operand
                                  is 0.

    • The input comes from the first valid data of DATA0.
    • If offset[5:2] is 8-15: The operation is mapped to 64b operation to take 2 dst registers as a combined one.
      The source data is still 32b. The post-op result is 64b and store back to the 2 dst registers. The return value
      takes 2 VGPRs.
    • If offset[5:2] is 0-7: The operation is mapped to normal 32b operation.
    • For ds_add_gs_reg_rtn, the atomic add operation is
        ◦ VDST[0] = GS_REG[offset0[5:2]][31:0]
       ◦ If (offset0[5:2] >= 8) VDST[1] = GS_REG[offset0[5:2]][63:32]
       ◦ GS_REG[offset0[4:2]] += DATA0
    • For ds_sub_gs_reg, the atomic sub operation is
        ◦ VDST[0] = GS_REG[offset0[5:2]][31:0]
       ◦ If (offset0[5:2] >= 8) VDST[1] = GS_REG[offset0[5:2]][63:32]
       ◦ GS_REG[offset0[4:2]] -= DATA0

12.7. Alignment and Errors
GDS and LDS operations (both direct & indexed) report Memory Violation (memviol) for misaligned atomics.
LDS handles misaligned indexed reads & writes, but only when SH_MEM_CONFIG. alignment_mode ==
UNALIGNED. Atomics must be aligned.

LDS Alignment modes (config-reg controlled, in SH_MEM_CONFIG):
    • ALIGNMENT_MODE_DWORD: Automatic alignment to multiple of element size
    • ALIGNMENT_MODE_UNALIGNED: No alignment requirements.

#       LDS Access     Source Inst       Controls                      Behavior
        Type           Types
1       Direct (Read   ALU ops           LDS_CONFIG.ADDR_OUT_ Out of range direct operations report memviol if
        Broadcast)                       OF_RANGE_REPORTING   ADDR_OUT_OF_RANGE_REPORTING is true.
2       Indexed        DS ops            LDS_CONFIG.ADDR_OUT_ Out of range atomic operations report memviol if
        Atomic         FLAT ops          OF_RANGE_REPORTING   ADDR_OUT_OF_RANGE_REPORTING is true.
3       Indexed Non- DS ops              LDS_CONFIG.ADDR_OUT_ the LSBs are ignored to force alignment. No memviol
        Atomic       FLAT ops            OF_RANGE_REPORTING   is generated.
                                                              Out of range indexed operations report memviol if
                                                              ADDR_OUT_OF_RANGE_REPORTING is true.
