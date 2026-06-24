# 3.4.8. SCRATCH_BASE

> RDNA4 ISA — pages 36–37

Whenever any instruction writes a value to VCC, the hardware automatically updates a "VCC summary" bit
called VCCZ. This bit indicates whether or not the entire VCC mask is zero for the current wave-size. Wave32
ignores VCC[63:32] and only bits[31:0] contribute to VCCZ. This is useful for early-exit branch tests. VCC is also set
for certain integer ALU operations (carry-out).

The EXEC mask determines which threads execute an instruction. The VCC indicates which executing threads
passed the conditional test, or which threads generated a carry-out from an integer add or subtract.

  S_MOV_B64       EXEC, 0x00000001       // set just one thread active; others are inactive
  V_CMP_EQ_B32    VCC, V0, V0         // compare (V0 == V0) and write result to VCC (all bits in VCC are updated)

                 VCC physically resides in the SGPR register file in a specific pair of SGPRs, so when an
                instruction sources VCC, that counts against the limit on the total number of SGPRs that can
                 be sourced for a given instruction.

Wave32 waves may use any SGPR for mask/carry/borrow operations, but may not use VCC_HI or EXEC_HI.

3.4.8. SCRATCH_BASE
SCRATCH_BASE is a 64-bit register that holds a pointer to the base of scratch memory for this wave. For waves
that have scratch space allocated, wave-launch hardware initializes the SCRATCH_BASE register with the
scratch base address unique to this wave. This register is read-only, except while in the trap handler where it is
writable. The value is a byte address and must be 256byte aligned. If the wave has no scratch space allocated,
then reading SCRATCH_BASE returns zero.

3.4.9. Hardware Internal Registers
These registers are read-only and can be accessed by the S_GETREG_B32 instruction. They return information
about hardware allocation and status. These registers are read-only unless otherwise specified.

HW_ID1

Field            Bits         Description
WAVE_ID          4:0          Wave id within the SIMD.
SIMD_ID          9:8          SIMD_ID within the WGP: [0] = CU within WGP, [1] = SIMD within CU.
WGP_ID           13:10        Physical WGP ID.
SA_ID            16           Shader Array ID

HW_ID2

Field                 Bits       Description
QUEUE_ID              3:0        Queue_ID (also encodes shader stage)
PIPE_ID               5:4        Pipeline ID
ME_ID                 9:8        MicroEngine ID: 0 = graphics, 1 & 2 = ACE compute
STATE_ID              14:12      State context ID
WG_ID                 20:16      Work-group ID (0-31) within the WGP.
VM_ID                 27:24      Virtual Memory ID

IB_STS2

Field                       Bits          Description

PERF_SNAPSHOT_DATA

Note that all PERF_SNAPSHOT registers can only be read by the wave that was snapshot (others read zero).
Users should read PERF_SNAPSHOT_PC_HI last, as reading this resets (unlocks) the perf_snapshot registers
for the next snapshot to be taken, as does the wave terminating.

Field                              Bits      Description
VALID                              0         1: snapshot data is written & sampled wave is reading it, 0: invalid or sampled
                                             wave is not the targeted wave
WAVE_ISSUE                         1         1: wave issued an instruction on the cycle it was snapshot; 0 = did not issue.
INST_TYPE                          5:2       Instruction type that was issued or wave wanted to issue:

                                              0    VALU                                8     Branch not taken
                                              1    Scalar                              9     Branch taken
                                              2    VMEM                                10    Jump
                                              3    LDS                                 11    Other
                                              4    LDS direct / Param                  12    None
                                              5    Export                              13    Dual VALU
                                              6    Message                             14    Flat
                                              7    Barrier                             15    VALU-Matrix
NO_ISSUE_REASON                    8:6       If an instruction was not issued, why?

                                              0          no instructions available
                                              1          waiting on ALU dependency
                                              2          waiting on s_waitcnt
                                              3          did not win arbitration vs. other waves
                                              4          sleep
                                              5          barrier wait
                                              6          other causes
                                              7          internal op
WAVE_ID                            13:9      Wave ID of wave that had snapshot taken

PERF_SNAPSHOT_DATA1

Field                              Bits      Description
WAVE_CNT                           5:0       number of waves active on this CU with this VMID
Issued Instruction                 12:6      SIMD issued an instruction of these types during the snapshot cycle

                                              6          Branch/Message
                                              7          Export
                                              8          LDS-Direct/param load
                                              9          LDS
                                              10         Texture/Vmem
                                              11         Scalar ALU/mem
                                              12         Vector ALU
