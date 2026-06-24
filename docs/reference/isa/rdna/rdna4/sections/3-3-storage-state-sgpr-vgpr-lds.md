# 3.3. Storage State: SGPR, VGPR, LDS

> RDNA4 ISA — pages 25–25

3.2.2. EXECute Mask
The Execute mask (64-bit) controls which threads in the vector are executed. Each bit indicates how one thread
behaves for vector instructions: 1 = execute, 0 = do not execute. EXEC can be read and written via scalar
instructions, and can also be written as a result of a vector-alu compare. EXEC affects: vector-alu, vector-
memory, LDS , and export instructions. It does not affect scalar execution or branches.

Wave64 uses all 64 bits of the exec mask. Wave32 waves use only bits 31:0 and hardware does not act upon the
upper bits.

There is a summary bit (EXECZ) that indicates that the entire execute mask is zero. It can be used as a condition
for branches to skip code when EXEC is zero. For wave32, this reflects the state of EXEC[31:0].

3.2.2.1. Instruction Skipping: EXEC==0
The shader hardware may skip vector instructions when EXEC==0.

Instructions that are skipped result in the same wave state as if they had executed with EXEC==0: no wave state
changes.

Instructions that may be skipped are:

  • VALU - skip if EXEC == 0
     ◦ Not skipped if the instruction writes SGPRs/VCC
      ◦ Does not skip WMMA or SWMMAC ops
      ◦ This skipping is timing-dependent and might not occur depending on timing after a V_CMPX.
  • These are not skipped regardless of EXEC mask value , and are issued only once in wave64 mode
     ◦ V_NOP, V_PIPEFLUSH, V_READLANE, V_READFIRSTLANE, V_WRITELANE
      ◦ GLOBAL_INV, GLOBAL_WB, GLOBAL_WBINV
  • These are not skipped and are issued twice in wave64 mode regardless of EXEC mask value
     ◦ V_CMP that writes SGPR or VCC (not V_CMPX - may skip one pass but not both)
      ◦ Any VALU that writes an SGPR
  • Export Request - skip unless: Done==1 or if export target is POS0
     ◦ Skipped if the wave was created with SKIP_EXPORT=1
  • DS_param_load / DS_direct_load: are skipped when EXEC==0 and EXPcnt==0
  • LDS, Memory - typically do not skip
     ◦ VMEM can be skipped only if: EXEC == 0 and LOADcnt / STOREcnt / BVHcnt / SAMPLEcnt == 0
         ▪ FLAT also requires that DScnt==0 to be skipped.
         ▪ otherwise for wave64 one pass can be skipped if EXEC==0 for that half, but not both halves.
      ◦ LDS can be skipped only if: DScnt==0 and EXEC==0

3.3. Storage State: SGPR, VGPR, LDS
