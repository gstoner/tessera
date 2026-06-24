# 3.3.3. Memory Alignment and Out-of-Range Behavior

> RDNA3.5 ISA — pages 27–27

  • (Vs + M0) >= VGPR_SIZE
  • (Ve + M0) >= VGPR_SIZE

Out of range consequences:
  • If a dest VGPR is out of range, the instruction is ignored (treat as NOP).
  • V_SWAP & V_SWAPREL : since both arguments are destinations, if either is out of range, discard the
    instruction.
      ◦ VALU instructions with multiple destination (e.g. VGPR and SGPR): nothing is written to any GPR
  • If a source VGPR is out of range in a VMEM or Export instruction: VGPR0 is used
       ◦ Memory instructions that use a group of consecutive VGPRs that are out of range use VGPR0 for the
         individual out of range VGPRs.
  • If a source VGPR in a VALU instruction is out of range in a VALU instruction: VGPR0
       ◦ VOPD has different rules: the source address forced to (VGPRaddr % 4).

Instructions with multiple destinations (e.g. V_ADD_CO): if any destination is out of range, no results are
written.

3.3.3. Memory Alignment and Out-of-Range Behavior
This section defines the behavior when a source or destination GPR or memory address is outside the legal
range for a wave. Except where noted, these rules apply to LDS, GDS, buffer, global, flat and scratch memory
accesses.

Memory, LDS & GDS: Reads and Atomics with return:
  • If any source VGPR or SGPR is out-of-range, the data value is undefined.
  • If any destination VGPR is out-of-range, the operation is nullified by issuing the instruction as if the EXEC
    mask were cleared to 0.
       ◦ This out-of-range test checks all VGPRs which could be returned (e.g. VDST to VDST+3 for a
         BUFFER_LOAD_B128)
       ◦ This check also includes the extra PRT (partially resident texture) VGPR and nullifies the fetch if this
         VGPR would be out of range no matter whether the texture system actually returns this value or not.
       ◦ Atomic operations with out-of-range destination VGPRs are nullified: issued, but with EXEC mask of
         zero.
  • Image loads and stores consider DMASK bits when making an out-of-bounds determination.
  • Note: VDST is only checked for lds/gds/mem-atomic that actually return a value.

VMEM (texture) memory alignment rules are defined using the config register:
SH_MEM_CONFIG.alignment_mode. This setting also affects LDS, Flat/Scratch/Global operations.

  DWORD              Automatic alignment to multiple of the smaller of element size or a DWORD.

  UNALIGNED          No alignment requirements.

Formatted ops such as BUFFER_LOAD_FORMAT_* must be aligned as follows:
  • 1-byte formats require 1-byte alignment
  • 2-byte formats require 2-byte alignment
  • 4-byte and larger formats require 4-byte alignment
