# 3.3.3. Dynamic VGPR Allocation & Deallocation

> RDNA4 ISA — pages 28–29

3.3.2.2. VGPR Out of Range Behavior
Given an instruction operand that uses one or more DWORDs of VGPR data: "V"

   Vs = the first VGPR DWORD (start)
   Ve = the last VGPR DWORD (end)

For a 32-bit operand, Vs==Ve; for a 64-bit operand Ve=Vs+1, etc.

Operand is out of range if:
  • Vs < 0 || Vs >= VGPR_SIZE
  • Ve < 0 || Ve >= VGPR_SIZE

V_MOVREL indexed operand out of range if either:
  • Index > 255
  • (Vs + M0) >= VGPR_SIZE
  • (Ve + M0) >= VGPR_SIZE

Out of range consequences:
  • If a dest VGPR is out of range, the instruction is ignored (treat as NOP). VMEM instructions (including LDS)
    issue the instruction as if EXEC==0 to keep LOADcnt/STOREcnt/… correct.
  • V_SWAP & V_SWAPREL : since both arguments are destinations, if either is out of range, discard the
    instruction.
      ◦ VALU instructions with multiple destination (e.g. VGPR and SGPR): nothing is written to any GPR
  • If a source VGPR is out of range in a VMEM or Export instruction: VGPR0 is used for the out of range VGPR
       ◦ Memory instructions that use a group of consecutive VGPRs that are out of range use VGPR0 for the
         individual out of range VGPRs.
  • If a source VGPR in a VALU instruction is out of range: act as if the instruction’s source field is set to
    VGPR0. Operands > 32 bits use consecutive VGPRs: V0, V1, V2, …
       ◦ VOPD has different rules: the source address forced to (VGPRaddr % 4).

Instructions with multiple destinations (e.g. V_ADD_CO): if any destination is out of range, no results are
written.

3.3.3. Dynamic VGPR Allocation & Deallocation
Compute Shaders may be launched in a mode where they can dynamically allocate and deallocate VGPRs;
dynamic VGPRs is not supported for graphics-shaders. Waves must be launched in "dynamic VGPR" mode to be
granted this ability; without it instructions requesting to alter the VGPR allocation size are ignored.

Dynamic VGPRs are supported only for wave32, not wave64.

Dynamic-VGPR workgroups take over a WGP (no mixing of dynamic and non-dynamic VGPR waves on a WGP):
if any workgroup is using dynamic VGPRs, only dynamic VGPR enabled workgroups or waves may be running
on that WGP. DVGPR workgroups take over a WGP when the workgroup is launched in WGP-mode, and take
over a CU when launched in CU-mode.

VGPRs are allocated/deallocated in blocks of 16 or 32 VGPRs (configurable) and are added to or removed from
the highest numbered VGPRs, keeping the range of available logical VGPRs contiguous starting from VGPR0.

Waves may allocate up to a maximum of 8 blocks of VGPRs and have a minimum of one block.

Block Size
   The VGPR block size is configurable to be 16-VGPRs with a maximum allocation of 128 VGPRs per wave, or
   32-VGPRs with a maximum allocation of 256 VGPRs per wave. This block-size is a chip-wide config; it
   cannot be modified per draw or dispatch. "Blocks" are also called "segments" in some contexts. Waves using
   block-size of 16-VGPRs must not access VGPRs above 127 - results are unpredictable.

Waves in dynamic VGPR mode are initialized with one VGPR-block allocated.

3.3.3.1. Instruction

  S_ALLOC_VGPR   <NumVgprs>     // Number of VGPRs wave now wants to own. Either inline-constant or SGPR
                                // NumVgprs is rounded up to the nearest BlockSize.

S_ALLOC_VGPR attempts to allocate (when: NumVgprs > currentVgprs) or deallocate (when: NumVgprs <
currentVgprs). Allocation requests can fail and return SCC=1 for success, SCC=0 for failure. Allocations do not
partially succeed - it’s all or nothing. If an allocation fails, the shader may retry again until it succeeds.
Deallocations do not fail, as do S_ALLOC_VGPR that do not change the allocation size. Only bits [8:0] of
NumVgprs are considered.

Allocations requesting more than the maximum number of VGPRs automatically fail and return SCC=0.
A wave may deallocate down to zero VGPRs. Deallocating down to zero VGPRs is not the same as "S_SENDMSG
DEALLOC_VGPRS", and this latter message must not be issued while using dynamic VGPRs.

VGPR out of bounds: the same out-of-bounds rules apply as when not using dynamic VGPRs.

A single-state (chip-wide) config register defines the maximum number of waves per SIMD that can be present
when using dynamic VGPRs: SQ_DYN_VGPR. Each of those wave-slots up to the maximum has 1 block of
VGPRs reserved whether or not that wave-slot is in use, but wave-slots past the maximum have no VGPRs
allocated. With blockSize=16 this means every wave-slot has 16 VGPRs regardless if there is a wave using it or not, and
the remaining VGPRs are available for allocation.

3.3.3.2. Deadlock Avoidance
Dynamically allocating VGPRs can lead to deadlock when all VGPRs have been allocated but every wave needs
to allocate more VGPRs to make progress. Hardware mitigates this with a mode that reserves just enough
VGPRs that at least one wave can reach the maximum VGPR allocation at all times. This does not prevent deadlock
when multiple waves require the maximum allocation to progress.

3.3.3.3. Context Save & Restore
Dynamic VGPRs are supported with Compute Wave Save Restore (CWSR). Waves must save off all VGPRs to
memory and on restore, using S_ALLOC_VGPR to first allocate VGPRs before restoring them from memory.
Waves are re-launched on restore in dynamic-vgpr-mode.
