# 5.7.1. Memory Dependency Counters

> RDNA4 ISA — pages 62–65

Instructions                   Description
S_WAIT_STORECNT                Wait for the counter specified to be less-than or equal-to the count in SIMM16 before
S_WAIT_DSCNT                   continuing.
S_WAIT_KMCNT
S_WAIT_EXPCNT
S_WAIT_LOADCNT
S_WAIT_SAMPLECNT
S_WAIT_BVHCNT
S_WAIT_LOADCNT_DSCNT           Wait for the two counters specified to be less-than or equal-to the counts in SIMM16 before
                               continuing.
S_WAIT_STORECNT_DSCNT          SIMM16[15:8] = load/store
                               SIMM16[ 7:0] = DS
S_WAIT_EVENT                   Wait for an event to occur before proceeding
                               SIMM16[1] : 0=don’t wait, 1=wait for export-ready; other bits are reserved.
                               Any exception waits for this to complete before being processed, including: KILL, save-
                               context, host trap, memviol and anything that causes a trap to be taken.
S_DELAY_ALU                    Insert delay between dependent SALU/VALU instructions.
                               SIMM16[3:0] = InstID0
                               SIMM16[6:4] = InstSkip
                               SIMM16[10:7] = InstID1
                               This instruction describes dependencies for two instructions, directing the hardware to
                               insert delay if the dependent instruction was issued too recently to forward data to the
                               second. For details, see: S_DELAY_ALU.

To ensure correct operation some reads of SGPR registers must be guarded with s_wait_alu instructions. If a
subset of a 64b SGPR pair (i.e. s[0:1], s[2:3], s[4:5], etc) has been read by a VALU instruction, then reads of
subsequent values written to any subset of the same SGPR pair must be guarded as follows:

  • for VALU writes to VCC, VALU reads of VCC must wait using s_wait_alu 0xFFFD;
  • for VALU writes to an SGPR, VALU reads of the SGPR must wait using s_wait_alu 0xF1FF;
  • for SALU writes to an SGPR or VCC, SALU or VALU reads of the SGPR or VCC must wait using s_wait_alu
    0xFFFE.

      For example:
           v_mov_b32 v0, s1            // VALU reads pair s[0:1]
           s_mov_b32 s0, 0x42          // SALU writes s0
           s_wait_alu 0xFFFE           // Read of s0 must be guarded by waiting on the SALU to complete
           s_mov_b32 s2, s0            // SALU reads s0

5.7.1. Memory Dependency Counters
S_WAIT_*CNT waits for outstanding instructions that use the specified counter to complete to avoid data
hazards. Instructions within a type return in the order they were issued compared to other instructions of that
type (except for SMEM), but often return out of order with respect to instructions of different types.

Hardware prevents these counters from overflowing by stalling the issue of any instruction that by
incrementing the counter would cause it to overflow.

It is possible for load-data to be written to VGPRs out-of-order, but the counter-decrement still reflects in-order
completion. Stores from a wave are not kept in order with stores from that same wave when they write to

different addresses.

These counters count instructions, not threads.

This table describes the cases that increment and decrement the memory dependency counters.

Counter       Size     Description
LOADcnt       6        Vector memory count: image, buffer, flat, scratch, global (loads and atomic with return), global_inv.
                       Determines when memory loads have finished:

                         • Incremented by 1 every time a vector-memory load or atomic-with-return (VIMAGE, VBUFFER,
                           or FLAT/Scratch/Global format) instruction is issued.
                         • Decremented for loads when that instruction and those before it have completed.
SAMPLEcnt     6        Vector memory sample count.
                       Increments by 1 for each sample instruction issued, and decrements as they complete (in-order with
                       other SAMPLE ops).
BVHcnt        3        Vector memory BVH instruction count. Increments and decrements by 1 for each BVH instruction.
STOREcnt      6        Vector memory store count: image, buffer, flat, scratch, global (stores and atomic without return),
                       global_wb/wbinv.
                       Determines when memory stores have completed:

                         • Incremented every time a vector-memory store or atomic-without-return (VIMAGE, VBUFFER,
                           or Flat/Scratch/Global format) instruction is issued.
                         • Decremented for stores when the data has been written to the memory hierarchy level specified
                           in the SCOPE bits.
DScnt         6        Data-Store count (LDS and Flat)
                       Determines when one of these low-latency memory instructions have completed:

                         • Incremented by 1 for every LDS instruction issued
                         • Decremented by 1 for LDS loads or atomic-with-return when the data has been returned to
                           VGPRs.
                         • Decremented by 1 for LDS stores when the data has been written to LDS.
                         • Incremented by 1 for each Flat instruction issued, decremented when the LDS portion
                           completes.

                       Ordering:
                         • Instructions of different types are returned out-of-order.
                         • Instructions of the same type are returned in the order they were issued.
KMcnt         5        Constant Cache (Kcache) and Message Count

                         • Incremented by 1 by 32-bit and smaller loads, by 2 for larger loads
                         • Incremented by 1 for each S_SENDMSG issued. Decremented by 1 when message is sent out or
                           when data is returned.
                         • Decremented by 1 for each DWORD returned from the data-cache (SMEM).

                       Ordering:
                         • Instructions of different types are returned out-of-order.
                         • Instructions of the same type are returned in the order they were issued, except scalar-memory-
                           loads, which can return out-of-order (in which case only S_WAIT_KMcnt 0 is the only legitimate
                           value).

Counter       Size   Description
EXPcnt        3      VGPR-export count, and LDS-direct/param-load count.

                     Determines when data has been read out of the VGPR and sent out, at which time it is safe to
                     overwrite the contents of that VGPR.
                       • Incremented when an Export instruction is issued
                       • Decremented for exports when the last cycle of the export instruction is granted and executed
                         (VGPRs read out).
                       • Incremented when DS_PARAM_LOAD and DS_DIRECT_LOAD are issued, and decremented
                         when they complete and write data to VGPRs.

                     Ordering:
                       • Exports are kept in order only within each export type (color/null, position, primitive, dual-
                         source blend), but not between types.
                       • DS_PARAM_LOAD is ordered with DS_DIRECT_LOAD, but these are unordered with exports.

5.7.1.1. Scalar Memory
SMEM instructions use the KMcnt counter. The counter is incremented by 1 if only a single DWORD is read or
written, or incremented by 2 for any larger instruction. Cache invalidate instructions count 1 per scalar data
cache bank. SMEM instructions return out of order even to the same address, so the only sensible way to use
the counter is: S_WAIT_KMcnt <= 0.

The scalar data cache is read-only, and is not coherent with the first or second level vector memory cache.
Users must manually invalidate caches with register writes or shader instructions to stay coherent. The user
must use S_WAIT_KMcnt<=0 after sending cache-invalidate to confirm it has completed.

5.7.1.2. Vector Memory - Image, Buffer, Scratch, Global
LOADcnt counts the number of issued-but-not-completed instructions of type: load, atomic-with-return, and
cache-invalidate.

SAMPLEcnt counts the number of issued-but-not-completed instructions of type: sample (includes gather).

STOREcnt counts the number of memory-store and memory atomic-without-return data instructions issued but
not yet completed. It is only necessary to wait on STOREcnt when the shader must know that previous writes
are committed to memory before issuing subsequent writes. There is no hazard on issuing a memory store
followed by a subsequent instruction overwriting the source GPRs. There is also no hazard between a memory
store to one memory location followed by a load from the same location (if both are from the same wave) - the
load and store to the same address stay in order.

Global invalidate instructions are tracked with LOADcnt, and Global write-back (and write-back-invalidate) are
tracked with STOREcnt. The shader must use S_WAIT*CNT to know that these have completed.

Loads are kept in order with other loads and stores are ordered against other stores. Sample instruction are
kept in order with other samples.

5.7.1.3. Vector Memory - Flat
Since Flat instructions may have work-items going to vector memory and others going to LDS, Flat instructions
use both DScnt and LOADcnt/STOREcnt. Every Flat instruction increments both DScnt and either LOADcnt or
STOREcnt when the instruction issues, and both are decremented by the time the instruction completes but the
two counters are decremented at different times: DScnt when LDS is done with its work-items;
LOADcnt/STOREcnt when vector memory is done with its work-items. The LDS portion of the flat instruction is
placed in order with other LDS-indexed instructions, and the global memory portion is kept in order with other
buffer/image/global/scratch instructions.

Flat instructions effectively work as if both a VMEM and an LDS instruction were issued at the same time, with
complementary EXEC masks. Once launched, each half completes on its own schedule with no respect to the
other. This can introduce Write-After-Write hazards if multiple FLAT load instructions write to the same VGPR:
  A: flat_load_B32 v10, v[0:1]
  B: flat_load_B32 v10, v[2:3]

In this example, if "A" reads from vector memory, while "B" reads from LDS, it is likely B completes first, and
then A overwrites the data from B causing a Write-After-Write hazard.

Flat vs. LDS

   FLAT and LDS instructions issued from a single wave stay in order as they move through LDS and as they
   complete (and decrement DScnt). However the threads serviced by LDS complete out of order with
   respect to threads serviced by global memory.

5.7.1.4. LDS
LDS operations use the DScnt. The counter is incremented when the instruction is issued, and decremented
when the instruction completes. For writes, this means the data is written into LDS memory; for reads or
atomic-with-return, it means the return data is available in VGPRs. LDS instructions are kept in order.

LDS parameter and direct loads use EXPcnt to track when the loads have completed (VGPRs have been
written).

5.7.1.5. Export
Exports use the EXPcnt counter. This counter is used to prevent Write-after-Read hazards. When an export
instruction is issued, the EXPcnt counter is incremented and an export-request is made to the central export
arbiter. The shader program continues immediately, but must not overwrite any of the VGPRs holding data
which have yet to be exported until the export completes. Exports within each type complete in order, but
different types may complete out of order. For example, if a GS shader exports position and then parameters,
the parameter export request might be granted first.

Export instructions do not read the EXEC mask until after the export is granted (which can occur well after the
instruction is issued). The shader program must use "S_WAIT_EXPcnt 0" before overwriting the EXEC mask.
M0 is read when the export-request is made, so there is no WAR hazard on M0.
