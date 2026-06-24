# 4.1.1. Cache Controls: SCOPE and Temporal-Hint

> RDNA4 ISA — pages 49–52

                       Code            Meaning
Scalar     Scalar Dest 0-105           SGPR 0 .. 105        Scalar GPRs. One DWORD each.
Source     (7 bits)    106             VCC_LO               VCC[31:0]
(8 bits)               107             VCC_HI               VCC[63:32]
                       108-123         TTMP0 .. TTMP15      Trap handler temporary SGPRs (privileged)
                       124             NULL                 Reads return zero, writes are ignored. When used as an
                                                            SALU destination, nullifies the instruction.
                       125             M0                   Temporary register, use for a variety of functions
                       126             EXEC_LO              EXEC[31:0]
                       127             EXEC_HI              EXEC[63:32]
           Integer   128               0                    Inline constant zero
           Inline    129-192           int 1 .. 64          Integer inline constants
           Constants 193-208           int -1 .. -16
                       209-229         Reserved             Reserved
                       230             Reserved             Reserved
                       231             Reserved             Reserved
                       232             Reserved             Reserved
                       233             DPP8                 8-lane DPP (only valid as SRC0)
                       234             DPP8FI               8-lane DPP with Fetch-Invalid (only valid as SRC0)
                       235             SHARED_BASE          Memory Aperture Definition
                       236             SHARED_LIMIT
                       237             PRIVATE_BASE
                       238             PRIVATE_LIMIT
                       239             Reserved             Reserved
           Float     240               0.5                  Inline floating point constants. Can be used in 16, 32 and
           Inline    241               -0.5                 64 bit floating point math. They may be used with non-
           Constants 242               1.0                  float instructions but the value is treated as an integer
                                                            with the hex value of the float.
                       243             -1.0
                       244             2.0                  1/(2*PI) is 0.15915494. The hex values are:
                       245             -2.0                 half: 0x3118
                       246             4.0                  single: 0x3e22f983
                       247             -4.0                 double: 0x3fc45f306dc9c882
                       248             1.0 / (2 * PI)
                       249             Reserved             Reserved
                       250             DPP16                data parallel primitive (only valid as SRC0)
                       251             Reserved             Reserved
                       252             Reserved             Reserved
                       253             SCC                  { 31’b0, SCC }
                       254             Reserved             Reserved
                       255             Literal constant32   32 bit constant from instruction stream
Vector Src/Dst         256 - 511       VGPR 0 .. 255        Vector GPRs. One DWORD each.
(8 bits)

4.1.1. Cache Controls: SCOPE and Temporal-Hint
Scalar and Vector memory instructions contain two fields that control cache scope (which level of cache to
hold the data), and a temporal-hint to suggest how soon data might be reused.

See Cache System Hierarchy for a diagram of the cache hierarchy.

These bits also control memory atomic operations, indicating if they return the pre-op value or not.

The ISA SCOPE bits correspond to the 4 cache levels and indicates whether a given cache can do an operation
locally or whether it needs to forward the operation to a next level in the cache hierarchy to complete the
operation at the desired scope (coherence domain).

#         Name          Meaning
0         CU            Compute Unit (Work-group) Scope - coherent among all VMEM threads in a CU (work-group, but
                        not a WGP)
1         SE            Shader Engine - coherent among all clients (threads) sharing a SE-cache
2         DEV           Device - coherent among all threads on the same device
3         SYS           System

At each level of the cache hierarchy:
    • If the ISA.scope <= Cache-scope
         ◦ Reads can hit into the cache
       ◦ Writes into the cache and transaction is acknowledged from this level of cache
       ◦ Read-modify-write operations can occur locally in this cache
    • Else: must "forward" the operation to the next larger scope cache
        ◦ Forward = load cannot hit in this cache, store must propagate to the next higher cache (but updates
          state of this cache level as well)

A cache may be part of a coherence domain for certain memory pools, and not other memory pools, which
impacts how the CACHE_SCOPE is determined by the hardware for a given cache. For example, if a cache is
part of a full coherent data fabric for local memory, but access to remote memory (e.g. IO space) is via a non-
coherent fabric, then local memory could have CACHE_SCOPE = SYS, but remote memory is downgraded to
CACHE_SCOPE = DEV.

Acquire / Release and Scope

Release

When a thread is producing / writing data, it can perform a release function which can promote all prior writes
to a higher scope specified by the release. The hw ISA implementation of this is with a “WB” (write-back) op
with a scope. A cache may get a “WB” op w/ ISA.Scope..

    • If WB op and ISA.SCOPE > CACHE_SCOPE
         ◦ Flush all dirty blocks to next level in hierarchy
       ◦ Transaction completion is when all evicted dirty blocks have been acknowledged by next level in
         hierarchy
    • Else If WB op and ISA.SCOPE <= CACHE_SCOPE
        ◦ NOP – no flush is required, wb op is not forwarded to next level of hierarchy

Acquire

Similarly, a consuming / reading thread can perform an acquire function which can promote all subsequent
reads to the scope specified by the acquire. The hw ISA implementation of this is with a “INV” (invalidate) op
with a scope. A cache may get a “INV” op w/ ISA.Scope..

    • If INV op and ISA.SCOPE > CACHE_SCOPE
         ◦ Invalidate all blocks such that subsequent reads re-read data from the next level cache
    • Else If INV op and ISA.SCOPE <= CACHE_SCOPE
        ◦ NOP – no invalidate is required

Temporal Hint (TH) ISA bits provide an indicator of expected data re-use and is used for prioritization in
retention of data in the cache hierarchy.

Stores to a certain scope return 'done' from that scope (decrement STOREcnt).

                               Table 13. TH Policies for Load Ops
TH[2:0]     Code       Meaning
0           RT         regular temporal (default) for both near and far caches
1           NT         non-temporal (re-use not expected) for both near and far caches
2           HT         High-priority temporal (precedence over RT) for both near and far caches
3           LU         Last-use (non-temporal AND discard dirty if it hits)
4           NT_RT      non-temporal for near cache(s) and regular for far caches
5           RT_NT      regular for near cache(s) and non-temporal for far caches
6           NT_HT      non-temporal for near cache(s) and high-priority temporal for far caches
7           reserved

                               Table 14. TH Policies for Store Ops
TH[2:0]     Code       Meaning
0           RT         regular temporal (default) for both near and far caches (default wr-rinse)
1           NT         non-temporal (re-use not expected) for both near and far caches
2           HT         High-priority temporal (precedence over RT) for both near and far caches
                       (default wr-rinse)
3           WB         Same as “HT”, but also overrides wr-rinse in far cache where it forces to
                       stay dirty in cache
4           NT_RT      non-temporal for near cache(s) and regular for far caches
5           RT_NT      regular for near cache(s) and non-temporal for far caches
6           NT_HT      non-temporal for near cache(s) and HT for far caches
7           NT_WB      non-temporal for near cache(s) and WB for far cache

For Read-Modify-Write Atomic operations, the TH field is split into 3 independent fields

                           Table 15. TH Policies for RMW Atomic Ops
TH[0]        0 : non-returning atomic RMW operation (no data is returned to VGPR)
             1 : returning atomic RMW operation (data is returned to VGPR)
TH[1]        0 : "RT" – Regular (or default) temporal re-use is expected.
             1 : "NT" – Non-temporal where re-use is not expected. Also known as “stream”.
TH[2]        Cascade (deferred scope)
             0 : regular atomic op, do not defer the scope (TH[0] == 1 atomics are not deferred)
             1 – cascading atomic op, where scope is deferred if TH[0] == 0

    • Atomic operations have the option to return the value from memory before the atomic op is performed or
      not.
    • A cascading atomic op is for histogram (non-returning) type atomic ops where the full specified scope of
      the op is not realized until a subsequent release “synchronization/fence/atomic-operation” of a matching
      or higher scope occurs.

        Table 16. VMEM Policies for Writeback and Invalidate Ops
Scope          CU$                         L2$
0-CU           NOP                         NOP
1-SE           wb/wbinv/inv                NOP
2-DEV          wb/wbinv/inv                NOP
3-SYS          wb/wbinv/inv                wb/wbinv/inv

SMEM Scope Field for Load Ops:

SMEM ops have the same 2-bit SCOPE field with same definition as the VMEM-load op encoding. Note the "CU"
scope only means it has scope with other scalar threads on the same CU, not with vector threads. One must use
scope > 0 for synchronizing scalar and vector memory threads of a CU.

SMEM also supports a 2-bit TH field (same enums as TH[1:0] of VMEM TH[1:0] = 0 to 3); SMEM doesn’t support
independent near-cache and far-cache temporal hint, and the hint is uniform across all caches in the
hierarchy.

SMEM loads of scope==CU are not coherent with VMEM store/atomics.
