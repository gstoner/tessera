# 11.1.2. Global

> RDNA4 ISA — pages 148–148

11.1. Instructions

11.1.1. FLAT
The Flat instruction set consists of loads, stores and memory atomic operations. Flat instructions do not use a
resource constant (V#) or sampler (S#), but they make use of wave state (SCRATCH_BASE) that holds scratch-
space information in case any threads' address resolves to scratch space. See Scratch section below.

Flat instructions are executed as both an LDS and a Global instruction, and so Flat instructions increment both
LOADcnt (or STOREcnt) and DScnt and are not considered done until both have been decremented. Loads,
atomics-with-return and GLOBAL_INV are tracked with LOADcnt. Stores, atomics-without-return,
GLOBAL_WB and GLOBAL_WBINV are tracked with STOREcnt.

When the address from a Flat instruction falls into scratch (private) space, a different ("swizzled") addressing
mechanism is used. The address (per thread) points to the memory space for a specific DWORD of scratch data
owned by this thread. The hardware maps this address to the actual memory address that holds data for all of
the threads in the wave. Flat atomics that map into scratch: 4-byte atomics are supported, and 8-byte atomics
return MEMVIOL.

The wave supplies the offset (for space allocated to this wave) with every Flat request. This is stored in a
dedicated per-wave register: SCRATCH_BASE, that holds a 64-bit byte address.

The aperture check occurs on the value read from VGPRs, with invalid addresses being routed to the texture
unit. The "aperture check" is performed before IOFFSET is added into the address, so it is undefined what
occurs if the addition of IOFFSET pushes the address into a different memory aperture.

For threads whose address fall into LDS space, the only address check applied is:
      Out of bounds = Logical_ADDR[16:0] >= Wave_allocated_LDS
If the thread is not out of bounds, the address is mapped into LDS space by discarding upper address bits. This
means many virtual addresses may map to the same physical LDS storage.

11.1.2. Global
Global operations transfer data between VGPR and global memory. Global instructions are similar to Flat, but
the programmer is responsible to make sure that no threads access LDS or private space. Because of this, no
LDS bandwidth is used by global instructions. Since these instructions do not access LDS, only LOADcnt (or
STOREcnt) is used, not DScnt.

Global includes two instructions that do not use any VGPRs for addressing, just SGPRs and IOFFSET:
  • GLOBAL_LOAD_ADDTID_B32
  • GLOBAL_STORE_ADDTID_B32

11.1.3. Scratch
Scratch instructions are similar to global but they access a private (per-thread) memory space that is swizzled.
Because of this, no LDS bandwidth is used by scratch instructions. Scratch instructions also support multi-
DWORD access and mis-aligned access (although mis-aligned is slower).
