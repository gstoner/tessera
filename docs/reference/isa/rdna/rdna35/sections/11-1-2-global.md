# 11.1.2. Global

> RDNA3.5 ISA — pages 125–125

used. The address from the VGPR points to the memory space for a specific DWORD of scratch data owned by
this thread. The hardware maps this address to the actual memory address that holds data for all of the threads
in the wave. Flat atomics which map into scratch: 4-byte atomics are supported, and 8-byte atomics return
MEMVIOL.

The wave supplies the offset (for space allocated to this wave) with every Flat request. This is stored in a
dedicated per-wave register: FLAT_SCRATCH, that holds a 64-bit byte address.

The aperture check occurs when VGPRs are read, with invalid addresses being routed to the texture unit. The
"aperture check" is performed before "inst_offset" is added into the address, so it is undefined what occurs if
the addition of inst_offset pushes the address into a different memory aperture.

                        (Hole) Addr[48]   Addr[47]   Addr[46]    Aperture
                        0                 x          x           Normal (global memory)
                        1                 0          0           Potential Private (scratch)
                        1                 0          1           Potential Shared (LDS)
                        1                 1          x           Invalid

Ordering
   Flat instructions may complete out of order with each other. If one Flat instruction finds all of its data in
   Texture cache, and the next finds all of its data in LDS, the second instruction might complete first. If the
   two fetches return data to the same VGPR, the result is unknown (order is not deterministic). Flat
   instructions decrement VMcnt in order for the threads that went to global memory and those are in order
   with other scratch, global, texture and buffer instructions. Separately each Flat instruction increments and
   decrements LGKMcnt. This is out-of-order with the VMcnt path but is in-order with other DS (LDS)
   instructions. Since the data for a Flat load can come from either LDS or the texture cache, and because
   these units have different latencies, there is a potential race condition with respect to the VMcnt/VScnt and
   LGKMcnt counters. Because of this, the only sensible S_WAITCNT value to use after Flat instructions is
   zero.

11.1.2. Global
Global operations transfer data between VGPR and global memory. Global instructions are similar to Flat, but
the programmer is responsible to make sure that no threads access LDS or private space. Because of this, no
LDS bandwidth is used by global instructions.

Since these instructions do not access LDS, only VMcnt (or VScnt) is used, not LGKMcnt. If a global instruction
does attempt to access LDS, the instruction returns MEMVIOL.

Global includes two instructions which do not use any VGPRs for addressing, just SGPRs and INST_OFFSET:
  • GLOBAL_LOAD_ADDTID_B32
  • GLOBAL_STORE_ADDTID_B32

11.1.3. Scratch
Scratch instructions are similar to global but they access a private (per-thread) memory space that is swizzled.
Because of this, no LDS bandwidth is used by scratch instructions. Scratch instructions also support multi-
DWORD access and mis-aligned access (although mis-aligned is slower).
