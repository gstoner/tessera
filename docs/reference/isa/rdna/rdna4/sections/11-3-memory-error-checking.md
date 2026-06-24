# 11.3. Memory Error Checking

> RDNA4 ISA — pages 150–150

  BUFFER_LOAD: Addr = T#.base + Soffset + swizzle( (Vidx + TID) * stride + Ioff + Voff)

Flat Addressing
                      Flat instructions use the "GV" addressing mode. Aperture test on the address determines:
                      Global/LDS/Scratch per thread. The aperture test uses only the base address, not the IOFFSET.
Normal (GV)           addr[63:0] = VGPRU64 + IOFFSETI24

Next the aperture check is performed on each thread to determine which memory space it falls into, using
"ADDR" from the previous table:

      Aperture check, given 64-bit address from VGPR:
             isLDS       = (ADDR[63:32] == { SH_MEM_BASES.SHARED_BASE[15:0],        16'b0000 }
             isScratch = (ADDR[63:32] == { SH_MEM_BASES.PRIVATE_BASE[15:0], 16'b0000 }
             isHole      =   ADDR[63:47] != (all zeros or all ones) && !isLDS && !isScratch
             isGlobal    = !isLDS && !isScratch && !isHole

See Memory Aperture Query for the definition of the aperture ranges.

Depending on which memory space each thread falls into a different addressing calculation applies:

      Aperture            Address Calculation
      GLOBAL              mem_addr = addr
      SCRATCH (SV) mem_addr = SCRATCH_BASE(sgpr:U64) + SWIZZLE(addr[31:0], ThreadID)
      LDS                 LDS_ADDR.U17 = VGPR(addr)[16:0] + IOFFSET[16:0]
                          LDS address math is truncated and may wrap around without being detected as out-of-range.
                          The only range check is: LDS_ADDR.U17 < LDS_SIZE (space allocated to wave), with the
                          LDS_ADDR zero-extended (not sign-extended) for the range-check.
      Hole                Memory Violation

11.3. Memory Error Checking
Both Cache and LDS can report that an error occurred due to a bad address. This can occur due to:
  • Invalid address (outside any aperture)
  • Write to read-only global memory address (page is read-only)
  • Misaligned data (scratch accesses may be misaligned)
  • Out-of-range address:
     ◦ LDS access with an address outside the range: [ 0, LDS_SIZE-1 ]

The policy for threads with bad addresses is: stores outside this range do not store a value, and reads return
zero. The aperture check for invalid address occurs before adding any address offsets - it is based only on the
base address; the other checks are performed after adding the offsets.

Addressing errors from either LDS or VMEM set the wave’s MEMVIOL bit, and also causes an exception (trap).
