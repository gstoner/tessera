# 3.3.6. Scratch (Private) Memory

> RDNA4 ISA — pages 31–31

the boundary and be partially in each CU. The location of the LDS storage is unrelated to which CU the wave is
on.

Pixel parameters are loaded into the same CU side as the wave resides and do not cross over into the other side
of LDS storage. Pixel shaders are run only in CU mode. Pixel shader may request additional LDS space in addition
to what is required for vertex parameters.

3.3.5.1. LDS Alignment and Out-of-Range
Any DS_LOAD or DS_STORE of any size can be byte aligned if the alignment mode is set to "unaligned". For all
other alignment modes, LDS forces alignment by zeroing out address least significant bits.

  • 32-bit Atomics must be aligned to a 4-byte address; 64-bit atomics to an 8-byte address, otherwise they
    return MEMVIOL.
  • LDS operations report MEMVIOL if the LDS-address is out of range

Out Of Range
  • If the LDS-ADDRESS is out of range (addr < 0 or >= LDS_size):
       ◦ Loads return the value zero. For multi-DWORD loads, this is checked per DWORD and when any
         portion of a DWORD is out of range, it returns zero.
       ◦ Any portion of a store operation that is in range is written to LDS, and stored bytes out of range are
         discarded
       ◦ For both loads and stores the out of bounds check is performed at byte granularity for UNALIGNED
         mode, and DWORD granularity for DWORD alignment mode.
  • If any source-VGPR is out of range, the value from VGPR0 is used to supply the LDS address or data.
  • If the dest-VGPR is out of range, nullify the instruction (issue with EXEC=0)

"Native" Alignment in LDS is:

   B8: byte aligned
   B16 or D16: 2 byte aligned
   B32: 4 byte aligned
   B64: 8 byte aligned
   B128 and B96: 16 byte aligned

If the alignment mode is set to "unaligned", the LDS disables its auto-alignment and doesn’t report error for
misaligned loads & stores

           if (sh_alignment_mode == unaligned)      align = 0xffff
           else if (B32)                            align = 0xfffC
           else if (B64)                            align = 0xfff8
           else if (B96 or B128)                    align = 0xfff0
           LDSaddr = (addr + offset) & align

3.3.6. Scratch (Private) Memory
Waves may be allocated a block of global memory at wave-launch time that can be used for thread-private data.
