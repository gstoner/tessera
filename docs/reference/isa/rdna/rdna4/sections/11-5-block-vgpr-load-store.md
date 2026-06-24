# 11.5. Block VGPR Load & Store

> RDNA4 ISA — pages 151–151

11.4. Data
FLAT instructions can use from zero to four consecutive DWORDs of data in VGPRs and/or memory. The DATA
field determines which VGPR(s) supply source data (if any) and the VDST VGPRs hold return data (if any).
There is no data-format conversion performed.

"D16" instructions use only 16-bit of the VGPR instead of the full 32bits. "D16_HI" instructions read or write
only the high 16-bits, while "D16" use the low 16-bits.

11.5. Block VGPR Load & Store
These instructions can be used to efficiently move consecutive blocks of up to 32 VGPRs to and from memory:

                GLOBAL_LOAD_BLOCK              VDST, VADDR, SADDR, IOFFSET, (M0)
                GLOBAL_STORE_BLOCK             VSRC, VADDR, SADDR, IOFFSET, (M0)
                SCRATCH_LOAD_BLOCK             VDST, VADDR, SADDR, IOFFSET, (M0)
                SCRATCH_STORE_BLOCK            VSRC, VADDR, SADDR, IOFFSET, (M0)

Data is transferred from consecutive VGPRs to consecutive DWORDs in memory (per thread), with the option
of skipping some VGPRs (but still skipping the memory location - not compacted in memory).

The instruction fields are identical to SCRATCH and GLOBAL instructions except for the use of M0 to carry a
bitmask of which VGPRs to load/store (1) or skip (0). The LSB of M0 is for the first VGPR.

GLOBAL_LOAD and GLOBAL_STORE support the "GV" and "GVS" addressing modes, but not "GT";
SCRATCH_LOAD and SCRATCH_STORE support the "SS", "SV" and "SVS" addressing modes, but not "ST".

The block load/store addresses memory as if the IOFFSET field was incremented by 4 bytes for each VGPR,
regardless of whether it was actually transferred or not (based on M0 bits).

Block loads must load data into their own source VGPRs.

       for (n = 0..31)
             if (M0[n])
                 load/store using IOFFSET and VGPR[vdst/vdata + n]
             IOFFSET += 4 bytes

If M0==0, no data is transferred.

The entire block load/store is tracked with LOADcnt or STOREcnt: increments 1 for the entire block transfer,
and decrements when the block transfer has completed.

11.5.1. Error Handling
Out-of-range address VGPRs follow the same rules as Global and Scratch.

Out-of-range dest VGPRs for loads:
each DWORD of the instruction is individually checked for out-of-range. Those that are in-range execute
