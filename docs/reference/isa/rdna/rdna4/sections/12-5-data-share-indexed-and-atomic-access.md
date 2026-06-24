# 12.5. Data Share Indexed and Atomic Access

> RDNA4 ISA — pages 160–161

12.4. LDS Direct Load
Direct access is allowed only in CU mode, not WGP mode.

The DS_DIRECT_LOAD instruction reads a single DWORD from LDS and returns it to a VGPR, broadcasting it
to all active lanes in the wave. M0 provides the address and data type. DS_DIRECT_LOAD uses EXEC per quad,
not per pixel: if any pixel in a quad is enabled then the data is written to all 4 pixels in the quad.
DS_DIRECT_LOAD uses EXPcnt to track completion.

DS_DIRECT_LOAD uses the same instruction format and fields as DS_PARAM_LOAD. See Pixel Parameter
Interpolation.

  LDS_addr = M0[15:0] (byte address and must be DWORD aligned)
  DataType = M0[18:16]
      0 unsigned byte
      1 unsigned short
      2 DWORD
      3 unused
      4 signed byte
      5 signed short
      6,7 Reserved

  Example:   DS_DIRECT_LOAD   V4       // load the value from LDS-address in M0[15:0] to V4

Signed byte and short data is sign-extend to 32 bits before writing the result to a VGPR; unsigned byte and short
data is zero-extended to 32 bits before writing to a VGPR.

12.5. Data Share Indexed and Atomic Access
The data-share can perform indexed and atomic data share operations.

Indexed and atomic operations supply a unique address per work-item from the VGPRs to the LDS, and supply
or return unique data per work-item back to VGPRs. Due to the internal banked structure of LDS, operations
can complete in as little as one cycle, or take many more, depending upon the number of bank conflicts
(addresses that map to the same memory bank).

Indexed operations are simple LDS load and store operations that read data from, and return data to, VGPRs.

Atomic operations are arithmetic operations that combine data from VGPRs and data in LDS, and write the
result back to LDS. Atomic operations have the option of returning the LDS "pre-op" value to VGPRs.

LDS Indexed and atomic instructions use DScnt to track when they have completed. DScnt is incremented as
each instruction is issued, and decremented when they have completed execution. LDS instructions stay in-
order with other LDS instructions from the same wave.

The table below lists and briefly describes the LDS instruction fields.

                                             Table 71. VDS Instruction Fields
Field    Size Description
OP       8     LDS opcode.
OFFSET0 8      Immediate address offset. Interpretation varies with opcode:
               Instructions with one address: combine the offset fields into a 16-bit unsigned byte offset: {offset1,
               offset0}.
OFFSET1 8
               Instructions that have 2 addresses (e.g. {LOAD, STORE, XCHG}_2ADDR): use the offsets separately as 2 8-
               bit unsigned offsets. Each offset is multiplied by 4 for 8, 16 and 32-bit data; multiplied by 8 for 64-bit data.
VDST     8     VGPR to which result is written: either from LDS-load or atomic return value.
ADDR     8     VGPR that supplies the byte address offset.
DATA0    8     VGPR that supplies first data source.
DATA1    8     VGPR that supplies second data source.
M0       16    Unsigned byte Offset[15:0] used for: ds_load_addtid_b32, ds_write_addtid_b32

The M0 register is not used for most LDS-indexed operations: only the "ADDTID" instructions read M0 and for
these it represents a byte address.

                                             Table 72. LDS Indexed Load/Store
Load / Store                                           Description
DS_LOAD_{B32,B64,B96,B128,U8,I8,U16,I16}               Load one value per thread into VGPRs; if signed, sign extend to
                                                       DWORD; zero extend if unsigned.
DS_LOAD_2ADDR_{B32,B64}                                Load two values at unique addresses.
DS_LOAD_2ADDR_STRIDE64_{B32,B64}                       Load 2 values at unique addresses; offset *= 64.
DS_STORE_{B8,B16,B32,B64,B96,B128}                     Store one value from VGPR to LDS.
DS_STORE_2ADDR_{B32,B64}                               Store two values.
DS_STORE_2ADDR_STRIDE64_{B32,B64}                      Store two values, offset *= 64.
DS_STOREXCHG_RTN_{B32,B64}                             Exchange GPR with LDS-memory.
DS_STOREXCHG_2ADDR_RTN_{B32,B64}                       Exchange two separate GPRs with LDS-memory.
DS_STOREXCHG_2ADDR_STRIDE64_RTN_{B32,B64} Exchange GPR with LDS-memory; offset *= 64.
                "D16 ops" - Load ops write only 16bits of VGPR, low or high; Store ops use 16bits of VGPR:
DS_STORE_{B8, B16}_D16_HI                              Store 8 or 16 bits using high 16 bits of VGPR.
DS_LOAD_{U8, I8, U16}_D16                              Load unsigned or signed 8 or 16 bits into low-half of VGPR
DS_LOAD_{U8, I8, U16}_D16_HI                           Load unsigned or signed 8 or 16 bits into high-half of VGPR
DS_LOAD_ADDTID_B32                                     Load and store B32, using thread-ID as part of the address-offset.
DS_STORE_ADDTID_B32
                                  Other non-load/store ops not considered "atomic ops"
DS_PERMUTE_B32                                         Forward & Backward permute. Does not write any LDS memory. See
DS_BPERMUTE_B32                                        LDS Lane-permute Ops for details.
DS_BPERMUTE_FI_B32
DS_SWIZZLE_B32                                         DWORD swizzle; no data is written to LDS memory.
DS_CONSUME                                             Subtract countBits(EXEC) from memory, return pre-op value.
DS_APPEND                                              Add countBits(EXEC) to memory, return pre-op value.
