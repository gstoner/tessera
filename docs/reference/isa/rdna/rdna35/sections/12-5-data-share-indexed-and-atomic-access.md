# 12.5. Data Share Indexed and Atomic Access

> RDNA3.5 ISA — pages 136–138

to all active lanes in the wave. M0 provides the address and data type. LDS_DIRECT_LOAD uses EXEC per
quad, not per pixel: if any pixel in a quad is enabled then the data is written to all 4 pixels in the quad.
LDS_DIRECT_LOAD uses EXPcnt to track completion.

LDS_DIRECT_LOAD uses the same instruction format and fields as LDS_PARAM_LOAD. See Pixel Parameter
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

  Example:   LDS_DIRECT_LOAD   V4        // load the value from LDS-address in M0[15:0] to V4

Signed byte and short data is sign-extend to 32 bits before writing the result to a VGPR; unsigned byte and short
data is zero-extended to 32 bits before writing to a VGPR.

12.5. Data Share Indexed and Atomic Access
Both LDS and GDS can perform indexed and atomic data share operations. For brevity, "LDS" is used in the text
below and, except where noted, also applies to GDS.

Indexed and atomic operations supply a unique address per work-item from the VGPRs to the LDS, and supply
or return unique data per work-item back to VGPRs. Due to the internal banked structure of LDS, operations
can complete in as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending
upon the number of bank conflicts (addresses that map to the same memory bank).

Indexed operations are simple LDS load and store operations that read data from, and return data to, VGPRs.

Atomic operations are arithmetic operations that combine data from VGPRs and data in LDS, and write the
result back to LDS. Atomic operations have the option of returning the LDS "pre-op" value to VGPRs.

LDS Indexed and atomic instructions use LGKMcnt to track when they have completed. LGKMcnt is
incremented as each instruction is issued, and decremented when they have completed execution. LDS
instructions stay in-order with other LDS instructions from the same wave.

The table below lists and briefly describes the LDS instruction fields.

                                             Table 59. LDS Instruction Fields

Field       Size Description
OP          8    LDS opcode.
GDS         1    0 = LDS, 1 = GDS.
OFFSET0 8        Immediate address offset. Interpretation varies with opcode:
                 Instructions with one address:: combine the offset fields into a 16-bit unsigned byte offset: {offset1,
                 offset0}.
OFFSET1 8
                 Instructions that have 2 addresses (e.g. {LOAD, STORE, XCHG}_2ADDR):: use the offsets separately as 2 8-
                 bit unsigned offsets. Each offset is multiplied by 4 for 8, 16 and 32-bit data; multiplied by 8 for 64-bit data.
VDST        8    VGPR to which result is written: either from LDS-load or atomic return value.
ADDR        8    VGPR that supplies the byte address offset.
DATA0       8    VGPR that supplies first data source.
DATA1       8    VGPR that supplies second data source.
M0          16   Unsigned byte Offset[15:0] used for: ds_load_addtid_b32, ds_write_addtid_b32 and for GDS-base/size

The M0 register is not used for most LDS-indexed operations: only the "ADDTID" instructions read M0 and for
these it represents a byte address.

                                              Table 60. LDS Indexed Load/Store
Load / Store                                             Description
DS_LOAD_{B32,B64,B96,B128,U8,I8,U16,I16}                 Load one value per thread into VGPRs; if signed, sign extend to
                                                         DWORD; zero e xtend if unsigned.
DS_LOAD_2ADDR_{B32,B64}                                  Load two values at unique addresses.
DS_LOAD_2ADDR_STRIDE64_{B32,B64}                         Load 2 values at unique addresses; offset *= 64.
DS_STORE_{B32,B64,B96,B128,B8,B16}                       Store one value from VGPR to LDS.
DS_STORE_2ADDR_{B32,B64}                                 Store two values.
DS_STORE_2ADDR_STRIDE64_{B32,B64}                        Store two values, offset *= 64.
DS_STOREXCHG_RTN_{B32,B64}                               Exchange GPR with LDS-memory.
DS_STOREXCHG_2ADDR_RTN_{B32,B64}                         Exchange two separate GPRs with LDS-memory.
DS_STOREXCHG_2ADDR_STRIDE64_RTN_{B32,B64} Exchange GPR with LDS-memory; offset *= 64.
"D16 ops" - Load ops write only 16bits of VGPR, low or high; Store ops use 16bits of VGPR:
DS_STORE_{B8, B16}_D16_HI                                Store 8 or 16 bits using high 16 bits of VGPR.
DS_LOAD_{U8, I8, U16}_D16                                Load unsigned or signed 8 or 16 bits into low-half of VGPR
DS_LOAD_{U8, I8, U16}_D16_HI                             Load unsigned or signed 8 or 16 bits into high-half of VGPR
DS_PERMUTE_B32                                           Forward permute. Does not write any LDS memory. See LDS Lane-
                                                         permute Ops for details.
DS_BPERMUTE_B32                                          Backward permute. Does not write any LDS memory. See LDS Lane-
                                                         permute Ops for details.

Single Address Instructions

  LDS_Addr = LDS_BASE + VGPR[ADDR] + {InstOffset1,InstOffset0}

Double Address Instructions

  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + InstOffset0*ADJ +
  LDS_Addr1 = LDS_BASE + VGPR[ADDR] + InstOffset1*ADJ +
        Where ADJ = 4 for 8, 16 and 32-bit data types; and ADJ = 8 for 64-bit.

The double address instructions are: LOAD_2ADDR*, STORE_2ADDR*, and STOREXCHG_2ADDR_*. The
address comes from VGPR, and both VGPR[ADDR] and InstOffset are byte addresses. At the time of wave
creation, LDS_BASE is assigned to the physical LDS region owned by this wave or work-group.

DS_{LOAD,STORE}_ADDTID Addressing

  LDS_Addr = LDS_BASE + {InstOffset1, InstOffset0} + TID(0..63)*4 + M0
       Note: no part of the address comes from a VGPR.           M0 must be DWORD-aligned.

The "ADDTID" (add thread-id) is a separate form where the base address for the instruction is common to all
threads, but then each thread has a fixed offset added in based on its thread-ID within the wave. This can allow
a convenient way to quickly transfer data between VGPRs and LDS without having to use a VGPR to supply an
address.

LDS & GDS Opcodes
Instruction Fields: op, gds, offset0, offset1, vdst, addr, data0, data1
32-bit no return               32-bit with return                 64-bit no return           64-bit with return
ds_load_b{64,96,128}                                              ds_store_b{64,96,128}
ds_store_{b32,b16,b8}                                             ds_store_b64
ds_load_addtid_b32 (LDS        ds_permute_b32 (LDS only)
only)
ds_store_addtid_b32 (LDS       ds_bpermute_b32 (LDS only)
only)
ds_store_2addr_b32                                                ds_store_2addr_b64
ds_store_2addr_stride64_b3                                        ds_store_2addr_stride64_
2                                                                 b64
                               ds_load_{b32, u8,i8,u16,i16}                                  ds_load_b64
ds_store_b8_d16_hi             ds_load_2addr_b32                                             ds_load_2addr_b64
ds_store_b16_d16_hi            ds_load_2addr_stride64_b32                                    ds_load_2addr_stride64_b64
ds_load_u8_d16                 ds_consume
ds_load_u8_d16_hi              ds_append                                                     ds_condxchg32_rtn_b64
ds_load_i8_d16
ds_load_i8_d16_hi              ds_swizzle_b32 (LDS only)
ds_load_u16_d16
ds_load_u16_d16_hi
                                                       GDS-only Opcodes
                               ds_ordered_count
                               gws_init
                               gws_sema_v
                               gws_sema_bf
                               gws_sema_p
                               gws_barrier
                               gws_sema_release_all
                               ds_add_gs_reg_rtn
                               ds_sub_gs_reg_rtn
