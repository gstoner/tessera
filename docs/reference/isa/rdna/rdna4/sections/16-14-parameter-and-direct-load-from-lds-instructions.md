# 16.14. Parameter and Direct Load from LDS Instructions

> RDNA4 ISA — pages 584–584

16.14. Parameter and Direct Load from LDS Instructions
These instructions load data from LDS into a VGPR where the LDS address is derived from wave state and the
M0 register.

DS_PARAM_LOAD                                                                                                      0

Transfer parameter data from LDS to VGPRs and expand data in LDS using the NewPrimMask (provided in M0)
to place per-quad data into lanes 0-3 of each quad as follows:

{P0, P10, P20, 0.0}

This data may be extracted using DPP8 for interpolation operations. The V_INTERP_* instructions unpack data
automatically.

When loading FP16 parameters, two attributes are loaded into a single VGPR: Attribute 2*ATTR is loaded into
the low 16 bits and attribute 2*ATTR+1 is loaded into the high 16 bits.

This instruction runs in whole quad mode: if any pixel of a quad is active then all 4 pixels of that quad are
written. This is required for interpolation instructions to have all the parameter information available for the
quad.

DS_DIRECT_LOAD                                                                                                     1

Read a single 32-bit value from LDS to all lanes. A single DWORD is read from LDS memory at ADDR[M0[15:0]],
where M0[15:0] is a byte address and is dword-aligned. M0[18:16] specify the data type for the read and may be
0=UBYTE, 1=USHORT, 2=DWORD, 4=SBYTE, 5=SSHORT. See M0_FMT_LDS_DIRECT structure.

                 This instruction runs in whole quad mode: if any pixel of a quad is active then all 4 pixels of
                that quad are written.
