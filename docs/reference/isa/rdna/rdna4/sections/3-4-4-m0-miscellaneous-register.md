# 3.4.4. M0 : Miscellaneous Register

> RDNA4 ISA — pages 34–34

Field                       Bit     Description
                            Pos
SYS_PRIO                    11:10   Wave priority set at wave creation time. See S_SETPRIO instruction for details. 0 is
                                    lowest, 3 is highest priority.
USER_PRIO                   13:12   Wave’s priority set by shader program itself. See S_SETPRIO instruction for details.
HALT                        14      Wave is halted or scheduled to halt.
                                    HALT can be set by the host via wave-control messages, or by the shader. The HALT bit
                                    is ignored while in the trap handler (PRIV = 1). HALT is also ignored if a host-initiated
                                    trap is received (request to enter the trap handler).
SCRATCH_EN                  18      Indicate that the wave has scratch memory allocated. This bit gets set to 1 if the wave has
                                    SCRATCH_BASE initialized; otherwise is zero.

3.4.3. MODE register
Mode register fields can be read from, and written to, by the shader through scalar instructions.

                                                  Table 7. Mode Register Fields
Field                       Bit     Description
                            Pos
FP_ROUND                    3:0     Controls round modes for math operations
                                    [1:0] Single precision round mode
                                    [3:2] Double precision and half precision (FP16) round mode
                                    Round Modes: 0=nearest even, 1= +infinity, 2= -infinity, 3= toward zero
                                    Round mode affects float ops in VALU, but not LDS or memory.
FP_DENORM                   7:4     Controls whether floating point denormals are flushed or not.
                                    [5:4] Single precision denormal mode
                                    [7:6] Double precision and FP16 denormal mode
                                    Denormal modes: 2 bits = { allow_output_denorms, allow_input_denorms }
                                      0 = flush input and output denorms
                                      1 = allow input denorms, flush output denorms
                                      2 = flush input denorms, allow output denorms
                                      3 = allow input and output denorms
                                    Denorm mode affects float ops in: VALU, LDS-atomics (DS ops, not flat).
                                    Flat (including those serviced by LDS) and memory atomics ignore this mode and do not
                                    flush denorms.
FP16_OVFL                   23      If set, an overflowed FP16 VALU result is clamped to +/- MAX_FP16 regardless of round
                                    mode, while still preserving true INF values. (Inputs that are infinity may result in infinity,
                                    as does divide-by-zero).
SCALAR_PREFETCH_EN 24               1 = enable scalar prefetch instructions (both instruction & data); 0 = ignore them
                                    (S_NOP).
DISABLE_PERF                27      1 = temporarily disable performance counting for this wave.

3.4.4. M0 : Miscellaneous Register
There is one 32-bit M0 register per wave and is it used for:

                                                   Table 8. M0 Register Fields
Operation                   M0 Contents                             Notes
S/V_MOVREL                  GPR index                               See S_MOVREL and V_MOVREL instructions
LDS ADDTID                  { 16’h0, lds_offset[15:0] }             offset is in bytes, must be 4-byte aligned
