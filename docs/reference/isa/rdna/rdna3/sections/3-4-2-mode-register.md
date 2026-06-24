# 3.4.2. Mode register

> RDNA3 ISA — pages 31–31

3.4.2. Mode register
Mode register fields can be read from, and written to, by the shader through scalar instructions. The table
below describes the mode register fields.

                                                  Table 6. Mode Register Fields
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
                                    Denorm mode affects float ops in: VALU, LDS, and VMEM atomics.
                                    Texture/Buffer/Flat considers only bits 4 and 6 (allowing mode control over input-denorm
                                    flushing, and not flushing output denorms), while LDS uses all bits for DS ops (but not for
                                    FLAT).
DX10_CLAMP                  8       Used by the vector ALU to force DX10 style treatment of NaN’s. When set, clamp NaN to
                                    zero, otherwise pass NaN thru and also suppress all VALU exceptions. The clamping only
                                    occurs when the instruction has the CLAMP bit set to 1, but exceptions are suppressed
                                    when DX10_CLAMP==1.
IEEE                        9       IEEE==0: IEEE-754-1985/DX10 behavior for Min and Max, pass signaling NaN.
                                    IEEE==1: IEEE-754-2008 behavior for Min and Max, quiet signaling NaN.
                                    When set to 1, floating point opcodes that support exception flag gathering quiet and
                                    propagate signaling NaN inputs per IEEE 754-2008. Min_f32/f64 and Max_f32/f64 become
                                    IEEE 754-2008 compliant due to signaling NaN propagation and quieting. When set to 1,
                                    MAX performs a ">" compare, but when set to zero (directX mode/IEEE 754-1985 mode)
                                    MAX performs a ">=" compare. This only affects results for +/-0 and input denormals
                                    which are flushed to zero.
LOD_CLAMPED                 10      Sticky status bit - indicates that one or more texture accesses had their LOD clamped.
TRAP_AFTER_ INST            11      Forces the wave to jump to the exception handler after each instruction is executed (but
                                    not after ENDPGM). Only works if TRAP_EN = 1.
EXCP_EN                     21:12   Enable mask for exceptions. Enabled means if the exception occurs and if TRAP_EN==1, a
                                    trap may be taken.

                                       [12] : invalid
                                       [13] : inputDenormal
                                       [14] : float_div0
                                       [15] : overflow
                                       [16] : underflow
                                       [17] : inexact
                                       [18] : int_div0
                                       [19] : addr_watch - take exception when TC sees wave access an "address of interest"
                                       [21] : trap on wave end - h/w clears this upon entering trap handler for end-of-wave
