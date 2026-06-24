# Chapter 13. Float Memory Atomics

> RDNA4 ISA — pages 166–166

Chapter 13. Float Memory Atomics
Floating point atomics can be issued as LDS, Buffer, and Flat/Global/Scratch instructions.

Memory atomics do not report any numeric exceptions (e.g. signaling NaN).

13.1. Rounding
LDS and Memory atomics have the rounding mode for float-atomic-add fixed at "round to nearest even". The
MODE.round bits are ignored for atomics.

13.2. Denormals
When these operate on floating point data, there is the possibility of the data containing denormal numbers, or
the operation producing a denormal. The floating point atomic instructions have the option of passing
denormal values through, or flushing them to zero.

LDS-indexed instructions allow denormals to be passed through or flushed to zero based on the
MODE.denormal wave-state register. As with VALU ops, "denorm_single" affects F32 ops and "denorm_double"
affects F64 and F16. LDS instructions use both FP_DENORM bits (allow_input_denormal,
allow_output_denormal) to control flushing of inputs and outputs separately. Using MODE allows LDS and
VALU to produce identical results for float ops.

  • Float 16 and 32 bit adder uses both input and output denorm flush controls from MODE
  • Float 64 bit adder does not flush denorms
  • Float MIN and MAX use only the "input denormal" flushing control
      ◦ Each input to the comparisons flushes the mantissa of both operands to zero before the compare if the
        exponent is zero and the flush denorm control is active.
      ◦ Min and Max flush input denormals first when enabled, then perform the Min/Max operation on the
        flushed denormals
      ◦ Float CompareStore ("compare swap") flushes both input and output denormals based on the "flush
        input denormals" control.

                                 LDS (DS)            LDS (FLAT)            L2 Cache                Data Fabric
Operation               Flush          Flush    Flush     Flush     Flush       Flush        Flush      Flush
                        Input          Output   Input     Output    Input       Output       Input      Output
                        Denorm         Denorm   Denorm    Denorm    Denorm      Denorm       Denorm     Denorm
ADD_F32                 Mode.I         Mode.O   NoFlush   NoFlush   NoFlush     NoFlush      Reg        Reg
PK_ADD_F16 / _BF16      Mode.I         Mode.O   NoFlush   NoFlush   NoFlush     NoFlush      Reg        Reg
Compare-Store /         Mode.I         Mode.I   NoFlush   NoFlush   NoFlush     NoFlush      NoFlush    NoFlush
Compare-Swap

  • NoFlush: Do No flush denorms
  • Mode.I : flushing is controlled by MODE.denorm (input denormal control)
  • Mode.O : flushing is controlled by MODE.denorm (output denormal control)
  • Reg : denormal flushing is controlled by a config register (which should be set to "No flush")
