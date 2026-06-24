# Chapter 13. Float Memory Atomics

> RDNA3 ISA — pages 141–141

Chapter 13. Float Memory Atomics
Floating point atomics can be issued as LDS, Buffer, and Flat/Global/Scratch instructions.

13.1. Rounding
LDS and Memory atomics have the rounding mode for float-atomic-add fixed at "round to nearest even". The
MODE.round bits are ignored.

13.2. Denormals
When these operate on floating point data, there is the possibility of the data containing denormal numbers, or
the operation producing a denormal. The floating point atomic instructions have the option of passing
denormal values through, or flushing them to zero.

LDS instructions allow denormals to be passed through or flushed to zero based on the MODE.denormal wave-
state register. As with VALU ops, "denorm_single" affects F32 ops and "denorm_double" affects F64. LDS
instructions use both FP_DENORM bits (allow_input_denormal, allow_output_denormal) to control flushing of
inputs and outputs separately.

  • Float 32 bit adder uses both input and output denorm flush controls from MODE
  • Float CMP, MIN and MAX use only the "input denormal" flushing control
      ◦ Each input to the comparisons flushes the mantissa of both operands to zero before the compare if the
        exponent is zero and the flush denorm control is active. For Min and Max the actual result returned is
        the selected non-flushed input.
      ◦ CompareStore ("compare swap") flushes the result when input denormal flushing occurs.

                                             Cache Atomic Float Denormal
                                             (Buffer, Flat, Global, Scratch)
                        Min/Max_F32                               Mode
                        CmpStore_F32, _F64                        Mode
                        Add_F32                                   Flush
                                                   LDS Float Atomics
                        Min/Max_F32                               Mode
                        CmpStore_F32, _F64                        Mode
                        Add_F32                                   Mode
                        Min/Max_F64                               Mode

  • "Flush" = flush all input denorm
  • "No Flush" = don’t flush input denorm
  • "Mode" = denormal flush controlled by bit from shader’s "MODE . fp_denorm" register

Note that MIN and MAX when flushing denormals only do it for the comparison, but the result is an
unmodified copy of one of the sources. CompareStore ("compare swap") flushes the result when input
denormal flushing occurs.

Memory Atomics:
