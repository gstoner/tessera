# 3.4.7. FLAT_SCRATCH

> RDNA3 ISA — pages 33–33

  Arithmetic operations: 1 = carry out
  Bit/logical operations: 1 = result was not zero
  Move: does not alter SCC

The SCC can be used as the carry-in for extended-precision integer arithmetic, as well as the selector for
conditional moves and branches.

3.4.6. Vector Compares: VCC and VCCZ
Vector ALU comparison instructions (V_CMP) compare two values and return a bit-mask of the result, where
each bit represents one lane (work-item) where: 1= pass, 0 = fail. This result mask is the Vector Condition Code
(VCC). VCC is also set for selected integer ALU operations (carry-out).

These instructions write this mask either to VCC, an SGPR or to EXEC, but do not write to both EXEC and
SGPRs. Wave32 writes only the low 32 bits of VCC, EXEC or a single SGPR; Wave64 writes 64-bits of VCC, EXEC
or an aligned pair of SGPRs.

Whenever any instruction writes a value to VCC, the hardware automatically updates a "VCC summary" bit
called VCCZ. This bit indicates whether or not the entire VCC mask is zero for the current wave-size. Wave32
ignores VCC[63:32] and only bits[31:0] contribute to VCCZ. This is useful for early-exit branch tests. VCC is also set
for certain integer ALU operations (carry-out).

The EXEC mask determines which threads execute an instruction. The VCC indicates which executing threads
passed the conditional test, or which threads generated a carry-out from an integer add or subtract.

  S_MOV_B64       EXEC, 0x00000001     // set just one thread active; others are inactive
  V_CMP_EQ_B32    VCC, V0, V0          // compare (V0 == V0) and write result to VCC (all bits in VCC are
  updated)

                 VCC physically resides in the SGPR register file in a specific pair of SGPRs, so when an
                instruction sources VCC, that counts against the limit on the total number of SGPRs that can
                 be sourced for a given instruction.

Wave32 waves may use any SGPR for mask/carry/borrow operations, but may not use VCC_HI or EXEC_HI.

3.4.7. FLAT_SCRATCH
FLAT_SCRATCH is a 64-bit register that holds a pointer to the base of scratch memory for this wave. For waves
that have scratch space allocated, wave-launch hardware initializes the FLAT_SCRATCH register with the
scratch base address unique to this wave. This register is read-only, except while in the trap handler where it is
writable. The value is a byte address and must be 256byte aligned. If the wave has no scratch space allocated,
then reading FLAT_SCRATCH returns zero.

The value for FLAT_SCRATCH is computed in hardware and initialized for any wave that has scratch space
allocated:
    scratch_base = scratch_base[63:0] + spi_scratch_offset[31:0]
