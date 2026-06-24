# Chapter 2. Shader Concepts

> RDNA3 ISA — pages 18–18

Chapter 2. Shader Concepts
RDNA3 shader programs (kernels) are programs executed by the GPU processor. Conceptually, the shader
program is executed independently on every work-item, but in reality the processor groups up to 32 or 64
work-items into a wave, which executes the shader program on all 32 or 64 work-items in one pass.

The RDNA3 processor consists primarily of:
  • A scalar ALU, which operates on one value per wave (common to all work-items)
  • A vector ALU, which operates on unique values per work-item
  • Local data storage, which allows work-items within a work-group to communicate and share data
  • Scalar memory, which can transfer data between SGPRs and memory through a cache
  • Vector memory, which can transfer data between VGPRs and memory, including sampling texture maps
  • Exports which transfer data from the shader to dedicated rendering hardware

Program control flow is handled using scalar ALU instructions. This includes if/else, branches and looping.
Scalar ALU (SALU) and memory instructions work on an entire wave and operate on up to two SGPRs, as well
as literal constants.

Vector memory and ALU instructions operate on all work-items in the wave at one time. In order to support
branching and conditional execute, every wave has an EXECute mask that determines which work-items are
active at that moment, and which are dormant. Active work-items execute the vector instruction, and dormant
ones treat the instruction as a NOP. The EXEC mask can be written at any time by Scalar ALU instructions or
VALU comparisons.

Vector ALU instructions can typically take up to three arguments, which can come from VGPRs, SGPRs, or
literal constants that are part of the instruction stream. They operate on all work-items enabled by the EXEC
mask. Vector compare and add-with-carry-out return a bit-per-work-item mask back to the SGPRs to indicate,
per work-item, which had a "true" result from the compare or generated a carry-out.

Vector memory instructions transfer data between VGPRs and memory. Each work-item supplies its own
memory address and supplies or receives unique data. These instructions are also subject to the EXEC mask.

2.1. Wave32 and Wave64
The shader supports both waves of 32 work-items ("wave32") and waves of 64 work-items ("wave64").

Both wave sizes are supported for all operations, but shader programs must be compiled for and run as a
particular wave size, regardless of how many work-items are active in any given wave.

Wave32 waves issue each instruction at most once. Wave64 waves typically issue each instruction twice: once
for the low half (work-items 31-0) and then again for the high half (work-items 63-32). This occurs only for
VALU and VMEM (LDS, texture, buffer, flat) instructions; scalar ALU and memory as well as branch and
messages are issued only once regardless of the wave size. Export requests also issue just once regardless of
wave size. It is possible that instructions from other waves may be executed in between the low and high half
of a given wave’s instructions.

Hardware may choose to skip either half if the EXEC mask for that half is all zeros, but does not skip both
halves for VMEM instructions as that would confuse the outstanding-memory-instruction counters, unless
