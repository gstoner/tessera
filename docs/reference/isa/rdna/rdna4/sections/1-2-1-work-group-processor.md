# 1.2.1. Work-Group Processor

> RDNA4 ISA — pages 16–16

                              Figure 1. AMD RDNA4 Generation Series Block Diagram

The RDNA4 device includes a data-parallel processor array, a command processor, a memory controller, and
other logic (not shown). The command processor reads commands that the host has written to memory-
mapped registers in the system-memory address space. The command processor sends hardware-generated
interrupts to the host when the command is completed. The memory controller has direct access to all device
memory and the host-specified areas of system memory. To satisfy read and write requests, the memory
controller performs the functions of a direct-memory access (DMA) controller, including computing memory-
address offsets based on the format of the requested data in memory.

In the RDNA4 environment, a complete application includes two parts:
  • a program running on the host processor, and
  • programs, called shader programs or kernels, running on the RDNA4 processor.

The RDNA4 programs are controlled by a driver running on the host that:
  • Sets internal base-address and other configuration registers,
  • Specifies the data domain on which the RDNA4 processor is to operate,
  • Invalidates and flushes caches on the RDNA4 processor, and
  • Causes the RDNA4 processor to begin execution of a program.

1.2.1. Work-Group Processor
The processor array is the heart of the RDNA4 processor. The array is organized as a set of Work-Group
Processor (WGP) pipelines, each independent from the others, that operate in parallel on streams of floating-
point or integer data.

The WGP processor pipelines can process data or, through the memory controller, transfer data to, or from,
memory. Computation in a WGP processor pipeline can be made conditional. Outputs written to memory can
also be made conditional.
