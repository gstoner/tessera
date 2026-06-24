# 1.2.1. Work-group Processor (WGP)

> RDNA3 ISA — pages 15–15

The RDNA3 device includes a data-parallel processor array, a command processor, a memory controller, and
other logic (not shown). The command processor reads commands that the host has written to memory-
mapped registers in the system-memory address space. The command processor sends hardware-generated
interrupts to the host when the command is completed. The memory controller has direct access to all device
memory and the host-specified areas of system memory. To satisfy read and write requests, the memory
controller performs the functions of a direct-memory access (DMA) controller, including computing memory-
address offsets based on the format of the requested data in memory.

In the RDNA3 environment, a complete application includes two parts:
  • a program running on the host processor, and
  • programs, called shader programs or kernels, running on the RDNA3 processor.

The RDNA3 programs are controlled by a driver running on the host that:
  • sets internal base-address and other configuration registers,
  • specifies the data domain on which the GPU is to operate,
  • invalidates and flushes caches on the GPU, and
  • causes the GPU to begin execution of a program.

1.2.1. Work-group Processor (WGP)
The processor array is the heart of the GPU. The array is organized as a set of work-group processor (WGP)
pipelines, each independent from the others, that operate in parallel on streams of floating-point or integer
data. The work-group processor pipelines can process data or, through the memory controller, transfer data to,
or from, memory. Computation in a work-group processor pipeline can be made conditional. Outputs written
to memory can also be made conditional.

When it receives a request, the work-group processor pipeline loads instructions and data from memory,
begins execution, and continues until the end of the kernel. As kernels are running, the GPU hardware
automatically fetches instructions from memory into on-chip caches; software plays no role in this. Kernels
can load data from off-chip memory into on-chip general-purpose registers (GPRs) and caches.

The GPU devices can detect floating point exceptions and can generate interrupts to the host. In particular,
they detect IEEE-754 floating-point exceptions in hardware; these can be recorded for post-execution analysis.

The GPU hides memory latency by keeping track of potentially hundreds of work-items in various stages of
execution, and by overlapping compute operations with memory-access operations.

1.2.2. Data Sharing
The processors may share data between different work-items. Data sharing can boost performance. The figure
below shows the memory hierarchy that is available to each work-item. The actual number of GPRs may differ
from what is shown in the image below.
