# Scheduling algorithm

> Micro Engine Scheduler ISA — pages 10–10

Scheduling algorithm
             Here are queue terminologies with descriptions to assist in understanding the queue state
             transitions, before describing the scheduling algorithm.

             •     User queue
                   Represents a linear command stream of draws or dispatches from an application. It would
                   be analogous to a thread in the CPU world. There we few memory resources allocated for
                   user queue such as ring buffer where command packets are submitted by the application
                   and a memory to save the HW execution state of the queue when it is preempted. A user
                   queue does not execute on its own. It needs to be mapped onto a HW queue for it to
                   execute.

             •     Hardware queue
                   A hardware descriptor that holds the user queue state (for e.g. ring buffer address, read,
                   write pointers etc). A hardware queue could be in a mapped or unmapped state. And a
                   mapped queue could be in a connected or a disconnected state.

             •     Queue mapping/un-mapping
                   Mapping is an act of loading a user queue state onto a hardware queue. And un mapping
                   is an act of moving the queue state from a hardware queue descriptor to memory. A
                   hardware queue can only be unmapped after preemption.

             •     Connected queue
                   Hardware queue that is selected by queue manager to run on the 3d/CS complex.

April 2024                                                                                                10
